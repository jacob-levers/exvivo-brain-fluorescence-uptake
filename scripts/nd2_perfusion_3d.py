#!/usr/bin/env python
"""
nd2_perfusion_3d.py -- 3D animation of a fluorescent drug penetrating tissue,
from a multi-channel, multi-timepoint Nikon ND2 z-stack timelapse.

Written to be re-run on other recordings of the same kind: every number it
depends on is read from the file, not hard-coded.

Stages (each caches its output and is independently re-runnable):

  probe     print the ND2 metadata and stop
  cache     one lazy streaming pass over the ND2 -> XY-binned float32 cache
  register  rigid 3D translation registration on the STRUCTURAL channel only,
            written as a CSV of per-timepoint shifts + a diagnostic plot
  preview   render single frames at chosen timepoints
  render    render the full MP4
  all       cache -> register -> render

Invariants this script maintains, by construction:
  * the transform is computed from --struct-channel and applied to every
    channel;  --signal-channel is never used to drive registration
  * the ND2 is read lazily, one timepoint at a time, via dask
  * the volume is NOT resampled to isotropic; the renderer marches rays through
    real micron coordinates using the true anisotropic voxel spacing
  * one intensity window, chosen once, is held fixed for every frame
  * no bleach correction and no per-frame normalisation is applied to the
    signal channel; the background subtracted is constant in time

Example
-------
  python nd2_perfusion_3d.py all --nd2 "1-AMA 60.nd2"
  python nd2_perfusion_3d.py render --nd2 "1-AMA 60.nd2" --rotate 360 --out spin.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# array backend: CuPy when available, NumPy otherwise
# --------------------------------------------------------------------------

_GPU = None


def gpu_available() -> bool:
    global _GPU
    if _GPU is None:
        try:
            import cupy
            cupy.zeros(1)
            _GPU = True
        except Exception:
            _GPU = False
    return _GPU


def backend(use_gpu: bool):
    """Return (array_module, ndimage_module, to_host_fn)."""
    if use_gpu and gpu_available():
        import cupy as cp
        from cupyx.scipy import ndimage as cnd
        return cp, cnd, (lambda a: cp.asnumpy(a))
    from scipy import ndimage as snd
    return np, snd, (lambda a: np.asarray(a))


def to_host(a):
    if type(a).__module__.startswith("cupy"):
        import cupy as cp
        return cp.asnumpy(a)
    return np.asarray(a)


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

def read_meta(nd2_path: str) -> dict:
    """Pull every number we depend on straight out of the file."""
    import nd2

    with nd2.ND2File(nd2_path) as f:
        sizes = dict(f.sizes)
        vox = f.voxel_size()                 # (x, y, z) in um
        axes = list(sizes.keys())            # dask axis order

        channels = []
        for i, c in enumerate(f.metadata.channels):
            channels.append(dict(
                index=i,
                name=c.channel.name,
                emission_nm=c.channel.emissionLambdaNm,
                excitation_nm=c.channel.excitationLambdaNm,
                objective_mag=c.microscope.objectiveMagnification,
                na=c.microscope.objectiveNumericalAperture,
            ))

        # prefer the ZStackLoop's own step over voxel_size()
        z_step = float(vox.z)
        n_t_loop = None
        period_ms = None
        for lp in f.experiment:
            tn = type(lp).__name__
            if "ZStack" in tn:
                try:
                    z_step = float(lp.parameters.stepUm)
                except Exception:
                    pass
            if "Time" in tn:
                n_t_loop = int(lp.count)
                try:
                    period_ms = float(lp.parameters.periods[0].periodMs)
                except Exception:
                    period_ms = getattr(lp.parameters, "periodMs", None)

        # real per-timepoint wall clock from the frame event log
        times_s = None
        stack_dur_s = None
        stage = {}
        try:
            ev = f.events(orient="list")
            t = np.asarray(ev["Time [s]"], float)
            ti = np.asarray(ev["T Index"], int)
            nT = int(sizes.get("T", 1))
            t0 = np.array([t[ti == k].min() for k in range(nT)])
            stack_dur_s = float(np.median(
                [t[ti == k].max() - t[ti == k].min() for k in range(nT)]))
            times_s = (t0 - t0[0]).tolist()
            for key, short in [("X Coord [\u00b5m]", "x"),
                               ("Y Coord [\u00b5m]", "y"),
                               ("Z Coord [\u00b5m]", "z")]:
                if key in ev:
                    a = np.asarray(ev[key], float)
                    aT = np.array([a[ti == k][0] for k in range(nT)])
                    stage[short] = dict(range_um=float(aT.max() - aT.min()),
                                        first=float(aT[0]), last=float(aT[-1]))
        except Exception as e:                       # pragma: no cover
            print(f"  (warning: could not read event log: {e})")

        text = f.text_info or {}

    return dict(
        path=os.path.abspath(nd2_path),
        sizes=sizes, axes=axes,
        dx_um=float(vox.x), dy_um=float(vox.y),
        dz_um=float(z_step), dz_um_voxelsize=float(vox.z),
        channels=channels,
        time_loop_count=n_t_loop, time_period_ms=period_ms,
        times_s=times_s, stack_duration_s=stack_dur_s,
        stage_um=stage,
        date=text.get("date"), optics=text.get("optics"),
    )


def print_meta(meta: dict) -> None:
    s = meta["sizes"]
    print("=" * 76)
    print("ND2 METADATA  --", os.path.basename(meta["path"]))
    print("=" * 76)
    print(f"  axis order (dask) : {meta['axes']}")
    print(f"  sizes             : {s}")
    nx, ny, nz = s.get("X"), s.get("Y"), s.get("Z", 1)
    print(f"  pixel size        : {meta['dx_um']:.4f} x {meta['dy_um']:.4f} um")
    print(f"  field of view     : {nx * meta['dx_um']:.2f} x {ny * meta['dy_um']:.2f} um")
    print(f"  z step            : {meta['dz_um']:.4f} um "
          f"(voxel_size reports {meta['dz_um_voxelsize']:.4f})")
    print(f"  z range           : {(nz - 1) * meta['dz_um']:.2f} um over {nz} planes")
    print(f"  ANISOTROPY        : {meta['dz_um'] / meta['dx_um']:.1f} : 1  (z : xy)")
    if meta["times_s"]:
        t = np.asarray(meta["times_s"])
        d = np.diff(t)
        print(f"  timepoints        : {len(t)}")
        print(f"  frame interval    : median {np.median(d):.2f} s "
              f"(min {d.min():.2f}, max {d.max():.2f})")
        print(f"  total duration    : {t[-1]:.1f} s = {t[-1] / 60:.2f} min")
        print(f"  z-stack dwell     : {meta['stack_duration_s']:.2f} s per volume")
    if meta.get("time_period_ms"):
        print(f"  programmed period : {meta['time_period_ms'] / 1000:.2f} s")
    print("  channels:")
    for c in meta["channels"]:
        print(f"    c{c['index']}  {c['name']!r}  ex={c['excitation_nm']}  "
              f"em={c['emission_nm']}")
    if meta["stage_um"]:
        print("  stage travel over the recording (sample motion is NOT this):")
        for k, v in meta["stage_um"].items():
            print(f"    {k}: {v['range_um']:.3f} um")
    print(f"  objective         : {meta.get('optics')}")
    print(f"  acquired          : {meta.get('date')}")
    print()


# --------------------------------------------------------------------------
# stage: cache
# --------------------------------------------------------------------------

def cache_dir_of(args) -> Path:
    """Where the binned cache lives.  Kept separate from --outdir so several
    analyses (different channels, settings) can share one expensive cache."""
    return Path(args.cache_dir or args.outdir)


def cache_paths(outdir, bin_xy: int):
    outdir = Path(outdir)
    return outdir / f"cache_bin{bin_xy}.npy", outdir / f"cache_bin{bin_xy}_meta.json"


def stage_cache(args, meta: dict) -> None:
    """One lazy streaming pass over the ND2 -> XY-binned float32 cache.

    The 40 GB file is never held in memory: nd2 hands us a dask array and we
    pull exactly one timepoint (all Z, all C) at a time, block-mean it in XY,
    and write it straight to a memmap.

    Binning in XY rather than interpolating Z is deliberate: with a z step of
    tens of microns, resampling to isotropic would mean throwing away almost
    all of the lateral resolution.
    """
    import nd2

    outdir = cache_dir_of(args)
    outdir.mkdir(parents=True, exist_ok=True)
    cpath, mpath = cache_paths(outdir, args.bin)

    if cpath.exists() and mpath.exists() and not args.force:
        print(f"[cache] {cpath} exists -- skipping (--force to rebuild)")
        return

    s = meta["sizes"]
    nT, nZ, nC = s.get("T", 1), s.get("Z", 1), s.get("C", 1)
    nY, nX = s["Y"], s["X"]
    b = args.bin
    Yb, Xb = nY // b, nX // b
    cropY, cropX = Yb * b, Xb * b

    print(f"[cache] {nT}T x {nC}C x {nZ}Z x {nY}x{nX} -> bin {b} -> {Yb}x{Xb}")
    print(f"[cache] writing {cpath} ({nT * nC * nZ * Yb * Xb * 4 / 1e9:.2f} GB float32)")

    arr = np.lib.format.open_memmap(cpath, mode="w+", dtype=np.float32,
                                    shape=(nT, nC, nZ, Yb, Xb))

    with nd2.ND2File(args.nd2) as f:
        dk = f.to_dask()
        ax = meta["axes"]
        iT = ax.index("T")
        rem = [a for i, a in enumerate(ax) if i != iT]
        order = [rem.index("C"), rem.index("Z"), rem.index("Y"), rem.index("X")]

        t0 = time.time()
        for t in range(nT):
            vol = np.asarray(dk[t])                       # e.g. (Z, C, Y, X)
            vol = np.transpose(vol, order).astype(np.float32)   # -> (C, Z, Y, X)
            v = vol[:, :, :cropY, :cropX]
            arr[t] = v.reshape(nC, nZ, Yb, b, Xb, b).mean(axis=(3, 5))
            if t % 10 == 0 or t == nT - 1:
                el = time.time() - t0
                print(f"[cache]  t={t + 1}/{nT}  {el:6.1f}s elapsed, "
                      f"~{el / (t + 1) * (nT - t - 1):5.1f}s left", flush=True)

    arr.flush()
    del arr

    cmeta = dict(meta)
    cmeta.update(bin_xy=b, dx_um=meta["dx_um"] * b, dy_um=meta["dy_um"] * b,
                 cache_shape=[nT, nC, nZ, Yb, Xb])
    mpath.write_text(json.dumps(cmeta, indent=1))
    print(f"[cache] done -> {cpath}")


def load_cache(args):
    cpath, mpath = cache_paths(cache_dir_of(args), args.bin)
    if not cpath.exists():
        sys.exit(f"cache missing: {cpath}\nRun the 'cache' stage first.")
    return np.load(cpath, mmap_mode="r"), json.loads(mpath.read_text())


# --------------------------------------------------------------------------
# background
# --------------------------------------------------------------------------

def compute_background(arr, cmeta, args):
    """Fixed, time-invariant, per-channel per-z background image.

    Estimated from the first --baseline-n timepoints (pre-drug) and held
    constant for the whole series.  It removes camera offset, vignetting and
    the spinning-disk out-of-focus haze without touching the time course:
    because it does not vary with t it cannot flatten the drug's rise.

    It lives in the CAMERA frame, so it is subtracted before the registration
    shift is applied.
    """
    from scipy.ndimage import gaussian_filter

    mode = getattr(args, "bg_mode", "image")
    bpath = (cache_dir_of(args) /
             f"background_bin{args.bin}_{mode}_n{args.baseline_n}"
             f"_s{args.bg_sigma:g}_p{getattr(args, 'bg_pct', 25):g}.npy")
    if bpath.exists() and not args.force_bg:
        return np.load(bpath)

    nT, nC, nZ = arr.shape[:3]
    n = min(args.baseline_n, nT)
    base = np.asarray(arr[:n]).mean(axis=0)              # (C, Z, Y, X)
    bg = np.empty_like(base)

    if mode == "image":
        # Smoothed baseline IMAGE.  Removes offset, vignette and haze -- and
        # also whatever sample signal was already present at t=0, so what is
        # rendered is the CHANGE since the recording started.  Right when the
        # signal channel is genuinely empty at t=0.
        print(f"[bg] fixed background image from first {n} timepoints, "
              f"gaussian sigma {args.bg_sigma} binned px")
        for c in range(nC):
            for z in range(nZ):
                bg[c, z] = gaussian_filter(base[c, z], args.bg_sigma)
    elif mode == "plane-const":
        # One constant per (channel, z), taken from a low percentile of the
        # baseline -- i.e. the surrounding medium.  Removes camera offset and
        # bath level while LEAVING the tissue signal intact, so a sample that
        # already contains signal at t=0 still shows it.
        print(f"[bg] fixed per-plane constant = p{args.bg_pct} of the first "
              f"{n} timepoints (bath level; tissue signal preserved)")
        for c in range(nC):
            for z in range(nZ):
                bg[c, z] = np.percentile(base[c, z], args.bg_pct)
    else:
        raise SystemExit(f"unknown --bg-mode {mode!r}")

    np.save(bpath, bg)
    return bg


# --------------------------------------------------------------------------
# stage: register
# --------------------------------------------------------------------------

def _prep_for_reg(v, sigma_hp, xp, ndi):
    """High-pass and window a volume so phase correlation locks onto the sample
    rather than the stationary vignette / dish edge."""
    v = v.astype(xp.float32)
    if sigma_hp > 0:
        v = v - ndi.gaussian_filter(v, (0, sigma_hp, sigma_hp))
    v = xp.clip(v, 0, None)
    nz, ny, nx = v.shape
    wy = xp.hanning(ny).astype(xp.float32)
    wx = xp.hanning(nx).astype(xp.float32)
    return v * wy[None, :, None] * wx[None, None, :]


def _pcc(ref, mov, upsample):
    """3D phase cross-correlation -> shift to apply to `mov` to match `ref`."""
    from skimage.registration import phase_cross_correlation
    ref, mov = to_host(ref), to_host(mov)
    out = phase_cross_correlation(ref, mov, upsample_factor=upsample,
                                  normalization="phase")
    shift = out[0] if isinstance(out, tuple) else out
    return np.asarray(shift, float)


def find_roi(arr, bg, cs, args, xp, ndi, n_probe=12):
    """Bounding box of the structural channel's own signal.

    Derived from --struct-channel only, so the signal channel takes no part in
    it.  Cropping to the sample keeps the stationary dish edge and the vignette
    out of the correlation.
    """
    nT, nC, nZ, Yb, Xb = arr.shape
    idx = np.linspace(0, nT - 1, min(n_probe, nT)).astype(int)
    acc = None
    for t in idx:
        m = np.asarray(arr[t, cs]).max(axis=0) - bg[cs].max(axis=0)
        acc = m if acc is None else np.maximum(acc, m)
    thr = np.percentile(acc, args.roi_pct)
    mask = acc > thr
    from scipy import ndimage as sndi
    mask = sndi.binary_closing(mask, iterations=3)
    lbl, n = sndi.label(mask)
    if n:                       # keep the largest connected blob
        sizes = sndi.sum(mask, lbl, range(1, n + 1))
        mask = lbl == (int(np.argmax(sizes)) + 1)
    mask = sndi.binary_dilation(mask, iterations=args.roi_dilate)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, Yb, 0, Xb
    # The ROI must be able to hold the sample at every position it visits, plus
    # the correlation search range -- otherwise a large displacement pushes the
    # sample out of the crop and the correlation has nothing to lock onto.
    pad = int(max(args.roi_pad, args.reg_bound + 20))
    return (max(0, ys.min() - pad), min(Yb, ys.max() + pad),
            max(0, xs.min() - pad), min(Xb, xs.max() + pad))


def reg_feature(arr, bg, t, c, roi, args, xp, ndi, clipped=True):
    """2D feature image the lateral registration is computed from.

    MIP over z, background-subtracted, cropped to the sample, high-passed, and
    optionally winsorised.

    Two variants are needed, because they are good at different things.  In a
    sparsely-labelled structural channel a handful of very bright puncta
    dominate the correlation.  Those puncta are the only feature strong enough
    to bridge a large single-frame displacement -- but they also brighten, dim
    and drift through focus, so over a long run their apparent motion is not
    the tissue's motion and they inflate the measured slow drift.  Winsorising
    removes that inflation and, with it, the ability to catch the jump.
    """
    y0, y1, x0, x1 = roi
    v = xp.asarray(np.asarray(arr[t, c]) - bg[c], dtype=xp.float32)
    p = v.max(axis=0)[y0:y1, x0:x1]
    if args.reg_hp_sigma > 0:
        p = p - ndi.gaussian_filter(p, args.reg_hp_sigma)
    p = xp.clip(p, 0, None)
    if clipped and args.reg_clip_pct and args.reg_clip_pct < 100:
        p = xp.minimum(p, xp.percentile(p, args.reg_clip_pct))
    w = xp.outer(xp.hanning(p.shape[0]),
                 xp.hanning(p.shape[1])).astype(xp.float32)
    return p * w


def z_profile(arr, bg, t, c, roi, xp, ndi):
    """Axial intensity profile of the sample, for the (coarse) z estimate."""
    y0, y1, x0, x1 = roi
    v = xp.asarray(np.asarray(arr[t, c]) - bg[c], dtype=xp.float32)
    p = v[:, y0:y1, x0:x1]
    p = xp.clip(p, 0, None).mean(axis=(1, 2))
    return p - p.mean()


def stage_register(args, cmeta, arr):
    """Rigid 3D translation registration, computed on the structural channel."""
    spath = Path(args.outdir) / "shifts.csv"
    if spath.exists() and not args.force_reg:
        print(f"[reg] {spath} exists -- loading (--force-reg to recompute)")
        return np.genfromtxt(spath, delimiter=",", names=True)

    from pcc_bounded import pcc_bounded

    xp, ndi, _ = backend(args.gpu)
    nT, nC, nZ, Yb, Xb = arr.shape
    cs = args.struct_channel
    dz, dy, dx = cmeta["dz_um"], cmeta["dy_um"], cmeta["dx_um"]

    print(f"[reg] structural channel = c{cs} "
          f"({cmeta['channels'][cs]['name']!r}); signal channel c"
          f"{args.signal_channel} is NOT used to drive registration")
    print(f"[reg] mode={args.reg_mode}  backend={'GPU' if xp is not np else 'CPU'}")

    bg = compute_background(arr, cmeta, args)
    roi = find_roi(arr, bg, cs, args, xp, ndi)
    print(f"[reg] sample ROI from c{cs}: y {roi[0]}:{roi[1]} x {roi[2]}:{roi[3]}"
          f"  ({(roi[1]-roi[0])*dy:.0f} x {(roi[3]-roi[2])*dx:.0f} um)")

    feat_cache = {}

    def feat(t, clipped):
        key = (t, clipped)
        if key not in feat_cache:
            if len(feat_cache) > 12:
                feat_cache.clear()
            feat_cache[key] = reg_feature(arr, bg, t, cs, roi, args, xp, ndi,
                                          clipped=clipped)
        return feat_cache[key]

    ref_t = max(0, args.ref_frame)
    up = args.upsample
    bnd = args.reg_bound
    jump = args.reg_jump_px
    dual = args.reg_clip_pct < 100
    shifts = np.zeros((nT, 3), float)          # (dz, dy, dx) in binned pixels
    used_raw = []
    t0 = time.time()

    if args.reg_mode in ("sequential", "hybrid"):
        # Frame-to-frame, accumulated.  Adjacent frames are always comparable:
        # equally bleached, equally noisy, and the sample has moved only a
        # little between them.  This is what survives the flood, where a direct
        # correlation against a distant reference loses lock.
        #
        # Two features per frame (see reg_feature).  The winsorised one tracks
        # slow drift accurately; the raw one is the only thing that can bridge a
        # large single-frame displacement.  Pick per frame pair, on measured
        # displacement, so the choice transfers to other recordings.
        for direction in (+1, -1):
            prev_t = ref_t
            acc = np.zeros(3)
            rng = (range(ref_t + 1, nT) if direction > 0
                   else range(ref_t - 1, -1, -1))
            for t in rng:
                d_raw = pcc_bounded(feat(prev_t, False), feat(t, False),
                                    max_shift=(bnd, bnd), predicted=(0, 0),
                                    xp=xp, upsample=up)
                if dual and np.linalg.norm(d_raw) <= jump:
                    d = pcc_bounded(feat(prev_t, True), feat(t, True),
                                    max_shift=(jump, jump), predicted=(0, 0),
                                    xp=xp, upsample=up)
                else:
                    d = d_raw
                    if dual:
                        used_raw.append(t)
                acc = acc + np.r_[0.0, d]
                shifts[t] = acc
                prev_t = t
                if direction > 0 and t % 20 == 0:
                    print(f"[reg]  seq t={t}/{nT} cum=("
                          f"{acc[1]:+.2f}y {acc[2]:+.2f}x) px "
                          f"{time.time() - t0:.0f}s", flush=True)
        if dual:
            print(f"[reg] frames needing the un-winsorised feature (large "
                  f"displacement): {sorted(used_raw)}")

    if args.reg_mode == "direct":
        pred = np.zeros(2)
        for t in range(nT):
            d = pcc_bounded(feat(ref_t, True), feat(t, True), max_shift=(bnd, bnd),
                            predicted=pred, xp=xp, upsample=up)
            shifts[t] = np.r_[0.0, d]
            pred = d

    if args.reg_mode == "hybrid":
        # refine against the reference, but only within a tight window around
        # the sequential answer, so the refinement can correct accumulated
        # error without being free to hop to a different correlation peak
        for t in range(nT):
            pre = shifts[t]
            cur = ndi.shift(feat(t, True), (float(pre[1]), float(pre[2])), order=1,
                            mode="constant", cval=0.0)
            d = pcc_bounded(feat(ref_t, True), cur,
                            max_shift=(args.reg_refine_bound,) * 2,
                            predicted=(0, 0), xp=xp, upsample=up)
            shifts[t] = pre + np.r_[0.0, d]
            if t % 20 == 0:
                print(f"[reg]  refine t={t}/{nT} ({shifts[t][1]:+.2f}y "
                      f"{shifts[t][2]:+.2f}x) px {time.time() - t0:.0f}s", flush=True)

    # ---- axial, estimated separately and reported whether or not applied ----
    zprof = np.zeros(nT)
    if args.reg_z or args.reg_z_report:
        pz_ref = to_host(z_profile(arr, bg, ref_t, cs, roi, xp, ndi))
        for t in range(nT):
            pz = to_host(z_profile(arr, bg, t, cs, roi, xp, ndi))
            d = pcc_bounded(pz_ref, pz, max_shift=(args.reg_z_max,),
                            predicted=(0,), xp=np, upsample=up)
            zprof[t] = float(d[0])
        print(f"[reg] axial estimate: range {zprof.min():+.2f} to "
              f"{zprof.max():+.2f} planes "
              f"({zprof.min()*dz:+.1f} to {zprof.max()*dz:+.1f} um)")
    if args.reg_z:
        shifts[:, 0] = np.clip(zprof, -args.reg_z_max, args.reg_z_max)
    else:
        shifts[:, 0] = 0.0
        print(f"[reg] axial shift NOT applied (z step is {dz:.1f} um -- one "
              f"plane; sub-plane axial motion is not resolvable). "
              f"Use --reg-z to apply it anyway.")

    um = shifts * np.array([dz, dy, dx])
    t_s = (np.asarray(cmeta["times_s"]) if cmeta.get("times_s")
           else np.arange(nT, dtype=float))

    # ---- transit / smear detection -------------------------------------
    # Two independent signals:
    #   step_um  -- how far the sample moved BETWEEN this frame and the last
    #   smear_um -- how far it moved WITHIN this volume.  Each volume takes
    #               several seconds to acquire plane by plane, so if the sample
    #               is moving during that window the top and bottom halves of
    #               the stack disagree.  No rigid transform can fix that, which
    #               is why these frames get flagged rather than corrected.
    step_um = np.r_[0.0, np.linalg.norm(np.diff(um[:, 1:], axis=0), axis=1)]

    smear_um = np.zeros(nT)
    if args.smear_check:
        y0, y1, x0, x1 = roi
        half = nZ // 2

        def half_mip(t, lo, hi):
            v = xp.asarray(np.asarray(arr[t, cs]) - bg[cs], dtype=xp.float32)
            p = v[lo:hi, y0:y1, x0:x1].max(axis=0)
            p = xp.clip(p - ndi.gaussian_filter(p, args.reg_hp_sigma), 0, None)
            w = xp.outer(xp.hanning(p.shape[0]),
                         xp.hanning(p.shape[1])).astype(xp.float32)
            return p * w

        raw = np.zeros((nT, 2))
        for t in range(nT):
            raw[t] = pcc_bounded(half_mip(t, 0, half), half_mip(t, half, nZ),
                                 max_shift=(args.reg_bound,) * 2,
                                 predicted=(0, 0), xp=xp, upsample=up)
        # the two halves image different depths, so a constant offset between
        # them is normal; only the DEPARTURE from that baseline means motion
        raw = raw - np.median(raw, axis=0)
        smear_um = np.linalg.norm(raw * np.array([dy, dx]), axis=1)

    big = step_um > args.transit_um
    if args.smear_check:
        big |= smear_um > args.smear_um
    # A volume takes several seconds to acquire out of a much longer frame
    # interval, and we cannot tell where inside that interval the sample moved.
    # So when a large step is detected, the frames on either side of it are
    # suspect too: widen the flag by --transit-widen frames.
    transit = big.copy()
    for k in range(1, int(args.transit_widen) + 1):
        transit |= np.r_[big[k:], np.zeros(k, bool)]      # earlier neighbours
        transit |= np.r_[np.zeros(k, bool), big[:-k]]     # later neighbours
    idx = np.where(transit)[0]
    print(f"[reg] TRANSIT FRAMES (step > {args.transit_um} um, widened by "
          f"+/-{int(args.transit_widen)} frames): {idx.tolist()}")
    for i in idx:
        print(f"        t={i:3d}  {t_s[i] / 60:6.2f} min   "
              f"step={step_um[i]:7.1f} um   intra-volume smear="
              f"{smear_um[i]:6.1f} um")
    if args.smear_check:
        print(f"[reg] intra-volume smear: median {np.median(smear_um):.1f} um, "
              f"p95 {np.percentile(smear_um, 95):.1f} um, "
              f"max {smear_um.max():.1f} um (at t={int(smear_um.argmax())})")

    hdr = ("t,time_s,shift_z_px,shift_y_px,shift_x_px,"
           "shift_z_um,shift_y_um,shift_x_um,step_um,transit,smear_um,"
           "z_estimate_planes,z_estimate_um")
    np.savetxt(spath,
               np.c_[np.arange(nT), t_s, shifts, um, step_um,
                     transit.astype(int), smear_um, zprof, zprof * dz],
               delimiter=",", header=hdr, comments="",
               fmt=["%d", "%.3f"] + ["%.4f"] * 7 + ["%d"] + ["%.4f"] * 3)
    print(f"[reg] wrote {spath}  ({time.time() - t0:.0f}s)")

    plot_shifts(args, cmeta, t_s, um, step_um, transit, zprof * dz, smear_um)
    return np.genfromtxt(spath, delimiter=",", names=True)


def plot_shifts(args, cmeta, t_s, um, step_um, transit, z_est_um=None,
                smear_um=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tm = t_s / 60.0
    fig, ax = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)

    ax[0].plot(tm, um[:, 2], label="x", lw=1.7)
    ax[0].plot(tm, um[:, 1], label="y", lw=1.7)
    ax[0].set_ylabel("lateral shift (um)")
    ax[0].legend(loc="best"); ax[0].grid(alpha=.3)
    ax[0].set_title("Measured sample motion  --  registration computed on "
                    f"c{args.struct_channel} "
                    f"({cmeta['channels'][args.struct_channel]['name']}), "
                    f"mode={args.reg_mode}")

    if z_est_um is not None:
        ax[1].plot(tm, z_est_um, color="tab:gray", lw=1.2, ls=":",
                   label="z estimated (not applied)" if not args.reg_z else
                         "z estimated")
    ax[1].plot(tm, um[:, 0], color="tab:green", lw=1.7, label="z applied")
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].axhline(cmeta["dz_um"], color="k", ls="--", lw=.6)
    ax[1].axhline(-cmeta["dz_um"], color="k", ls="--", lw=.6)
    ax[1].set_ylabel("axial shift (um)")
    ax[1].legend(loc="best", fontsize=8); ax[1].grid(alpha=.3)
    ax[1].text(.012, .06,
               f"dashed = +/- one z plane ({cmeta['dz_um']:.1f} um). Axial "
               f"motion smaller than this is not resolvable by this stack.",
               transform=ax[1].transAxes, fontsize=8, color="dimgray")

    ax[2].plot(tm, step_um, color="tab:red", lw=1.5,
               label="step BETWEEN frames")
    if smear_um is not None and np.any(smear_um):
        ax[2].plot(tm, smear_um, color="tab:purple", lw=1.4,
                   label="motion WITHIN one volume (smear)")
    ax[2].axhline(args.transit_um, color="k", ls="--", lw=.8,
                  label=f"transit threshold {args.transit_um:g} um")
    ax[2].set_yscale("symlog", linthresh=10)
    ax[2].set_ylabel("lateral motion (um)")
    ax[2].set_xlabel("time (min)")
    ax[2].legend(loc="best", fontsize=8); ax[2].grid(alpha=.3)

    for i in np.where(transit)[0]:
        for a in ax:
            a.axvspan(tm[max(i - 1, 0)], tm[i], color="red", alpha=.13, lw=0)

    p = Path(args.outdir) / "shifts.png"
    plt.tight_layout(); plt.savefig(p, dpi=130); plt.close()
    print(f"[reg] wrote {p}")


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

_CUSTOM_LUTS = {
    # single-hue fluorescence ramps: black -> hue -> pale, like the LUTs used
    # for a green-emitting fluorophore.  The pale tip marks the top of the scale
    # so saturation is visible instead of hiding in solid colour.
    "green":   [(0.0, (0, 0, 0)), (0.72, (0.05, 0.90, 0.25)), (1.0, (0.88, 1.0, 0.85))],
    "cyan":    [(0.0, (0, 0, 0)), (0.72, (0.05, 0.85, 0.95)), (1.0, (0.85, 1.0, 1.0))],
    "magenta": [(0.0, (0, 0, 0)), (0.72, (0.95, 0.10, 0.90)), (1.0, (1.0, 0.85, 1.0))],
}


def make_lut(name, n=256):
    if name in _CUSTOM_LUTS:
        stops = _CUSTOM_LUTS[name]
        xs = np.array([s[0] for s in stops])
        cols = np.array([s[1] for s in stops], dtype=np.float32)
        t = np.linspace(0, 1, n)
        return np.stack([np.interp(t, xs, cols[:, k]) for k in range(3)],
                        axis=1).astype(np.float32)
    import matplotlib
    cm = matplotlib.colormaps[name]
    return np.asarray([cm(i / (n - 1))[:3] for i in range(n)], dtype=np.float32)


class Raycaster:
    """Orthographic emission/absorption volume raycaster in PHYSICAL space.

    The volume is strongly anisotropic and is deliberately NOT resampled to
    isotropic.  Rays are marched through real micron coordinates and converted
    to voxel indices at sample time using the true spacing, so the rendered
    geometry is correct without touching the data.
    """

    def __init__(self, shape_zyx, spacing_zyx_um, width, height, xp, ndi):
        self.xp, self.ndi = xp, ndi
        self.nz, self.ny, self.nx = shape_zyx
        self.dz, self.dy, self.dx = spacing_zyx_um
        self.W, self.H = width, height
        self.Lz = (self.nz - 1) * self.dz
        self.Ly = (self.ny - 1) * self.dy
        self.Lx = (self.nx - 1) * self.dx
        self.radius = 0.5 * math.sqrt(self.Lx ** 2 + self.Ly ** 2 + self.Lz ** 2)

    @staticmethod
    def _rot(az_deg, el_deg, spin_deg=0.0):
        """Camera <- world rotation.

        `spin` turns the sample about its own axial (z) axis -- a turntable.
        For a volume that is much thinner in z than it is wide, this is the
        rotation worth using: the view axis stays close to the z axis, so the
        render stays in the well-sampled regime the whole way round.  `az`
        instead swings the sample about y, which tips a thin slab edge-on twice
        per revolution, where the 20 um plane spacing shows badly.
        """
        a, e = math.radians(az_deg), math.radians(el_deg)
        s = math.radians(spin_deg)
        ca, sa, ce, se = math.cos(a), math.sin(a), math.cos(e), math.sin(e)
        cs, ss = math.cos(s), math.sin(s)
        Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
        Rz = np.array([[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]])
        return Rx @ Ry @ Rz

    def um_per_screen_px(self, zoom):
        return (2 * self.radius / zoom) / self.W

    def render(self, vols, luts, vmins, vmaxs, opac, az, el,
               n_samples=320, mode="composite", zoom=1.0, gamma=1.0,
               emit=None, cutaway=False, cut_center=(0.0, 0.0), spin=0.0,
               cut_span=45.0, cut_yaw=0.0, roll=0.0, pan=(0.0, 0.0)):
        """Emission-absorption volume integral.

        Emission and absorption are separate: `emit` sets how much light each
        unit of signal contributes, `opac` sets how much it hides what is
        behind it.  Tying them together (a single "opacity") makes a bright
        surface saturate at the first sample and the interior never shows,
        which is exactly wrong for watching something fill a volume.
        """
        xp, ndi = self.xp, self.ndi
        if emit is None:
            emit = list(opac)
        R = self._rot(az, el, spin)
        Rm = xp.asarray(R, dtype=xp.float32)

        # Quarter cutaway: drop a 90-degree wedge of the lateral plane, full
        # depth, so the ray reaches the interior instead of stopping at the
        # near surface.
        #
        # The wedge is centred on the camera's BEARING in the xy plane, not on
        # one of the four axis-aligned quadrants.  Picking a quadrant from the
        # signs of the camera direction only faces the viewer when the camera
        # is near a diagonal; on the axes one sign is degenerate and the cut
        # ends up behind the sample, invisible.  Centring the wedge on the
        # bearing keeps it facing the viewer at every azimuth.
        cut_c = cut_s = 0.0
        cut_cos = 0.0
        if cutaway:
            bx, by = float(R[0, 2]), float(R[1, 2])
            phi = math.atan2(by, bx) if math.hypot(bx, by) > 1e-6 else \
                math.radians(45.0)
            # yaw swings the cut away from dead-on.  At yaw 0 you look straight
            # into the cut and see it edge-on; swung round, the exposed face is
            # presented at an angle, the way a sectioned specimen is normally
            # photographed.
            phi += math.radians(cut_yaw)
            cut_c, cut_s = math.cos(phi), math.sin(phi)
            # half-angle of the removed wedge: 45 deg removes a quarter,
            # 90 deg removes half the volume and leaves one flat face.
            cut_cos = math.cos(math.radians(cut_span))

        # Square pixels.  u and v must span physical distances in the same
        # ratio as the frame's pixel dimensions -- spanning the same distance
        # over different pixel counts stretches the image along the longer axis.
        half_u = self.radius / zoom
        half_v = half_u * (self.H / self.W)
        u = xp.linspace(-half_u, half_u, self.W, dtype=xp.float32)
        v = xp.linspace(half_v, -half_v, self.H, dtype=xp.float32)
        U, V = xp.meshgrid(u, v)
        U, V = U.ravel(), V.ravel()

        # roll turns the image in its own plane, about the view axis
        if roll:
            r = math.radians(roll)
            cr, sr = math.cos(r), math.sin(r)
            U, V = U * cr - V * sr, U * sr + V * cr
        # pan slides the view window; sign chosen so +pan moves the OBJECT the
        # way the control says, not the camera
        if pan[0] or pan[1]:
            U = U - float(pan[0])
            V = V - float(pan[1])

        ws = np.linspace(self.radius, -self.radius, n_samples)   # front to back
        ds = abs(float(ws[1] - ws[0]))
        ds_norm = ds / (2 * self.radius / 256.0)

        colour = xp.zeros((U.size, 3), dtype=xp.float32)
        trans = xp.ones(U.size, dtype=xp.float32)      # remaining transmittance
        peak = [xp.zeros(U.size, dtype=xp.float32) for _ in vols]

        for w in ws:
            X = Rm[0, 0] * U + Rm[0, 1] * V + Rm[0, 2] * w
            Y = Rm[1, 0] * U + Rm[1, 1] * V + Rm[1, 2] * w
            Z = Rm[2, 0] * U + Rm[2, 1] * V + Rm[2, 2] * w
            cz = (Z + self.Lz / 2) / self.dz
            cy = (Y + self.Ly / 2) / self.dy
            cx = (X + self.Lx / 2) / self.dx
            inside = ((cz >= 0) & (cz <= self.nz - 1) &
                      (cy >= 0) & (cy <= self.ny - 1) &
                      (cx >= 0) & (cx <= self.nx - 1))
            if cutaway:
                # rotate into the cut's own frame: `along` is the component
                # towards the cut direction, `across` the perpendicular.  A
                # sample is removed when its angle from that direction is less
                # than the half-angle, i.e. cos(angle) > cos(half-angle).
                # Written this way, span 90 degrees degenerates cleanly to the
                # half-space cut (along > 0) with no special case.
                px = X - cut_center[0]
                py = Y - cut_center[1]
                along = px * cut_c + py * cut_s
                across = -px * cut_s + py * cut_c
                inside &= ~(along > cut_cos * xp.sqrt(along * along +
                                                      across * across))
            inside = inside.astype(xp.float32)
            # Rays are cast through the volume's bounding SPHERE, so most
            # samples fall outside the box -- cz spans roughly -30..50 for a
            # 20-plane stack.  Clamp before sampling: CuPy's map_coordinates
            # still forms an address for far out-of-range coordinates even with
            # mode="constant", which faults with an illegal memory access.
            # `inside` zeroes these samples immediately afterwards, so clamping
            # changes no rendered value.
            coords = xp.stack([
                xp.clip(cz, 0, self.nz - 1),
                xp.clip(cy, 0, self.ny - 1),
                xp.clip(cx, 0, self.nx - 1),
            ])

            for k, (vol, lut, vmn, vmx, op, em) in enumerate(
                    zip(vols, luts, vmins, vmaxs, opac, emit)):
                # mode="nearest", not "constant": the coordinates above are
                # already clamped into the volume and `inside` masks off every
                # sample that came from outside it, so the two modes give
                # identical results here -- but CuPy's constant-mode kernel
                # faults with an illegal memory access on this data.
                s = ndi.map_coordinates(vol, coords, order=1, mode="nearest")
                s = (s - vmn) / max(vmx - vmn, 1e-6)
                s = xp.clip(s, 0.0, 1.0) * inside
                if gamma != 1.0:
                    s = s ** gamma
                if mode == "mip":
                    peak[k] = xp.maximum(peak[k], s)
                    continue
                idx = xp.clip((s * (lut.shape[0] - 1)).astype(xp.int32),
                              0, lut.shape[0] - 1)
                colour += (trans * em * s * ds_norm)[:, None] * lut[idx]
                trans = trans * xp.exp(-op * s * ds_norm)

        if mode == "mip":
            colour = xp.zeros((U.size, 3), dtype=xp.float32)
            for k, lut in enumerate(luts):
                s = peak[k]
                idx = xp.clip((s * (lut.shape[0] - 1)).astype(xp.int32),
                              0, lut.shape[0] - 1)
                colour = xp.maximum(colour, lut[idx])

        return xp.clip(colour, 0, 1).reshape(self.H, self.W, 3)


# ---------------- overlays ----------------

_FONT_CACHE = {}


def _font(size):
    from PIL import ImageFont
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for d in (r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu",
              "/Library/Fonts", "/usr/share/fonts"):
        for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf",
                     "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
            p = Path(d) / name
            if p.exists():
                try:
                    f = ImageFont.truetype(str(p), size)
                    _FONT_CACHE[size] = f
                    return f
                except Exception:
                    pass
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def annotate(img_u8, *, elapsed_s, um_per_screen_px, label=None, transit=False,
             cbar=None, scalebar_um=200.0, mode="full", frame=None,
             n_frames=None):
    """Burn elapsed time, a scale bar, a colour key and any flags into a frame.

    mode: "full" (everything), "minimal" (time + frame count only), "none".
    """
    from PIL import Image, ImageDraw

    if mode == "none":
        return img_u8

    im = Image.fromarray(img_u8)
    d = ImageDraw.Draw(im)
    W, H = im.size
    f_big = _font(max(20, W // 22))
    f_sm = _font(max(11, W // 60))

    mm, ss = divmod(int(round(elapsed_s)), 60)
    x0, y0 = W * 0.035, H * 0.03

    if mode == "minimal":
        d.text((x0, y0), f"{mm:02d}:{ss:02d}", font=f_big, fill=(255, 255, 255))
        if frame is not None:
            fr = (f"frame {frame + 1}" +
                  (f" / {n_frames}" if n_frames else ""))
            d.text((x0, y0 + f_big.size * 1.12), fr, font=f_sm,
                   fill=(200, 200, 200))
        if cbar is not None:
            _draw_colorbar(im, d, cbar, W, H, f_sm)
        return np.asarray(im)

    d.text((x0, y0), f"{mm:02d}:{ss:02d}", font=f_big, fill=(255, 255, 255))
    d.text((x0, y0 + f_big.size * 1.12), "mm:ss after start of recording",
           font=f_sm, fill=(165, 165, 165))

    if transit:
        d.text((x0, y0 + f_big.size * 1.12 + f_sm.size * 1.5),
               "MOTION TRANSIT - volume internally smeared",
               font=f_sm, fill=(255, 95, 95))

    if label:
        d.text((x0, H - H * 0.05), label, font=f_sm, fill=(190, 190, 190))

    px = scalebar_um / um_per_screen_px
    if px < W * 0.55:
        xr = W - W * 0.05
        xl = xr - px
        y = H - H * 0.055
        d.rectangle([xl, y, xr, y + max(3, H // 220)], fill=(255, 255, 255))
        d.text((xl, y - f_sm.size * 1.45), f"{scalebar_um:g} um", font=f_sm,
               fill=(255, 255, 255))

    if cbar is not None:
        _draw_colorbar(im, d, cbar, W, H, f_sm)

    return np.asarray(im)


def _nice_ticks(lo, hi, n=5):
    """Round tick values spanning [lo, hi], like a plotting library would pick."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw),
               default=10) * mag
    first = math.ceil(lo / step) * step
    ticks = []
    v = first
    while v <= hi + 1e-9:
        ticks.append(v)
        v += step
    return ticks


def _draw_colorbar(im, d, cbar, W, H, f_sm):
    """A ticked vertical colour key, hard against the right margin with every
    label to its LEFT so it never sits over the sample."""
    from PIL import Image

    lut, vmn, vmx, name = cbar[:4]
    units = cbar[4] if len(cbar) > 4 else "counts above bath"
    bw, bh = max(10, int(W * 0.018)), int(H * 0.30)
    bx, by = int(W - W * 0.040 - bw), int(H * 0.16)
    ramp = np.linspace(lut.shape[0] - 1, 0, bh).astype(int)
    strip = np.repeat((lut[ramp] * 255).astype(np.uint8)[:, None, :], bw, axis=1)
    im.paste(Image.fromarray(strip), (bx, by))
    d.rectangle([bx, by, bx + bw, by + bh], outline=(255, 255, 255))
    rx = bx + bw
    d.text((rx, by - f_sm.size * 2.7), name, font=f_sm, fill=(235, 235, 235),
           anchor="ra")
    d.text((rx, by - f_sm.size * 1.4), units, font=f_sm, fill=(170, 170, 170),
           anchor="ra")
    # round ticks inside the range, plus the two end values so the reader
    # knows where the scale actually starts and stops; drop any round tick
    # that would collide with an end label
    span = max(vmx - vmn, 1e-9)
    ticks = [v for v in _nice_ticks(vmn, vmx)
             if abs(v - vmn) > 0.07 * span and abs(v - vmx) > 0.07 * span]
    for v, end in [(vmn, True), (vmx, True)] + [(v, False) for v in ticks]:
        y = by + bh - (v - vmn) / span * bh
        d.line([bx - 5, y, bx, y], fill=(255, 255, 255), width=1)
        d.text((bx - 8, y), f"{v:g}", font=f_sm,
               fill=(225, 225, 225) if not end else (255, 255, 255),
               anchor="rm")


# ---------------- frame production ----------------

def sample_bbox(arr, bg, channel, args, cmeta, pad_um=60.0, n_probe=8,
                frac=0.25):
    """Bounding box of the sample in the binned grid.

    Rendering the whole field wastes most of the frame on bath, and -- worse
    for a cutaway -- the removed wedge is a quarter of the FIELD, which is far
    larger than the sample, so it swallows the picture.  Cropping to the sample
    keeps the wedge proportionate to the object.
    """
    from scipy import ndimage as sndi

    nT, nC, nZ, Yb, Xb = arr.shape
    idx = np.linspace(0, nT - 1, min(n_probe, nT)).astype(int)
    acc = None
    for t in idx:
        m = (np.asarray(arr[t, channel]) - bg[channel]).max(axis=0)
        acc = m if acc is None else np.maximum(acc, m)
    sm = sndi.gaussian_filter(acc, 8)
    # Threshold relative to the sample's own brightness, not to a percentile of
    # the frame: the fraction of the field the sample covers varies between
    # recordings, so a fixed percentile crops into a large sample and includes
    # bath around a small one.  The background has already been removed, so the
    # medium sits near zero and a fraction of the bright end separates them.
    thr = frac * float(np.percentile(sm, 99.5))
    mask = sm > thr
    mask = sndi.binary_closing(mask, iterations=6)
    lbl, n = sndi.label(mask)
    if n:
        sizes = sndi.sum(mask, lbl, range(1, n + 1))
        mask = lbl == (int(np.argmax(sizes)) + 1)
    mask = sndi.binary_fill_holes(mask)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, Yb, 0, Xb
    py = int(round(pad_um / cmeta["dy_um"]))
    px = int(round(pad_um / cmeta["dx_um"]))
    return (max(0, int(ys.min()) - py), min(Yb, int(ys.max()) + py),
            max(0, int(xs.min()) - px), min(Xb, int(xs.max()) + px))


def load_roi_mask(path, crop, shape_yx, cmeta, xp):
    """Rasterise a saved ROI (roi_tool.py) into a float mask for the current
    crop.  Polygons are stored in FULL-cache binned-pixel coordinates so the
    same file applies whatever crop is in use.  `feather_um` softens the edge
    so the cut-out doesn't read as a hard paper edge."""
    import json
    from PIL import Image, ImageDraw
    from scipy import ndimage as sndi

    spec = json.loads(Path(path).read_text())
    if spec.get("bin") != cmeta.get("bin_xy"):
        print(f"[roi] WARNING: ROI drawn at bin {spec.get('bin')}, cache is bin "
              f"{cmeta.get('bin_xy')} -- coordinates will not line up")
    y0, x0 = (crop[0], crop[2]) if crop is not None else (0, 0)
    H, W = shape_yx
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for poly in spec["keep"]:
        pts = [(float(x) - x0, float(y) - y0) for x, y in poly]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    for poly in spec.get("remove", []):
        pts = [(float(x) - x0, float(y) - y0) for x, y in poly]
        if len(pts) >= 3:
            d.polygon(pts, fill=0)
    m = np.asarray(im, dtype=np.float32) / 255.0
    f_um = float(spec.get("feather_um", 0.0))
    if f_um > 0:
        m = sndi.gaussian_filter(m, f_um / cmeta["dx_um"])
    print(f"[roi] {len(spec['keep'])} keep polygon(s), "
          f"{len(spec.get('remove', []))} remove, feather {f_um:g} um, "
          f"keeps {100 * m.mean():.0f}% of the crop")
    return xp.asarray(m)


def build_volume(arr, bg, shifts, t, channels, xp, ndi, apply_shift=True,
                 haze=0.0, haze_sigma_px=0.0, crop=None, zrange=None,
                 roi_mask=None):
    """Background-subtract in the CAMERA frame, then shift into the sample frame.

    Order matters: the vignette and dish artefacts are fixed relative to the
    camera, the sample is not.

    `haze` optionally removes spinning-disk out-of-focus light: each plane has a
    fraction of its own large-scale blur subtracted.  This is the same fixed
    linear filter on every frame, so it cannot distort the time course -- if the
    signal doubles, the filtered signal doubles.  That is what separates it from
    per-frame normalisation.  It does suppress genuinely broad structure along
    with the glare, so it is off unless asked for.
    """
    out = []
    for c in channels:
        v = xp.asarray(np.asarray(arr[t, c]) - bg[c], dtype=xp.float32)
        if haze > 0 and haze_sigma_px > 0:
            v = v - haze * ndi.gaussian_filter(v, (0, haze_sigma_px,
                                                   haze_sigma_px))
        if apply_shift:
            sh = tuple(float(x) for x in shifts[t])
            if any(abs(x) > 1e-6 for x in sh):
                v = ndi.shift(v, sh, order=1, mode="constant", cval=0.0)
        if crop is not None:
            # crop AFTER shifting, so the crop is fixed in the sample frame
            y0, y1, x0, x1 = crop
            v = v[:, y0:y1, x0:x1]
        if zrange is not None:
            z0, z1 = zrange
            v = v[z0:z1]
        if roi_mask is not None:
            # applied after crop, in the sample frame, through every z-plane
            v = v * roi_mask[None, :, :]
        out.append(v)
    return out


def pick_display_range(arr, bg, shifts, args, xp, ndi, hz=(0.0, 0.0), crop=None,
                       zrange=None, roi_mask=None):
    """One fixed intensity window for the entire movie, taken from a late frame.

    Deliberately NOT recomputed per frame: per-frame auto-contrast would
    normalise away the very thing the animation exists to show.
    """
    nT = arr.shape[0]
    t = args.range_frame if args.range_frame >= 0 else nT - 1
    v = to_host(build_volume(arr, bg, shifts, t, [args.signal_channel], xp, ndi,
                             haze=hz[0], haze_sigma_px=hz[1], crop=crop, zrange=zrange,
                             roi_mask=roi_mask)[0])
    hi = float(np.percentile(v, args.vmax_pct))

    # The bottom of the scale is the level the signal channel reaches BEFORE
    # the drug arrives -- i.e. the noise ceiling of the baseline.  Anything at
    # or below it is indistinguishable from no drug, so that is where the
    # display should start.  Taken from the baseline frames, which is a
    # property of the recording, not a knob tuned to make the picture look nice.
    nb = min(args.baseline_n, nT)
    base = np.concatenate([
        to_host(build_volume(arr, bg, shifts, b, [args.signal_channel],
                             xp, ndi, haze=hz[0], haze_sigma_px=hz[1],
                             crop=crop, zrange=zrange, roi_mask=roi_mask)[0]).ravel()
        for b in range(nb)])
    lo = float(np.percentile(base, args.vmin_baseline_pct))

    if args.vmin is not None:
        lo = args.vmin
    if args.vmax is not None:
        hi = args.vmax
    print(f"[range] fixed display window [{lo:.2f}, {hi:.2f}] -- held for EVERY "
          f"frame (no per-frame autoscaling)")
    print(f"[range]   vmin = p{args.vmin_baseline_pct} of the first {nb} "
          f"(pre-drug) timepoints = the baseline noise ceiling")
    print(f"[range]   vmax = p{args.vmax_pct} of t={t}")
    return lo, hi


def render_frames(args, cmeta, arr, tab, ts, out_mp4=None, preview_ts=None):
    xp, ndi, _ = backend(args.gpu)
    nT, nC, nZ, Yb, Xb = arr.shape

    bg = compute_background(arr, cmeta, args)
    hz = (args.haze, args.haze_um / cmeta["dx_um"] if args.haze > 0 else 0.0)
    if args.haze > 0:
        print(f"[render] haze removal: subtracting {args.haze:g} x a "
              f"{args.haze_um:g} um blur per plane (same filter every frame)")
    shifts = np.c_[tab["shift_z_px"], tab["shift_y_px"], tab["shift_x_px"]]
    if args.no_register:
        print("[render] --no-register: shifts NOT applied")
        shifts = np.zeros_like(shifts)
    transit = tab["transit"].astype(bool)

    zrange = None
    if args.z_min is not None or args.z_max is not None:
        zrange = (args.z_min or 0, args.z_max if args.z_max is not None else nZ)
        nZ = zrange[1] - zrange[0]
        print(f"[render] z slab: planes {zrange[0]}:{zrange[1]} "
              f"({nZ} of {arr.shape[2]})")

    crop = None
    if args.crop:
        crop = sample_bbox(arr, bg, args.signal_channel, args, cmeta,
                           pad_um=args.crop_pad_um, frac=args.crop_frac)
        print(f"[render] cropped to the sample: y {crop[0]}:{crop[1]} "
              f"x {crop[2]}:{crop[3]}  "
              f"({(crop[1]-crop[0])*cmeta['dy_um']:.0f} x "
              f"{(crop[3]-crop[2])*cmeta['dx_um']:.0f} um of "
              f"{Yb*cmeta['dy_um']:.0f} x {Xb*cmeta['dx_um']:.0f})")
        Yb, Xb = crop[1] - crop[0], crop[3] - crop[2]

    roi_mask = None
    if args.roi:
        roi_mask = load_roi_mask(args.roi, crop, (Yb, Xb), cmeta, xp)

    vmn, vmx = pick_display_range(arr, bg, shifts, args, xp, ndi, hz, crop,
                                  zrange, roi_mask)

    rc = Raycaster((nZ, Yb, Xb),
                   (cmeta["dz_um"], cmeta["dy_um"], cmeta["dx_um"]),
                   args.width, args.height, xp, ndi)

    chans = [args.signal_channel]
    luts = [xp.asarray(make_lut(args.cmap))]
    vmins, vmaxs, opac, emit = [vmn], [vmx], [args.opacity], [args.emit]
    if args.show_struct:
        chans.append(args.struct_channel)
        luts.append(xp.asarray(make_lut(args.struct_cmap)))
        sv = to_host(build_volume(arr, bg, shifts, 0, [args.struct_channel],
                                  xp, ndi, crop=crop, zrange=zrange,
                                  roi_mask=roi_mask)[0])
        # same window as the signal channel unless told otherwise, so that when
        # both are drawn in one colour their brightness means the same thing
        vmins.append(vmn if args.struct_vmin is None else args.struct_vmin)
        vmaxs.append(vmx if args.struct_vmax is None else args.struct_vmax)
        opac.append(args.struct_opacity)
        emit.append(args.struct_emit)

    # Anchor the cutaway on the sample, not the field.  The brain sits off to
    # one side of the 1124 um field, so a cut through the field centre would
    # shave its edge instead of opening its middle.
    cut_center = (0.0, 0.0)
    if args.cutaway:
        rf = args.range_frame if args.range_frame >= 0 else nT - 1
        m = to_host(build_volume(arr, bg, shifts, rf, [args.signal_channel],
                                 xp, ndi, haze=hz[0], haze_sigma_px=hz[1],
                                 crop=crop, zrange=zrange,
                                 roi_mask=roi_mask)[0]).max(axis=0)
        w = np.clip(m - vmn, 0, None)
        if w.sum() > 0:
            yc = float((w.sum(1) * np.arange(Yb)).sum() / w.sum())
            xc = float((w.sum(0) * np.arange(Xb)).sum() / w.sum())
            cut_center = (xc * cmeta["dx_um"] - rc.Lx / 2,
                          yc * cmeta["dy_um"] - rc.Ly / 2)
        cut_center = (cut_center[0] + args.cut_dx, cut_center[1] + args.cut_dy)
        print(f"[render] cutaway centred on the sample at "
              f"({cut_center[0]:+.0f}, {cut_center[1]:+.0f}) um from field centre")

    upp = rc.um_per_screen_px(args.zoom)
    cbar_lut = make_lut(args.cmap) if args.colorbar else None
    ts_list = list(preview_ts) if preview_ts is not None else list(range(nT))
    good = np.where(~transit)[0]

    writer = None
    if out_mp4:
        import imageio.v2 as iio
        writer = iio.get_writer(out_mp4, fps=args.fps, codec="libx264",
                                macro_block_size=None,
                                ffmpeg_params=["-crf", str(args.crf),
                                               "-pix_fmt", "yuv420p"])

    t0 = time.time()
    outs = []
    for i, t in enumerate(ts_list):
        smeared = bool(transit[t])
        if smeared and args.transit == "interp" and len(good):
            lo_i = good[good < t]
            hi_i = good[good > t]
            if len(lo_i) and len(hi_i):
                a, b = int(lo_i[-1]), int(hi_i[0])
                w = (t - a) / (b - a)
                va = build_volume(arr, bg, shifts, a, chans, xp, ndi,
                                  haze=hz[0], haze_sigma_px=hz[1], crop=crop, zrange=zrange,
                             roi_mask=roi_mask)
                vb = build_volume(arr, bg, shifts, b, chans, xp, ndi,
                                  haze=hz[0], haze_sigma_px=hz[1], crop=crop, zrange=zrange,
                             roi_mask=roi_mask)
                vols = [(1 - w) * p + w * q for p, q in zip(va, vb)]
            else:
                vols = build_volume(arr, bg, shifts, t, chans, xp, ndi,
                                haze=hz[0], haze_sigma_px=hz[1], crop=crop, zrange=zrange,
                             roi_mask=roi_mask)
        else:
            vols = build_volume(arr, bg, shifts, t, chans, xp, ndi,
                                haze=hz[0], haze_sigma_px=hz[1], crop=crop, zrange=zrange,
                             roi_mask=roi_mask)

        frac = i / max(len(ts_list) - 1, 1)
        az = args.azimuth + args.rotate * frac
        spin = args.spin_start + args.spin * frac
        img = rc.render(vols, luts, vmins, vmaxs, opac, az, args.elevation,
                        n_samples=args.samples, mode=args.mode, zoom=args.zoom,
                        gamma=args.gamma, emit=emit, cutaway=args.cutaway,
                        cut_center=cut_center, spin=spin,
                        cut_span=args.cut_span, cut_yaw=args.cut_yaw,
                        roll=args.roll, pan=(args.pan_x, args.pan_y))
        img8 = (to_host(img) * 255).astype(np.uint8)

        sig = cmeta["channels"][args.signal_channel]
        reg_txt = ("NOT registered (sample stationary)" if args.no_register
                   else f"registered on c{args.struct_channel} "
                        f"{cmeta['channels'][args.struct_channel]['name']}")
        bg_txt = ("signal above surrounding medium"
                  if getattr(args, "bg_mode", "image") == "plane-const"
                  else "change since t=0")
        lab = (f"c{args.signal_channel} {sig['name']}  |  {reg_txt}  |  "
               f"{bg_txt}  |  fixed intensity scale, no bleach correction")
        img8 = annotate(img8, elapsed_s=float(ts[t]), um_per_screen_px=upp,
                        label=lab,
                        transit=(smeared and args.transit == "mark"),
                        cbar=((cbar_lut, vmn, vmx,
                               args.signal_label or sig["name"],
                               args.signal_units)
                              if args.colorbar else None),
                        scalebar_um=args.scalebar, mode=args.overlay,
                        frame=t, n_frames=nT)

        if writer:
            writer.append_data(img8)
        outs.append((t, img8))

        if i % 10 == 0 or i == len(ts_list) - 1:
            el = time.time() - t0
            print(f"[render] {i + 1}/{len(ts_list)} (t={t}) {el:6.1f}s "
                  f"~{el / (i + 1) * (len(ts_list) - i - 1):5.1f}s left", flush=True)

    if writer:
        writer.close()
        print(f"[render] wrote {out_mp4}")
    return outs


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["probe", "cache", "register", "preview",
                                     "render", "all"])
    p.add_argument("--nd2", required=True)
    p.add_argument("--outdir", default="out")
    p.add_argument("--cache-dir", default=None,
                   help="read/write the binned cache here (defaults to --outdir)")
    p.add_argument("--bin", type=int, default=4,
                   help="XY binning factor for the cache (default 4)")
    p.add_argument("--force", action="store_true", help="rebuild the cache")
    p.add_argument("--gpu", action="store_true", default=True)
    p.add_argument("--no-gpu", dest="gpu", action="store_false")

    g = p.add_argument_group("channels")
    g.add_argument("--signal-channel", type=int, default=0,
                   help="channel to render (the drug)")
    g.add_argument("--struct-channel", type=int, default=1,
                   help="channel used to COMPUTE registration; never rendered "
                        "unless --show-struct")

    g = p.add_argument_group("background")
    g.add_argument("--baseline-n", type=int, default=5)
    g.add_argument("--bg-sigma", type=float, default=12.0)
    g.add_argument("--bg-mode", choices=["image", "plane-const"], default="image",
                   help="image: subtract a smoothed baseline IMAGE (also removes "
                        "signal already present at t=0, so you see the change "
                        "since t=0). plane-const: subtract one constant per "
                        "z-plane taken from the surrounding medium, keeping "
                        "tissue signal that was already there.")
    g.add_argument("--bg-pct", type=float, default=25.0,
                   help="percentile defining the medium level for --bg-mode "
                        "plane-const")
    g.add_argument("--force-bg", action="store_true")
    g.add_argument("--haze", type=float, default=0.0,
                   help="remove spinning-disk out-of-focus glare: subtract this "
                        "fraction of each plane's large-scale blur. Same linear "
                        "filter on every frame, so the time course is preserved. "
                        "0 = off.")
    g.add_argument("--haze-um", type=float, default=60.0,
                   help="blur scale for --haze, in microns")

    g = p.add_argument_group("registration")
    g.add_argument("--reg-mode", choices=["sequential", "direct", "hybrid"],
                   default="sequential",
                   help="sequential (default) chains frame-to-frame, which is "
                        "what survives a large single-frame displacement; "
                        "direct correlates every frame against --ref-frame")
    g.add_argument("--ref-frame", type=int, default=0)
    g.add_argument("--upsample", type=int, default=10)
    g.add_argument("--reg-hp-sigma", type=float, default=10.0,
                   help="high-pass sigma, binned px, before correlation")
    g.add_argument("--reg-clip-pct", type=float, default=99.0,
                   help="winsorise the registration feature at this percentile "
                        "so a few very bright puncta cannot dominate the "
                        "correlation (100 = off)")
    g.add_argument("--reg-jump-px", type=float, default=30.0,
                   help="frame-to-frame displacement above which the "
                        "un-winsorised feature is used instead")
    g.add_argument("--reg-bound", type=float, default=90.0,
                   help="max frame-to-frame displacement searched, binned px")
    g.add_argument("--reg-refine-bound", type=float, default=15.0,
                   help="max correction the hybrid refinement may apply")
    g.add_argument("--roi-pct", type=float, default=99.0,
                   help="percentile defining the sample ROI in the struct channel")
    g.add_argument("--roi-dilate", type=int, default=15)
    g.add_argument("--roi-pad", type=int, default=40)
    g.add_argument("--reg-z", action="store_true", default=False,
                   help="apply the axial shift too. Off by default: when the z "
                        "step is tens of microns the axial estimate is coarser "
                        "than the motion it is trying to remove.")
    g.add_argument("--reg-z-report", action="store_true", default=True,
                   help="estimate and report axial motion even when not applying it")
    g.add_argument("--reg-z-max", type=float, default=2.0,
                   help="clamp |z shift| to this many planes")
    g.add_argument("--transit-um", type=float, default=15.0)
    g.add_argument("--smear-um", type=float, default=15.0,
                   help="intra-volume displacement above which a volume is "
                        "flagged as internally smeared")
    g.add_argument("--transit-widen", type=float, default=2,
                   help="also flag this many frames either side of a large step")
    g.add_argument("--smear-check", action="store_true", default=False,
                   help="intra-volume smear estimate from top-vs-bottom half "
                        "stack. Off by default: on a sparsely-labelled channel "
                        "the two halves image different structures and the "
                        "correlation is not trustworthy.")
    g.add_argument("--no-smear-check", dest="smear_check", action="store_false")
    g.add_argument("--transit", choices=["mark", "interp", "keep"], default="mark")
    g.add_argument("--force-reg", action="store_true")
    g.add_argument("--no-register", action="store_true")

    g = p.add_argument_group("display scale (fixed for the whole movie)")
    g.add_argument("--range-frame", type=int, default=-1)
    g.add_argument("--vmin-baseline-pct", type=float, default=99.5,
                   help="vmin = this percentile of the pre-drug baseline frames")
    g.add_argument("--vmin-pct", type=float, default=50.0)
    g.add_argument("--vmax-pct", type=float, default=99.95)
    g.add_argument("--vmin", type=float, default=None)
    g.add_argument("--vmax", type=float, default=None)

    g = p.add_argument_group("render")
    g.add_argument("--width", type=int, default=900)
    g.add_argument("--height", type=int, default=900)
    g.add_argument("--samples", type=int, default=320)
    g.add_argument("--mode", choices=["composite", "mip"], default="composite")
    g.add_argument("--cmap", default="inferno")
    g.add_argument("--opacity", type=float, default=0.35,
                   help="absorption: how much signal hides what is behind it")
    g.add_argument("--emit", type=float, default=0.9,
                   help="emission: how much light each unit of signal adds")
    g.add_argument("--gamma", type=float, default=1.0)
    g.add_argument("--zoom", type=float, default=1.72)
    g.add_argument("--azimuth", type=float, default=30.0)
    g.add_argument("--elevation", type=float, default=18.0)
    g.add_argument("--roll", type=float, default=0.0,
                   help="turn the image in its own plane, degrees")
    g.add_argument("--pan-x", type=float, default=0.0,
                   help="slide the view horizontally, microns")
    g.add_argument("--pan-y", type=float, default=0.0,
                   help="slide the view vertically, microns")
    g.add_argument("--cut-dx", type=float, default=0.0,
                   help="move the cut centre in x, microns (from the sample centroid)")
    g.add_argument("--cut-dy", type=float, default=0.0,
                   help="move the cut centre in y, microns")
    g.add_argument("--z-min", type=int, default=None,
                   help="first z plane to render (slab)")
    g.add_argument("--z-max", type=int, default=None,
                   help="last z plane to render, exclusive")
    g.add_argument("--rotate", type=float, default=0.0,
                   help="total degrees of AZIMUTH rotation across the movie "
                        "(swings a thin slab edge-on twice per turn)")
    g.add_argument("--spin", type=float, default=0.0,
                   help="total degrees of TURNTABLE rotation about the sample's "
                        "own z axis across the movie. Preferred for volumes "
                        "much thinner in z than in xy: the view axis never "
                        "leaves the well-sampled regime.")
    g.add_argument("--spin-start", type=float, default=0.0)
    g.add_argument("--crop", action="store_true",
                   help="crop the rendered volume to the sample bounding box")
    g.add_argument("--crop-pad-um", type=float, default=60.0)
    g.add_argument("--cut-span", type=float, default=45.0,
                   help="half-angle of the removed wedge in degrees. 45 takes "
                        "a quarter out; 90 removes half the volume and leaves "
                        "one flat cut face, like a sectioned specimen.")
    g.add_argument("--cut-yaw", type=float, default=0.0,
                   help="rotate the cut away from facing the camera, so the "
                        "exposed face is seen at an angle rather than edge-on")
    g.add_argument("--crop-frac", type=float, default=0.25,
                   help="sample threshold for --crop, as a fraction of the "
                        "bright end of the background-subtracted image")
    g.add_argument("--cutaway", action="store_true",
                   help="cut away the lateral quadrant nearest the camera so "
                        "the interior is visible")
    g.add_argument("--show-struct", action="store_true")
    g.add_argument("--struct-cmap", default="bone")
    g.add_argument("--struct-vmin", type=float, default=None,
                   help="display window for the second channel (default: same as signal)")
    g.add_argument("--struct-vmax", type=float, default=None)
    g.add_argument("--struct-opacity", type=float, default=0.15)
    g.add_argument("--struct-emit", type=float, default=0.35)
    g.add_argument("--colorbar", action="store_true", default=True)
    g.add_argument("--no-colorbar", dest="colorbar", action="store_false")
    g.add_argument("--scalebar", type=float, default=200.0)
    g.add_argument("--signal-label", default=None,
                   help="title for the colour key (default: channel name)")
    g.add_argument("--signal-units", default="counts above bath",
                   help="units line under the colour key title")
    g.add_argument("--roi", default=None,
                   help="roi.json from roi_tool.py: keep only the drawn region")
    g.add_argument("--overlay", choices=["full", "minimal", "none"], default="full",
                   help="minimal = time and frame count only; none = clean frames")
    g.add_argument("--fps", type=float, default=12)
    g.add_argument("--crf", type=int, default=17)
    g.add_argument("--out", default=None)
    g.add_argument("--preview-t", default=None,
                   help="comma-separated timepoints for the preview stage")
    return p


def main():
    args = build_parser().parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    meta = read_meta(args.nd2)
    print_meta(meta)
    (Path(args.outdir) / "metadata.json").write_text(json.dumps(meta, indent=1))
    if args.stage == "probe":
        return

    if args.stage in ("cache", "all"):
        stage_cache(args, meta)
    if args.stage == "cache":
        return

    arr, cmeta = load_cache(args)
    ts = (np.asarray(cmeta["times_s"]) if cmeta.get("times_s")
          else np.arange(arr.shape[0], dtype=float))

    if args.stage in ("register", "all"):
        tab = stage_register(args, cmeta, arr)
    else:
        sp = Path(args.outdir) / "shifts.csv"
        if sp.exists():
            tab = np.genfromtxt(sp, delimiter=",", names=True)
        elif args.no_register:
            # no transform wanted, so none is needed on disk: an all-zero
            # table with the same columns the register stage would write
            nT = arr.shape[0]
            z = np.zeros(nT)
            tab = np.rec.fromarrays(
                [np.arange(nT), ts, z, z, z, z, z, z, z, z.astype(int), z, z, z],
                names="t,time_s,shift_z_px,shift_y_px,shift_x_px,shift_z_um,"
                      "shift_y_um,shift_x_um,step_um,transit,smear_um,"
                      "z_estimate_planes,z_estimate_um")
            print("[render] --no-register and no shifts.csv: using zero shifts")
        else:
            sys.exit("no shifts.csv -- run the 'register' stage first, "
                     "or pass --no-register")
    if args.stage == "register":
        return

    if args.stage == "preview":
        nT = arr.shape[0]
        pts = ([int(x) for x in args.preview_t.split(",")] if args.preview_t
               else [0, nT // 2, nT - 1])
        import imageio.v2 as iio
        for t, im in render_frames(args, cmeta, arr, tab, ts, preview_ts=pts):
            p = Path(args.outdir) / f"preview_t{t:03d}.png"
            iio.imwrite(p, im)
            print(f"[preview] wrote {p}")
        return

    out = args.out or str(Path(args.outdir) / "perfusion_3d.mp4")
    render_frames(args, cmeta, arr, tab, ts, out_mp4=out)


if __name__ == "__main__":
    main()
