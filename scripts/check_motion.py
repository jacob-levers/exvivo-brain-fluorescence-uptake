#!/usr/bin/env python
"""Sample-motion check: does the brain move during the recording?

For every timepoint, measures the rigid 3D displacement of the sample relative
to the middle timepoint by search-bounded phase cross-correlation, separately
on each channel, and reports it in microns alongside the microscope's own stage
log.  If the measured motion stays below a pixel throughout, registration is
unnecessary and the recordings can be analysed as acquired.

Reads the raw ND2 files directly, one timepoint at a time; no cache is needed.
Each volume is binned in xy (default 4 x 4) for speed, high-pass filtered in xy
so that the correlation locks onto tissue structure rather than the smooth
illumination profile or the overall rise in drug signal, and windowed to
suppress edge effects.  Sub-pixel displacement comes from upsampled-DFT
refinement of the correlation peak.

    python scripts/check_motion.py "path/to/folder_of_nd2s" --out out_motion

Outputs
  <out>/motion.csv     per timepoint, per channel: dz, dy, dx and lateral (um)
  <out>/motion.png     displacement against time, one panel per recording
  <out>/summary.txt    maximum lateral and axial displacement per recording
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd2_perfusion_3d as P
from batch_uptake import drug_channel
from pcc_bounded import pcc_bounded


def bin_xy(v, b):
    """Block-mean a (Z, Y, X) volume by b in y and x."""
    if b <= 1:
        return v
    Z, Y, X = v.shape
    Yb, Xb = Y // b, X // b
    return v[:, :Yb * b, :Xb * b].reshape(Z, Yb, b, Xb, b).mean(axis=(2, 4))


def feature(v, xp, ndi, hp_px, clip_pct=99.99):
    """Clip outliers, high-pass in xy, and window in all three axes.

    The clip matters: a transient bright speck (a few dozen saturated pixels in
    one plane of one frame) otherwise dominates the high-passed volume and the
    correlation locks onto it, reporting a large spurious displacement for that
    frame only.  Clipping the top 0.01 % of voxels removes such specks without
    touching tissue structure.
    """
    v = xp.asarray(v, dtype=xp.float32)
    if clip_pct and clip_pct < 100:
        v = xp.minimum(v, xp.percentile(v, clip_pct))
    v = xp.clip(v - ndi.gaussian_filter(v, (0, hp_px, hp_px)), 0, None)
    Z, Y, X = v.shape
    wz = xp.hanning(Z).astype(xp.float32)
    wy = xp.hanning(Y).astype(xp.float32)
    wx = xp.hanning(X).astype(xp.float32)
    return v * wz[:, None, None] * wy[None, :, None] * wx[None, None, :]


def check_file(path, a, xp, ndi):
    import nd2

    meta = P.read_meta(str(path))
    dz, dy, dx = meta["dz_um"], meta["dy_um"] * a.bin, meta["dx_um"] * a.bin
    tmin = np.asarray(meta["times_s"]) / 60.0
    axes = meta["axes"]
    rem = [x for x in axes if x != "T"]
    order = [rem.index("C"), rem.index("Z"), rem.index("Y"), rem.index("X")]

    with nd2.ND2File(str(path)) as f:
        dk = f.to_dask()
        nT = f.sizes["T"]
        nZ = f.sizes["Z"]
        ci = drug_channel(f)
        names = [c.channel.name for c in f.metadata.channels]
        chans = [ci, 1 - ci]                       # drug channel first

        def features(t):
            vol = np.transpose(np.asarray(dk[t]), order).astype(np.float32)
            return {c: feature(bin_xy(vol[c], a.bin), xp, ndi, a.hp_px, a.clip_pct)
                    for c in chans}

        t_ref = nT // 2
        ref = features(t_ref)
        zb = min(a.z_bound, nZ // 4)
        bound = (zb, a.lat_bound, a.lat_bound)

        ts = list(range(0, nT, a.step))
        if ts[-1] != nT - 1:
            ts.append(nT - 1)
        rows = []
        for k, t in enumerate(ts):
            cur = ref if t == t_ref else features(t)
            for c in chans:
                s = pcc_bounded(ref[c], cur[c], max_shift=bound,
                                predicted=(0, 0, 0), xp=xp, upsample=a.upsample)
                uz, uy, ux = s[0] * dz, s[1] * dy, s[2] * dx
                rows.append(dict(t_index=t, t_min=float(tmin[t]), channel=c,
                                 channel_name=names[c],
                                 role="drug" if c == ci else "other",
                                 dz_um=float(uz), dy_um=float(uy), dx_um=float(ux),
                                 lateral_um=float(np.hypot(uy, ux))))
            if k % 20 == 0 or t == ts[-1]:
                print(f"         t={t + 1}/{nT}", flush=True)

    return dict(name=Path(path).stem, rows=rows, stage=meta.get("stage_um", {}),
                dz=dz, dx_raw=meta["dx_um"], dx_bin=dx, t_ref=t_ref,
                t_ref_min=float(tmin[t_ref]), n=len(ts), names=names, ci=ci)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default="out_motion")
    ap.add_argument("--bin", type=int, default=4, help="xy binning factor")
    ap.add_argument("--hp-px", type=float, default=10.0,
                    help="high-pass sigma in binned pixels")
    ap.add_argument("--lat-bound", type=int, default=100,
                    help="lateral search radius, binned pixels")
    ap.add_argument("--z-bound", type=int, default=10,
                    help="axial search radius, planes")
    ap.add_argument("--clip-pct", type=float, default=99.99,
                    help="clip voxels above this percentile before correlating "
                         "(removes transient bright specks; 100 = off)")
    ap.add_argument("--upsample", type=int, default=10)
    ap.add_argument("--step", type=int, default=1, help="check every Nth timepoint")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-gpu", dest="gpu", action="store_false", default=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.folder).glob("*.nd2"))
    if a.only:
        files = [f for f in files if f.stem in a.only]
    if not files:
        sys.exit("no .nd2 files found")

    xp, ndi, _ = P.backend(a.gpu)
    print(f"[motion] backend: {'GPU' if xp is not np else 'CPU'}")

    results = []
    for f in files:
        print(f"[motion] {f.name}", flush=True)
        results.append(check_file(f, a, xp, ndi))

    with open(out / "motion.csv", "w", newline="") as fh:
        cols = ["name", "t_index", "t_min", "channel", "channel_name", "role",
                "dz_um", "dy_um", "dx_um", "lateral_um"]
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            for row in r["rows"]:
                w.writerow([r["name"]] + [f"{row[c]:.4f}" if isinstance(row[c], float)
                                          else row[c] for c in cols[1:]])

    lines = [f"Sample motion relative to the middle timepoint; 3D phase "
             f"cross-correlation on {a.bin}x{a.bin}-binned, high-passed volumes.",
             f"One raw pixel = {results[0]['dx_raw']:.3f} um; one binned pixel = "
             f"{results[0]['dx_bin']:.3f} um.", ""]
    hdr = (f"{'recording':22s} {'channel':16s} {'max lateral':>12s} "
           f"{'RMS lateral':>12s} {'max |axial|':>12s} {'RMS axial':>10s}  stage travel")
    lines += [hdr, "-" * len(hdr)]
    for r in results:
        stage = ", ".join(f"{k} {v['range_um']:.2f}" for k, v in r["stage"].items())
        for role in ("drug", "other"):
            rr = [x for x in r["rows"] if x["role"] == role]
            lat = np.array([x["lateral_um"] for x in rr])
            axl = np.array([x["dz_um"] for x in rr])
            label = r["name"][:22] if role == "drug" else ""
            tail = f"  {stage} um" if role == "drug" else ""
            lines.append(f"{label:22s} {rr[0]['channel_name'][:16]:16s} "
                         f"{lat.max():9.2f} um {np.sqrt((lat ** 2).mean()):9.2f} um "
                         f"{abs(axl).max():9.2f} um {np.sqrt((axl ** 2).mean()):7.2f} um"
                         f"{tail}")
    lines += ["", f"Timepoints checked per recording: {results[0]['n']} "
              f"(step {a.step}); reference = middle timepoint."]
    txt = "\n".join(lines)
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False, sharey=True)
    for k, r in enumerate(results):
        axk = axs[0, k]
        for role, col in (("drug", "tab:green"), ("other", "tab:blue")):
            rr = [x for x in r["rows"] if x["role"] == role]
            t = [x["t_min"] for x in rr]
            axk.plot(t, [x["lateral_um"] for x in rr], color=col, lw=1.3,
                     label=f"{rr[0]['channel_name']} lateral")
            axk.plot(t, [x["dz_um"] for x in rr], color=col, lw=1.0, ls="--",
                     label=f"{rr[0]['channel_name']} axial")
        axk.axhline(r["dx_raw"], color="0.5", lw=.7, ls=":")
        axk.axhline(-r["dx_raw"], color="0.5", lw=.7, ls=":")
        axk.set_title(r["name"], fontsize=9)
        axk.set_xlabel("Time (min)")
        axk.grid(alpha=.3)
    axs[0, 0].set_ylabel("Displacement (µm)")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Sample motion vs middle timepoint "
                 "(dotted: ± one raw pixel)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, .93])
    plt.savefig(out / "motion.png", dpi=150)
    print(f"\n[motion] wrote {out}/motion.csv, motion.png, summary.txt")


if __name__ == "__main__":
    main()
