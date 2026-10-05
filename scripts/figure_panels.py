#!/usr/bin/env python
"""Publication figure: the four brains over time, centred, levelled, one scale.

Each brain is centred on its own signal centroid and rotated so its long axis is
horizontal, so the panels are comparable rather than merely adjacent.  Both are
measured from the render itself and then corrected, rather than computed from
the volume and hoped for: the volume's array coordinates and the renderer's
screen coordinates differ in handedness, so a value derived from the volume has
the wrong sign for the renderer.

All brains are rendered at the same microns-per-pixel (zoom is solved per brain
from its own bounding radius) and the same fixed intensity window, so panel
brightness and object size are directly comparable across the figure.

    python scripts/figure_panels.py --out out_figs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd2_perfusion_3d as P
from scipy import ndimage as sndi

BRAINS = [("out4", 0, "Brain 1"),
          ("out5", 1, "Brain 2"),
          ("out6", 1, "Brain 3"),
          ("out7", 1, "Brain 4")]


class A:
    bin = 2; gpu = True; baseline_n = 5; bg_sigma = 24.0; force_bg = False
    bg_mode = "plane-const"; bg_pct = 50.0; crop_pad_um = 60.0

    def __init__(self, cd, sig):
        self.outdir = self.cache_dir = cd
        self.signal_channel = sig


def rendered_shape(img, frac=0.06):
    """Centroid and principal-axis angle of what actually got drawn."""
    g = img.sum(2)
    m = g > frac * g.max()
    ys, xs = np.where(m)
    cy, cx = ys.mean(), xs.mean()
    cov = np.cov(np.vstack([xs - cx, ys - cy]))
    ev, evec = np.linalg.eigh(cov)
    mj = evec[:, np.argmax(ev)]
    ang = np.degrees(np.arctan2(mj[1], mj[0]))
    return cx, cy, (ang + 90) % 180 - 90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_figs")
    ap.add_argument("--times", default="0,20,45,70,95",
                    help="timepoints to show, minutes")
    ap.add_argument("--upp", type=float, default=1.2, help="microns per pixel")
    ap.add_argument("--pw", type=int, default=720)
    ap.add_argument("--ph", type=int, default=600)
    ap.add_argument("--vmin", type=float, default=8.0)
    ap.add_argument("--vmax", type=float, default=72.0)
    ap.add_argument("--emit", type=float, default=0.65)
    ap.add_argument("--opacity", type=float, default=0.3)
    ap.add_argument("--samples", type=int, default=560)
    ap.add_argument("--cmap", default="green")
    ap.add_argument("--scalebar", type=float, default=200.0)
    ap.add_argument("--dpi", type=int, default=350)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    want_min = [float(x) for x in a.times.split(",")]
    W, H = a.pw, a.ph

    panels = {}
    for cd, sig, label in BRAINS:
        args = A(cd, sig)
        arr, cm = P.load_cache(args)
        bg = P.compute_background(arr, cm, args)
        xp, ndi, _ = P.backend(True)
        nT, nC, nZ, Yb, Xb = arr.shape
        ts = np.asarray(cm["times_s"]) / 60.0
        zeros = np.zeros((nT, 3))
        rc = P.Raycaster((nZ, Yb, Xb), (cm["dz_um"], cm["dy_um"], cm["dx_um"]),
                         W, H, xp, ndi)
        lut = xp.asarray(P.make_lut(a.cmap))
        zoom = 2 * rc.radius / (W * a.upp)

        def draw(vol, roll, pan, ns):
            return P.to_host(rc.render([vol], [lut], [a.vmin], [a.vmax],
                                       [a.opacity], 0.0, 0.0, n_samples=ns,
                                       zoom=zoom, emit=[a.emit], roll=roll,
                                       pan=pan))

        # --- solve roll and pan on the final frame, from the render itself ---
        vlast = P.build_volume(arr, bg, zeros, nT - 1, [sig], xp, ndi)[0]
        m = P.to_host(vlast).max(axis=0)
        sm = sndi.gaussian_filter(m, 8)
        msk = sm > np.percentile(sm, 78)
        msk = sndi.binary_fill_holes(sndi.binary_closing(msk, iterations=6))
        lbl, n = sndi.label(msk)
        if n:
            msk = lbl == (1 + int(np.argmax(sndi.sum(msk, lbl, range(1, n + 1)))))
        ys, xs = np.where(msk)
        pan = (-cm["dx_um"] * (xs.mean() - (Xb - 1) / 2),
               -cm["dy_um"] * (ys.mean() - (Yb - 1) / 2))

        _, _, ang0 = rendered_shape(draw(vlast, 0.0, pan, 260))
        roll = -ang0                                   # levels the long axis
        for _ in range(2):                             # exact in one, twice is cheap
            cx, cy, ang = rendered_shape(draw(vlast, roll, pan, 260))
            uc = (cx - (W - 1) / 2) * a.upp
            vc = ((H - 1) / 2 - cy) * a.upp
            cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
            pan = (pan[0] - (uc * cr - vc * sr), pan[1] - (uc * sr + vc * cr))
        cx, cy, ang = rendered_shape(draw(vlast, roll, pan, 260))
        print(f"{label}: roll {roll:+6.2f} deg, pan ({pan[0]:+7.1f},{pan[1]:+7.1f}) um "
              f"-> centre ({cx:5.1f},{cy:5.1f}) of ({W/2:.0f},{H/2:.0f}), "
              f"residual angle {ang:+.2f} deg", flush=True)

        row = []
        for tm in want_min:
            ti = int(np.argmin(abs(ts - tm)))
            vol = P.build_volume(arr, bg, zeros, ti, [sig], xp, ndi)[0]
            row.append(((np.clip(draw(vol, roll, pan, a.samples), 0, 1) * 255)
                        .astype(np.uint8), ts[ti]))
        panels[label] = row

    # ---------------- compose ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9})

    nr, nc = len(BRAINS), len(want_min)
    fig, ax = plt.subplots(nr, nc, figsize=(nc * 1.62, nr * 1.42))
    for r, (_, _, label) in enumerate(BRAINS):
        for c, (img, tmin) in enumerate(panels[label]):
            k = ax[r, c]
            k.imshow(img, interpolation="bilinear")
            k.set_xticks([]); k.set_yticks([])
            for s in k.spines.values():
                s.set_visible(False)
            if r == 0:
                k.set_title(f"{tmin:.0f} min", fontsize=9.5, pad=5)
            if c == 0:
                k.set_ylabel(label, fontsize=9.5, labelpad=6)
    # one scale bar, bottom-right panel
    k = ax[nr - 1, nc - 1]
    px = a.scalebar / a.upp
    k.add_patch(plt.Rectangle((W - px - 0.06 * W, H - 0.13 * H), px, 0.022 * H,
                              color="white"))
    k.text(W - px / 2 - 0.06 * W, H - 0.155 * H, f"{a.scalebar:g} µm",
           color="white", ha="center", va="bottom", fontsize=8)

    fig.subplots_adjust(left=.055, right=.90, top=.94, bottom=.02,
                        wspace=.03, hspace=.03)
    cols = P.make_lut(a.cmap)
    cb = fig.add_axes([0.915, 0.30, 0.013, 0.40])
    sm_ = ScalarMappable(norm=Normalize(a.vmin, a.vmax),
                         cmap=LinearSegmentedColormap.from_list("f", cols))
    bar = fig.colorbar(sm_, cax=cb)
    bar.set_label("1-AMA (counts above bath)", fontsize=8.5, labelpad=6)
    bar.ax.tick_params(labelsize=8, length=2.5)
    bar.outline.set_linewidth(0.6)

    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_brains_panels.{ext}", dpi=a.dpi,
                    bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}/fig_brains_panels.png and .pdf")
    print(f"scale {a.upp} um/px, field {W*a.upp:.0f} x {H*a.upp:.0f} um, "
          f"window [{a.vmin:g}, {a.vmax:g}]")


if __name__ == "__main__":
    main()
