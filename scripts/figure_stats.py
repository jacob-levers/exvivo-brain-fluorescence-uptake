#!/usr/bin/env python
"""Publication figure: mean uptake across brains, with error bars.

Reads out_batch/curves.csv (one row per brain per timepoint), interpolates every
brain onto a common time axis -- the recordings differ by a few seconds in their
frame timings -- and plots the mean with a shaded SD band plus discrete error
bars at intervals.  Error bars are SD between biological replicates, not SEM:
with n = 4 the point of the figure is the spread between animals, and SEM would
halve the bands and understate it.

    python scripts/figure_stats.py --in out_batch --out out_figs
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def model(t, b, A, t0, tau):
    return b + A * np.clip(1 - np.exp(-(t - t0) / tau), 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="out_batch")
    ap.add_argument("--out", dest="outdir", default="out_figs")
    ap.add_argument("--err", choices=["sd", "sem"], default="sd")
    ap.add_argument("--every", type=float, default=10.0,
                    help="spacing of discrete error bars, minutes")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    curves = defaultdict(list)
    for r in csv.DictReader(open(Path(a.indir) / "curves.csv")):
        curves[r["name"]].append((float(r["t_min"]), float(r["brain_minus_bath"])))
    names = sorted(curves)
    n = len(names)

    tmax = min(max(t for t, _ in curves[k]) for k in names)
    grid = np.arange(0, tmax + 1e-9, 0.5)
    Y = np.vstack([np.interp(grid, *map(np.array, zip(*sorted(curves[k]))))
                   for k in names])
    F = Y / Y[:, :1]                       # fold change vs first timepoint

    def stats(M):
        mu = M.mean(0)
        sd = M.std(0, ddof=1)
        return mu, (sd if a.err == "sd" else sd / np.sqrt(n))

    mu, err = stats(Y)
    fmu, ferr = stats(F)

    metrics = list(csv.DictReader(open(Path(a.indir) / "metrics.csv")))

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.direction": "out",
        "ytick.direction": "out", "legend.frameon": False,
    })
    COL = "#1b7837"
    # narrow panels for the per-brain parameters: they carry different units
    # (counts, minutes) and must not share a y-axis
    fig = plt.figure(figsize=(11.2, 3.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.35, .55, .55], wspace=.55)
    ax = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # (a) absolute
    mk = np.isclose(grid % a.every, 0)
    ax[0].fill_between(grid, mu - err, mu + err, color=COL, alpha=.20, lw=0)
    ax[0].plot(grid, mu, color=COL, lw=1.6)
    ax[0].errorbar(grid[mk], mu[mk], yerr=err[mk], fmt="o", ms=3.5,
                   color=COL, ecolor=COL, elinewidth=.9, capsize=2.5, lw=0)
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel("1-AMA in tissue\n(counts above bath)")
    ax[0].set_xlim(0, tmax); ax[0].set_ylim(0, None)

    # (b) fold change
    ax[1].fill_between(grid, fmu - ferr, fmu + ferr, color=COL, alpha=.20, lw=0)
    ax[1].plot(grid, fmu, color=COL, lw=1.6)
    ax[1].errorbar(grid[mk], fmu[mk], yerr=ferr[mk], fmt="o", ms=3.5,
                   color=COL, ecolor=COL, elinewidth=.9, capsize=2.5, lw=0)
    ax[1].axhline(1, color="0.6", lw=.7, ls=":")
    ax[1].set_xlabel("Time (min)")
    ax[1].set_ylabel("Fold change\n(relative to t = 0)")
    ax[1].set_xlim(0, tmax)

    # (c, d) per-brain parameters: every brain shown, with mean and SD
    rng = np.random.default_rng(0)
    pars = [("end", "Signal at 95 min", "Counts above bath"),
            ("half_t", "Time to half of rise", "Minutes")]
    for i, (key, lab, unit) in enumerate(pars):
        k = ax[2 + i]
        v = np.array([float(m[key]) for m in metrics])
        x = 1 + (rng.random(len(v)) - .5) * .30
        k.scatter(x, v, s=20, facecolor="white", edgecolor=COL, linewidth=1.1,
                  zorder=3, clip_on=False)
        k.errorbar(1, v.mean(), yerr=v.std(ddof=1), fmt="_", ms=20,
                   color="0.25", elinewidth=1.1, capsize=5, zorder=2)
        k.set_xticks([]); k.set_xlim(.55, 1.45)
        k.set_ylim(0, v.max() * 1.25)
        k.set_ylabel(unit)
        k.set_title(lab, fontsize=9, pad=6)
        k.spines["bottom"].set_visible(False)

    err_lab = "SD" if a.err == "sd" else "SEM"
    # offset in points, not axes fractions: the panels have different widths,
    # so a fractional offset puts the letter in a different physical place on
    # each one and collides with the narrow panels' labels
    for k, letter in zip(ax, "abcd"):
        k.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                   xytext=(-42, 20), textcoords="offset points",
                   fontsize=11, fontweight="bold", ha="left", va="bottom")
    # No caption is drawn into the figure: the legend belongs in the document,
    # where it can be edited and is searchable, not baked into the image.

    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_uptake_mean.{ext}", dpi=a.dpi, bbox_inches="tight")

    # ---- numbers for the legend ----
    lines = [f"n = {n} brains; error = {err_lab}", ""]
    for t in (0, 15, 30, 45, 60, 75, 95):
        if t <= tmax:
            i = int(np.argmin(abs(grid - t)))
            lines.append(f"  {t:3d} min   {mu[i]:6.2f} +/- {err[i]:.2f} counts   "
                         f"({fmu[i]:.2f} +/- {ferr[i]:.2f} fold)")
    lines.append("")
    for key, lab in [("end", "signal at 95 min"), ("at60", "signal at 60 min"),
                     ("onset", "onset (min)"), ("tau", "tau (min)"),
                     ("half_t", "half-time (min)"), ("plateau", "plateau"),
                     ("peak_rate", "peak rate (counts/min)")]:
        v = np.array([float(m[key]) for m in metrics])
        lines.append(f"  {lab:24s} {v.mean():7.2f} +/- {v.std(ddof=1):5.2f} "
                     f"(SD)   range {v.min():.2f}-{v.max():.2f}")
    txt = "\n".join(lines)
    (out / "fig_uptake_mean_values.txt").write_text(txt)
    print(txt)
    print(f"\nwrote {out}/fig_uptake_mean.png and .pdf")


if __name__ == "__main__":
    main()
