#!/usr/bin/env python
"""Uptake curves for every ND2 in a folder, on one axis, with per-brain metrics.

Reads each file directly (a subset of timepoints, no cache needed), finds the
drug channel from the excitation wavelength in the metadata rather than from
the channel index -- the two have been in either order in this series -- and
measures brain-minus-bath over time.  Fits a delayed first-order uptake to each
curve and writes one row per brain to a CSV, so the between-brain spread needed
for a sample-size calculation accumulates as recordings are added.

    python scripts/batch_uptake.py "C:/Users/.../Brains" --out out_batch

Outputs
  <out>/metrics.csv          one row per brain
  <out>/curves.csv           every sampled point, every brain
  <out>/comparison.png       overlaid curves, fold change, rates
  <out>/summary.txt          means, SDs, and n-per-group for a range of effects
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

DRUG_EX_NM = 488.0          # the 1-AMA channel in this series is the 488 line


def model(t, b, A, t0, tau):
    return b + A * np.clip(1 - np.exp(-(t - t0) / tau), 0, None)


def drug_channel(f):
    """Index of the channel excited at DRUG_EX_NM; falls back to the emission
    filter text if the excitation field is empty."""
    for i, c in enumerate(f.metadata.channels):
        ex = c.channel.excitationLambdaNm
        if ex is not None and abs(float(ex) - DRUG_EX_NM) < 5:
            return i
    desc = f.text_info.get("description", "") or ""
    for i, marker in enumerate(("Plane #1", "Plane #2", "Plane #3")):
        j = desc.find(marker)
        if j >= 0 and "525/50" in desc[j:j + 2500]:
            return i
    raise SystemExit(f"cannot identify the {DRUG_EX_NM:.0f} nm channel in {f.path}")


def measure(path, n_samples=20, brain_pct=78.0, bath_gap_px=60,
            mask_channel="drug", mask_frame="last"):
    import nd2
    from scipy import ndimage as ndi

    with nd2.ND2File(path) as f:
        dk = f.to_dask()
        sizes = dict(f.sizes)
        nT = sizes["T"]
        ci = drug_channel(f)
        cname = f.metadata.channels[ci].channel.name
        ev = f.events(orient="list")
        tt = np.asarray(ev["Time [s]"], float)
        ti = np.asarray(ev["T Index"], int)
        tsec = np.array([tt[ti == k].min() for k in range(nT)])
        tsec -= tsec[0]
        zstep = float(f.voxel_size().z)
        date = (f.text_info or {}).get("date")

        # Brain mask.  Which channel and which frame it is taken from is a
        # deliberate choice: taking it from the drug channel at the end gives
        # the clearest outline, but defines the measurement region using the
        # measurement signal.  Taking it from the other channel at the start
        # is independent of the drug signal being measured.
        mi = ci if mask_channel == "drug" else (1 - ci)
        mt = (nT - 1) if mask_frame == "last" else 0
        late = np.asarray(dk[mt]).astype(np.float32)[:, mi].max(axis=0)
        sm = ndi.gaussian_filter(late, 8)
        brain = sm > np.percentile(sm, brain_pct)
        brain = ndi.binary_fill_holes(ndi.binary_closing(brain, iterations=6))
        lbl, n = ndi.label(brain)
        if n:
            brain = lbl == (1 + int(np.argmax(ndi.sum(brain, lbl, range(1, n + 1)))))
        bath = ~ndi.binary_dilation(brain, iterations=bath_gap_px)

        ts = np.unique(np.r_[np.linspace(0, nT - 1, n_samples).astype(int), nT - 1])
        rows = []
        for T in ts:
            v = np.asarray(dk[T]).astype(np.float32)[:, ci].mean(axis=0)
            rows.append((tsec[T] / 60.0, float(v[brain].mean()), float(v[bath].mean())))

    r = np.array(rows)
    return dict(path=str(path), name=Path(path).stem, channel=ci, channel_name=cname,
                n_t=nT, minutes=tsec[-1] / 60.0, z_step_um=zstep, date=date,
                brain_frac=float(brain.mean()), t=r[:, 0], brain=r[:, 1], bath=r[:, 2])


def fit(t, d):
    from scipy.optimize import curve_fit
    p, _ = curve_fit(model, t, d, p0=[d[0], max(d[-1] - d[0], 1), 5.0, 30.0],
                     bounds=([-np.inf, 0, 0, 1], [np.inf, np.inf, 60, 1000]),
                     maxfev=50000)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default="out_batch")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--mask-channel", choices=["drug", "other"], default="drug",
                    help="channel the brain region is derived from")
    ap.add_argument("--mask-frame", choices=["first", "last"], default="last",
                    help="timepoint the brain region is derived from")
    ap.add_argument("--only", nargs="*", default=None,
                    help="process only these stems (default: every .nd2)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.folder).glob("*.nd2"))
    if a.only:
        files = [f for f in files if f.stem in a.only]
    if not files:
        sys.exit("no .nd2 files found")

    results = []
    for f in files:
        print(f"[batch] {f.name}", flush=True)
        m = measure(f, n_samples=a.n_samples, mask_channel=a.mask_channel,
                    mask_frame=a.mask_frame)
        d = m["brain"] - m["bath"]
        p = fit(m["t"], d)
        b, A, t0, tau = p
        plateau = b + A
        half_t = t0 + tau * np.log(2)
        tf = np.linspace(0, m["t"][-1], 400)
        rate = np.gradient(model(tf, *p), tf)
        m.update(diff=d, fit=p, plateau=plateau, onset=t0, tau=tau,
                 half_t=half_t, peak_rate=float(rate.max()),
                 end=float(d[-1]), start=float(d[0]),
                 at30=float(np.interp(30, m["t"], d)),
                 at60=float(np.interp(60, m["t"], d)),
                 pct_plateau=100 * d[-1] / plateau,
                 rise_pct=100 * (d[-1] / d[0] - 1))
        results.append(m)
        print(f"         drug channel c{m['channel']} ({m['channel_name']}), "
              f"{m['minutes']:.1f} min, z {m['z_step_um']:.2f} um | "
              f"{d[0]:.2f} -> {d[-1]:.2f} (+{m['rise_pct']:.0f}%)  "
              f"onset {t0:.1f}  tau {tau:.1f}  plateau {plateau:.1f}")

    # ---------- CSVs ----------
    cols = ["name", "date", "channel", "minutes", "z_step_um", "brain_frac",
            "start", "at30", "at60", "end", "rise_pct", "onset", "tau",
            "plateau", "pct_plateau", "half_t", "peak_rate"]
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in results:
            w.writerow([m[c] if not isinstance(m[c], float) else f"{m[c]:.4f}"
                        for c in cols])
    with open(out / "curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "t_min", "brain", "bath", "brain_minus_bath"])
        for m in results:
            for t, br, ba, d in zip(m["t"], m["brain"], m["bath"], m["diff"]):
                w.writerow([m["name"], f"{t:.3f}", f"{br:.3f}", f"{ba:.3f}", f"{d:.3f}"])

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for k, m in enumerate(results):
        c = cmap(k % 10)
        tf = np.linspace(0, m["t"][-1], 400)
        ax[0].plot(m["t"], m["diff"], "o", ms=4, color=c, label=m["name"])
        ax[0].plot(tf, model(tf, *m["fit"]), "--", color=c, lw=1)
        ax[1].plot(m["t"], m["diff"] / m["diff"][0], "o-", ms=3, color=c, label=m["name"])
        ax[2].plot(tf, np.gradient(model(tf, *m["fit"]), tf), color=c, label=m["name"])
    ends = np.array([m["end"] for m in results])
    ax[0].axhspan(ends.mean() - ends.std(ddof=1) if len(ends) > 1 else ends.mean(),
                  ends.mean() + ends.std(ddof=1) if len(ends) > 1 else ends.mean(),
                  color="k", alpha=.06, lw=0)
    ax[0].set_ylabel("1-AMA, counts above bath"); ax[0].set_title("Absolute uptake (band = mean ± SD at end)")
    ax[1].set_ylabel("fold change vs t=0"); ax[1].set_title("Relative uptake")
    ax[2].set_ylabel("uptake rate (counts / min)"); ax[2].set_title("Rate (from fit)")
    for x in ax:
        x.set_xlabel("time (min)"); x.grid(alpha=.3); x.legend(fontsize=8)
    plt.suptitle(f"{len(results)} brains, same acquisition, same analysis")
    plt.tight_layout(rect=[0, 0, 1, .93])
    plt.savefig(out / "comparison.png", dpi=130)

    # ---------- summary + n ----------
    lines = []
    n = len(results)
    lines.append(f"{n} brains\n")
    lines.append(f"{'metric':16s} " + " ".join(f"{m['name'][:14]:>14s}" for m in results)
                 + f" {'mean':>9s} {'SD':>8s} {'CV%':>6s}")
    for key, lab in [("end", "end (95 min)"), ("at30", "at 30 min"),
                     ("at60", "at 60 min"), ("rise_pct", "rise %"),
                     ("onset", "onset (min)"), ("tau", "tau (min)"),
                     ("plateau", "plateau"), ("half_t", "half-time (min)"),
                     ("peak_rate", "peak rate")]:
        v = np.array([m[key] for m in results], float)
        sd = v.std(ddof=1) if n > 1 else float("nan")
        cv = 100 * sd / v.mean() if v.mean() else float("nan")
        lines.append(f"{lab:16s} " + " ".join(f"{x:14.2f}" for x in v)
                     + f" {v.mean():9.2f} {sd:8.2f} {cv:6.1f}")
    lines.append("")
    if n >= 3:
        lines.append("n per group for a two-sample t-test, alpha 0.05 two-sided, "
                     "80% power  (n ~ 15.7 (SD/delta)^2 + 1)")
        lines.append(f"  SD estimated from {n} brains -- treat as rough until n >= 5")
        for key, lab, deltas in [("end", "end (95 min)", (1, 2, 3)),
                                 ("half_t", "half-time (min)", (10, 15, 20)),
                                 ("peak_rate", "peak rate", (0.05, 0.1))]:
            v = np.array([m[key] for m in results], float)
            sd = v.std(ddof=1)
            s = ", ".join(f"delta {d:g} -> n {int(np.ceil(15.7 * (sd / d) ** 2 + 1))}"
                          for d in deltas)
            lines.append(f"  {lab:16s} SD {sd:6.2f}:  {s}")
    txt = "\n".join(lines)
    (out / "summary.txt").write_text(txt)
    print("\n" + txt)
    print(f"\n[batch] wrote {out}/metrics.csv, curves.csv, comparison.png, summary.txt")


if __name__ == "__main__":
    main()
