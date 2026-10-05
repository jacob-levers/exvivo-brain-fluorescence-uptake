# Fluorescent drug uptake in ex vivo *Drosophila* brain — imaging analysis

Analysis and visualisation code for spinning-disk confocal timelapses of
ex vivo adult *Drosophila* brains exposed to 1-aminoanthracene (1-AMA). 1-AMA is
itself fluorescent, so its accumulation in tissue can be measured directly from
the 488 nm channel over time.

The pipeline reads Nikon `.nd2` recordings directly, measures drug uptake as
brain-minus-bath signal across the full imaged depth, fits uptake kinetics,
and renders 3D timelapses and publication figures on a fixed intensity scale.

---

## What it does

| Script | Purpose |
|---|---|
| `nd2_perfusion_3d.py` | Main pipeline: read ND2 lazily, build a binned cache, register, render 3D timelapses (MP4) |
| `batch_uptake.py` | Quantify uptake for every recording in a folder; fit kinetics; per-brain metrics and sample-size estimates |
| `check_motion.py` | Measure 3D sample motion at every timepoint on both channels, to decide whether registration is needed |
| `figure_stats.py` | Figure: mean ± SD uptake curve, fold change, per-brain parameters |
| `figure_panels.py` | Figure: brains × timepoints grid, each brain centred and levelled, common scale |
| `make_results_handoff.py` | Write a complete Markdown summary of all results from the analysed data |
| `viewer.py` | Interactive browser viewer for choosing a rendering viewpoint; outputs the matching render command |
| `roi_tool.py` | Browser tool for drawing a region to mask stray signal out of rendered frames |
| `pcc_bounded.py` | Search-bounded phase cross-correlation used for registration (with self-test) |
| `test_depth.py` | Correctness test for the renderer's depth ordering and cutaway geometry |

## Installation

Python 3.12 was used.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

A CUDA GPU is optional. With CuPy installed (`pip install cupy-cuda12x`),
rendering and registration run on the GPU; without it every script falls back
to NumPy/SciPy on the CPU and produces the same results, more slowly.

Each recording used here is about 160 GB. Files are read one timepoint at a
time, so memory use stays small, but the binned cache for one recording takes
about 80 GB of disk.

## Usage

### 1. Inspect a recording

```bash
python scripts/nd2_perfusion_3d.py probe --nd2 "recording.nd2" --outdir out_rec
```

Prints dimensions, voxel size, z-step, frame timing, channel configuration and
stage travel, all read from the file's own metadata.

### 2. Quantify uptake across recordings

```bash
python scripts/batch_uptake.py "path/to/folder_of_nd2s" --out out_batch \
    --mask-channel other --mask-frame last --n-samples 191
```

Reads the raw files directly; no cache needed. Writes `metrics.csv`,
`curves.csv`, `comparison.png` and `summary.txt` (including sample-size
estimates). The drug channel is identified by excitation wavelength (488 nm),
not channel index, because channel order is not guaranteed to be the same
between files.

### Check for sample motion

```bash
python scripts/check_motion.py "path/to/folder_of_nd2s" --out out_motion
```

Measures rigid 3D displacement of every timepoint against the middle one by
search-bounded phase cross-correlation, independently on each channel, and
reports it alongside the stage log. Writes `motion.csv`, `motion.png` and
`summary.txt`. If motion stays below a pixel, the recordings can be analysed
without registration. Reads the raw ND2 files; no cache needed.

### 3. Figures

```bash
python scripts/figure_stats.py  --in out_batch --out out_figs
python scripts/figure_panels.py --out out_figs
```

`figure_panels.py` and `make_results_handoff.py` expect a binned cache for each
brain (step 4, `cache` stage); the cache folders are listed at the top of each
script.

### 4. Cache and render a 3D timelapse

```bash
# one streaming pass over the ND2 -> 2x2-binned float32 cache
python scripts/nd2_perfusion_3d.py cache --nd2 "recording.nd2" \
    --outdir out_rec --cache-dir out_rec --bin 2

# render: top-down view, levelled, green LUT, fixed scale, minimal overlay
python scripts/nd2_perfusion_3d.py render --nd2 "recording.nd2" \
    --outdir out_rec --cache-dir out_rec --bin 2 \
    --signal-channel 1 --bg-mode plane-const --bg-pct 50 --no-register \
    --crop --crop-frac 0.3 --vmin 8 --vmax 72 --emit 0.65 --opacity 0.3 \
    --roll -0.4 --zoom 1.4 --width 900 --height 620 \
    --cmap green --overlay minimal --signal-label "1-AMA" \
    --out out_rec/timelapse.mp4
```

Run `python scripts/nd2_perfusion_3d.py --help` for every option, including
oblique views, cutaways, z-slabs and two-channel overlays.

### 5. Interactive tools

```bash
python scripts/viewer.py   --nd2 "recording.nd2" --cache-dir out_rec --bin 2
python scripts/roi_tool.py --cache-dir out_rec --bin 2 --signal-channel 1 \
    --out out_rec/roi.json
```

Both open in a browser at `http://127.0.0.1:<port>`. The viewer writes out the
render command for whatever view is on screen; the ROI tool saves a mask that
`render` accepts via `--roi`.

### Tests

```bash
python scripts/pcc_bounded.py    # registration: sign, sub-pixel accuracy, decoy rejection
python scripts/test_depth.py     # renderer: depth order and cutaway geometry
```

## Method summary

**Reading.** ND2 files are read lazily as dask arrays, one timepoint at a
time. Voxel dimensions, timestamps and channel wavelengths are taken from the
file. Quantification (`batch_uptake.py`) uses the data at full resolution;
a 2 × 2-binned cache is built separately for rendering and figures.

**Regions.** The brain region is segmented automatically from the 405 nm
channel at the final timepoint (Gaussian smoothing σ = 8 px, 78th-percentile
threshold, closing, hole filling, largest connected component) and applied
unchanged to every timepoint and both channels. The background region is the
bath beyond a 60-pixel (≈33 µm) margin around the brain.

**Measurement.** At each timepoint the mean projection through all z-planes is
taken, rather than a single optical section, and signal is the mean intensity
in the brain region minus the mean in the background region.

**Background in rendered images.** For movies and figures, one constant per
z-plane — the bath level, the 50th percentile of the first five timepoints —
is subtracted from every frame. It is fixed in time, so it cannot alter the
time course, and it preserves drug already present in tissue at the start of
acquisition. (For the quantified brain-minus-bath signal such a constant cancels
in the difference, so the two approaches agree.)

**Intensity scale.** One fixed intensity window is used for every frame of a
movie and every panel of a figure. No bleach correction or per-frame
normalisation is applied: the rise in signal is the result being measured, and
standard bleach correction assumes total intensity is conserved, which is
false while fluorophore is entering the tissue.

**Registration.** Rigid translation by search-bounded phase cross-correlation,
computed on the structural channel only and applied to all channels. Bounding
the search prevents the correlation peak hopping between candidates on weak
signal. Where measured motion is below one pixel, registration is skipped
(`--no-register`).

**Rendering.** A GPU emission–absorption raycaster marches rays through
physical micron coordinates using the true anisotropic voxel spacing, so the
volume is never resampled to isotropic. Rendered images are visualisations;
quantitative values come from the measured voxel intensities, not from
rendered pixel brightness.

## Notes

- The optical configurations on the microscope are named *DAPI Confocal*
  (405 nm excitation, 450/50 emission) and *A488 Confocal* (488 nm, 525/50).
  These are preset names only; no DAPI stain was used. The 405 nm channel
  records tissue autofluorescence plus some 1-AMA emission.
- Intensities are 16-bit camera counts above bath: linear and consistent
  between recordings, but not calibrated to concentration.
- Raw data are not included in this repository.

## Licence

Released under the MIT Licence — see [`LICENSE`](LICENSE).

## Acknowledgement

This code was developed with the help of AI assistance.
