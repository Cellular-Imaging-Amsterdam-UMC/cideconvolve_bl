"""
Diagnostic test: coarse-to-fine RI scan on channel 1 of Dendrites_Crop.ome.tiff.

Ground-truth parameters (from OME-XML / reference acquisition metadata):
  pixel XY : 29 nm
  pixel Z  : 150 nm
  NA       : 1.4
  immersion: Oil  (RI 1.515)
  medium   : ~1.338  (water-like reference value)
  channel 1: ex=564 nm, em=600 nm, pinhole ~1 AU

Expected result: best RI should be in the range 1.33–1.43, NOT 1.50–1.515.

Run as:
  conda activate deconvolve
  cd C:\rahoebe\Python\cideconvolve
  python tests/test_ri_fit_dendrites_ch1.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import tifffile

# make sure the repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.deconvolve_ci import ci_fit_psf_params  # noqa: E402

# ---------------------------------------------------------------------------
# Load image
# ---------------------------------------------------------------------------
IMAGE_PATH = Path(__file__).resolve().parent.parent / "localdata" / "Dendrites_Crop.ome.tiff"
print(f"Loading {IMAGE_PATH} ...")
with tifffile.TiffFile(IMAGE_PATH) as tf:
    arr = tf.asarray()          # CZYX after series conversion
    series = tf.series[0]
    axes = series.axes           # e.g. "CZYX"

# Reorder to CZYX if necessary
if axes == "CZYX":
    pass
elif axes == "ZCYX":
    arr = arr.transpose(1, 0, 2, 3)
elif axes == "TCZYX":
    arr = arr[0]
else:
    raise ValueError(f"Unexpected axes: {axes}")

ch1 = arr[1].astype(np.float32)   # second channel, shape (Z, Y, X)
print(f"Channel 1 shape: {ch1.shape}  dtype: {ch1.dtype}  range: {ch1.min():.1f}–{ch1.max():.1f}")

# ---------------------------------------------------------------------------
# Acquisition parameters (from OME-XML)
# ---------------------------------------------------------------------------
pixel_xy_nm = 29.0
pixel_z_nm  = 150.0
na          = 1.4
ri_immersion = 1.515
emission_nm  = 600.0
excitation_nm = 564.0
# Pinhole physical size = 6.2 µm; 1 AU = 1.22*600 nm/1.4 = 522.9 nm = 0.5229 µm
# AU = 6.2 / 0.5229 ≈ 11.9 → very open.  We'll also test with 1.0 AU (closed).
pinhole_au = 1.0

# z_p: estimated depth offset of the focus into the sample.
# Simple heuristic: half the stack depth.
n_z = ch1.shape[0]
z_p_nm = 0.5 * (n_z - 1) * pixel_z_nm      # ≈ 3900 nm

base_params = {
    "microscope_type":      "confocal",
    "na":                   na,
    "ri_immersion":         ri_immersion,
    "ri_sample":            1.47,           # starting guess (will be scanned)
    "pixel_size_xy_nm":     pixel_xy_nm,
    "pixel_size_z_nm":      pixel_z_nm,
    "emission_nm":          emission_nm,
    "excitation_nm":        excitation_nm,
    "pinhole_airy_units":   pinhole_au,
    "z_p":                  z_p_nm,
    "niter":                50,
}

# ---------------------------------------------------------------------------
# Stage 1 – coarse scan
# ---------------------------------------------------------------------------
COARSE_GRID = [1.330, 1.358, 1.385, 1.410, 1.435, 1.460, 1.480, 1.500, 1.515]

def progress_cb(trial_idx, total, trial_params, residual):
    ri   = trial_params.get("ri_sample", "?")
    print(f"  [{trial_idx}/{total}] RI={ri:.4f}  residual={residual:.5g}")

print("\n=== COARSE SCAN ===")
t0 = time.time()
result_coarse = ci_fit_psf_params(
    ch1,
    base_params,
    ri_sample_grid=COARSE_GRID,
    niter_fit=40,
    callback=progress_cb,
)
coarse_elapsed = time.time() - t0
coarse_log = result_coarse["search_log"]
coarse_best_ri = result_coarse["best_params"]["ri_sample"]

print(f"\nCoarse scan took {coarse_elapsed:.1f}s")
print(f"Coarse best RI: {coarse_best_ri:.4f}")
print("\nCoarse score table (sorted by i_div):")
sorted_log = sorted(coarse_log, key=lambda e: e.get("i_div", float("inf")))
for e in sorted_log:
    params = e.get("params", e)
    ri   = float(params.get("ri_sample", float("nan")))
    idiv = float(e.get("i_div", float("nan")))
    sc   = float(e.get("score", float("nan")))
    res  = float(e.get("residual", float("nan")))
    rou  = float(e.get("roughness", float("nan")))
    print(f"  RI={ri:.4f}  i_div={idiv:.5g}  score={sc:.5g}  residual={res:.5g}  roughness={rou:.5g}")

# ---------------------------------------------------------------------------
# Stage 2 – fine scan around coarse best
# ---------------------------------------------------------------------------
FINE_HALF = 0.015
fine_lo = max(1.330, coarse_best_ri - FINE_HALF)
fine_hi = min(1.515, coarse_best_ri + FINE_HALF)
FINE_GRID = [round(v, 4) for v in np.linspace(fine_lo, fine_hi, 9)]
print(f"\n=== FINE SCAN  (grid: {FINE_GRID}) ===")
t1 = time.time()
result_fine = ci_fit_psf_params(
    ch1,
    base_params,
    ri_sample_grid=FINE_GRID,
    niter_fit=40,
    callback=progress_cb,
)
fine_elapsed = time.time() - t1
fine_log = result_fine["search_log"]
fine_best_ri = result_fine["best_params"]["ri_sample"]

print(f"\nFine scan took {fine_elapsed:.1f}s")
print(f"Fine best RI: {fine_best_ri:.4f}")
print("\nFine score table (sorted by i_div):")
sorted_fine = sorted(fine_log, key=lambda e: e.get("i_div", float("inf")))
for e in sorted_fine:
    params = e.get("params", e)
    ri   = float(params.get("ri_sample", float("nan")))
    idiv = float(e.get("i_div", float("nan")))
    sc   = float(e.get("score", float("nan")))
    res  = float(e.get("residual", float("nan")))
    rou  = float(e.get("roughness", float("nan")))
    print(f"  RI={ri:.4f}  i_div={idiv:.5g}  score={sc:.5g}  residual={res:.5g}  roughness={rou:.5g}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Coarse best RI : {coarse_best_ri:.4f}")
print(f"  Fine best RI   : {fine_best_ri:.4f}")
print(f"  Expected ~     : 1.338  (reference medium RI)")
total = coarse_elapsed + fine_elapsed
print(f"  Total scan time: {total:.1f}s")

if fine_best_ri <= 1.44:
    print("  RESULT: PASS — realistic medium RI found (<= 1.44)")
else:
    print(f"  RESULT: WARN — RI {fine_best_ri:.4f} is unexpectedly high; check algorithm.")
