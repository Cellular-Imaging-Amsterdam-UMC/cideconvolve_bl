"""Diagnostic script: does 'Raw - estimated' fade to black at convergence?

Tests:
1. At each RL iteration: NMAE(PSF*deconv_k, raw) to check convergence trend
2. After DL refinement: NMAE(PSF*DL, raw) and sign of mean(PSF*DL - raw)
3. Plot convergence + difference image; save CSV

Usage:
    C:\\Users\\p000881\\AppData\\Local\\miniconda3\\envs\\deconvolve\\python.exe tests/test_raw_estimated_residual.py

Data:
    localdata/Dendrites_Crop.ome.tiff
    models/Full-Mix-Test-v2/best_model.pt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deconvolve import deconvolve_image
from deconvolve_ci import ci_rl_deconvolve
from deconvolve_ci_dl import deconvolve_ci_rl_dl, reconvolve_same

IMAGE_PATH = REPO_ROOT / "localdata" / "Dendrites_Crop.ome.tiff"
MODEL_PATH = REPO_ROOT / "models" / "Full-Mix-Test-v2" / "best_model.pt"
OUTPUT_DIR = REPO_ROOT / "tests" / "residual_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nmae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error normalised by p99 of b."""
    b_arr = np.asarray(b, dtype=np.float32)
    scale = max(float(np.percentile(b_arr, 99)), 1e-6)
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float32) - b_arr)) / scale)


def mip(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr.max(axis=0) if arr.ndim == 3 else arr


def pclip(img: np.ndarray, lo: float = 0.1, hi: float = 99.9) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    lo_v = float(np.percentile(arr, lo))
    hi_v = float(np.percentile(arr, hi))
    return np.clip((arr - lo_v) / max(hi_v - lo_v, 1e-6), 0.0, 1.0)


def psf_2d_or_3d(psf: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Return 2-D or 3-D PSF matching raw dimensionality."""
    if raw.ndim == 2 and psf.ndim == 3:
        return psf[psf.shape[0] // 2]
    return psf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Image : {IMAGE_PATH}")
    print(f"Model : {MODEL_PATH}")

    # ------------------------------------------------------------------
    # Step 1: Load image + build PSF via deconvolve_image (ci_rl baseline)
    # ------------------------------------------------------------------
    print("\n=== Step 1: ci_rl baseline + PSF ===")
    baseline = deconvolve_image(
        IMAGE_PATH,
        method="ci_rl",
        niter=51,
        convergence="fixed",
        check_every=5,
    )

    raw_ch0 = np.asarray(baseline["source_channels"][0], dtype=np.float32)
    ci_rl_ch0 = np.asarray(baseline["channels"][0], dtype=np.float32)
    psf_raw = np.asarray(baseline["psfs"][0], dtype=np.float32)
    meta = baseline.get("metadata", {})

    psf = psf_2d_or_3d(psf_raw, raw_ch0)
    raw = raw_ch0
    ci_rl = ci_rl_ch0

    bg_p1 = float(np.percentile(raw, 1))
    raw_bgsub = np.clip(raw - bg_p1, 0.0, None)

    print(f"  raw shape  : {raw.shape}  dtype={raw.dtype}")
    print(f"  raw range  : [{raw.min():.1f}, {raw.max():.1f}]")
    print(f"  bg p1      : {bg_p1:.3f}")
    print(f"  psf shape  : {psf.shape}  sum={psf.sum():.6f}")
    print(f"  metadata   : {meta}")

    # ------------------------------------------------------------------
    # Step 2: ci_rl reconvolution quality
    # ------------------------------------------------------------------
    print("\n=== Step 2: ci_rl reconvolution quality ===")
    recon_ci = reconvolve_same(ci_rl, psf)
    diff_ci = recon_ci - raw

    print(f"  ci_rl sum           : {ci_rl.sum():.1f}  (raw sum: {raw.sum():.1f})")
    print(f"  PSF*ci_rl sum       : {recon_ci.sum():.1f}  (ratio vs raw: {recon_ci.sum() / max(raw.sum(), 1e-6):.4f})")
    print(f"  NMAE(PSF*ci_rl, raw)       : {nmae(recon_ci, raw):.4f}")
    print(f"  NMAE(PSF*ci_rl, raw_bgsub) : {nmae(recon_ci, raw_bgsub):.4f}")
    print(f"  mean(PSF*ci_rl - raw)      : {diff_ci.mean():.4f}")
    print(f"  % pixels PSF*ci_rl > raw   : {100.0 * np.mean(diff_ci > 0):.1f}%")

    # ------------------------------------------------------------------
    # Step 3: Per-iteration reconvolution error
    # ------------------------------------------------------------------
    print("\n=== Step 3: Per-iteration reconvolution error (niter=51) ===")
    iters: list[int] = []
    errs_raw: list[float] = []
    errs_bgsub: list[float] = []
    bg_logged: list[float] = []

    def iter_cb(payload: dict) -> None:
        it = int(payload["iteration"])
        estimated = np.asarray(payload["estimated"], dtype=np.float32)
        bg = float(payload.get("background", 0.0))
        err_raw = nmae(estimated, raw)
        err_bgsub = nmae(estimated, raw_bgsub)
        iters.append(it)
        errs_raw.append(err_raw)
        errs_bgsub.append(err_bgsub)
        bg_logged.append(bg)
        if it <= 3 or it % 10 == 0 or payload.get("is_final"):
            print(
                f"  iter {it:3d}: NMAE(est,raw)={err_raw:.4f}  "
                f"NMAE(est,raw_bgsub)={err_bgsub:.4f}  "
                f"callback_bg={bg:.3f}"
            )

    ci_rl_out2 = ci_rl_deconvolve(
        raw,
        psf,
        niter=51,
        convergence="fixed",
        check_every=1,
        iteration_callback=iter_cb,
    )
    print(f"  iterations_used: {ci_rl_out2.get('iterations_used', '?')}")
    ci_rl2 = np.asarray(ci_rl_out2["result"], dtype=np.float32)
    recon_ci2 = reconvolve_same(ci_rl2, psf)
    print(f"  Final-iter NMAE(PSF*ci_rl, raw): {nmae(recon_ci2, raw):.4f}")
    print(f"  Callback final NMAE(estimated, raw): {errs_raw[-1] if errs_raw else 'N/A'}")
    print(f"  Ratio final-callback vs PSF*ci_rl reconvolution: "
          f"{errs_raw[-1] / max(nmae(recon_ci2, raw), 1e-8):.3f}x"
          if errs_raw else "")

    # ------------------------------------------------------------------
    # Step 4: DL refinement with diagnostics
    # ------------------------------------------------------------------
    print("\n=== Step 4: DL refinement ===")
    dl_out = deconvolve_ci_rl_dl(
        raw,
        psf,
        model_path=MODEL_PATH,
        device="auto",
        rl_kwargs={"niter": 51, "convergence": "fixed", "check_every": 5},
        return_diagnostics=True,
    )

    dl_result = np.asarray(dl_out["result"], dtype=np.float32)
    recon_dl = reconvolve_same(dl_result, psf)
    diff_dl = recon_dl - raw
    diag = dl_out.get("diagnostics", {})

    print(f"  RL iterations      : {diag.get('rl_iterations', '?')}")
    print(f"  intensity_sum_before_dl: {diag.get('intensity_sum_before_dl', 'N/A'):.1f}")
    print(f"  intensity_sum_after_dl : {diag.get('intensity_sum_after_dl', 'N/A'):.1f}")
    if "reconvolution_error_before" in diag:
        print(f"  reconvolution_error_before (from apply_dl): {diag['reconvolution_error_before']:.4f}")
    if "reconvolution_error_after" in diag:
        print(f"  reconvolution_error_after  (from apply_dl): {diag['reconvolution_error_after']:.4f}")

    print(f"  dl_result range    : [{dl_result.min():.2f}, {dl_result.max():.2f}]")
    print(f"  dl_result sum      : {dl_result.sum():.1f}  (ratio vs ci_rl: {dl_result.sum() / max(ci_rl.sum(), 1e-6):.4f})")
    print(f"  PSF*DL sum         : {recon_dl.sum():.1f}  (ratio vs raw: {recon_dl.sum() / max(raw.sum(), 1e-6):.4f})")
    print(f"  NMAE(PSF*DL, raw)       : {nmae(recon_dl, raw):.4f}")
    print(f"  NMAE(PSF*DL, raw_bgsub) : {nmae(recon_dl, raw_bgsub):.4f}")
    print(f"  mean(PSF*DL - raw)      : {diff_dl.mean():.4f}")
    print(f"  % pixels PSF*DL > raw   : {100.0 * np.mean(diff_dl > 0):.1f}%")

    # ------------------------------------------------------------------
    # Step 5: Residual of the DL model
    # ------------------------------------------------------------------
    residual_dl = np.asarray(dl_out.get("residual", np.zeros_like(dl_result)), dtype=np.float32)
    recon_residual = reconvolve_same(residual_dl, psf)
    print(f"\n=== Step 5: DL residual analysis ===")
    print(f"  residual mean  : {residual_dl.mean():.4f}")
    print(f"  residual p1    : {np.percentile(residual_dl, 1):.4f}")
    print(f"  residual p99   : {np.percentile(residual_dl, 99):.4f}")
    print(f"  PSF*residual mean: {recon_residual.mean():.4f}")
    print(f"  (PSF*residual contributes this offset to PSF*DL vs PSF*ci_rl)")

    # ------------------------------------------------------------------
    # Step 6: Plots
    # ------------------------------------------------------------------
    print("\n=== Step 6: Saving plots ===")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Plot 1: Convergence curve
    ax = axes[0, 0]
    ax.plot(iters, errs_raw, "b-", linewidth=2, label="NMAE(est_k, raw)")
    ax.plot(iters, errs_bgsub, "r--", linewidth=2, label="NMAE(est_k, raw_bgsub)")
    dl_e_raw = nmae(recon_dl, raw)
    dl_e_bgsub = nmae(recon_dl, raw_bgsub)
    ax.axhline(dl_e_raw, color="blue", linestyle=":", linewidth=1.5, alpha=0.8, label=f"DL vs raw={dl_e_raw:.4f}")
    ax.axhline(dl_e_bgsub, color="red", linestyle=":", linewidth=1.5, alpha=0.8, label=f"DL vs bgsub={dl_e_bgsub:.4f}")
    ax.set_xlabel("RL Iteration")
    ax.set_ylabel("NMAE")
    ax.set_title("Reconvolution error vs iteration\n(should decrease toward 0)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 2: raw MIP
    ax = axes[0, 1]
    ax.imshow(pclip(mip(raw)), cmap="gray", interpolation="nearest")
    ax.set_title("raw (MIP)")
    ax.axis("off")

    # Plot 3: PSF*ci_rl MIP
    ax = axes[0, 2]
    ax.imshow(pclip(mip(recon_ci2)), cmap="gray", interpolation="nearest")
    ax.set_title("PSF*ci_rl (MIP)")
    ax.axis("off")

    # Plot 4: PSF*ci_rl - raw
    diff_mip_ci = mip(diff_ci)
    vmax_ci = float(np.percentile(np.abs(diff_mip_ci), 99)) or 1.0
    ax = axes[1, 0]
    im = ax.imshow(diff_mip_ci, cmap="RdBu_r", vmin=-vmax_ci, vmax=vmax_ci, interpolation="nearest")
    ax.set_title("PSF*ci_rl − raw (red=over, blue=under)")
    plt.colorbar(im, ax=ax)
    ax.axis("off")

    # Plot 5: PSF*DL - raw
    diff_mip_dl = mip(diff_dl)
    vmax_dl = float(np.percentile(np.abs(diff_mip_dl), 99)) or 1.0
    ax = axes[1, 1]
    im = ax.imshow(diff_mip_dl, cmap="RdBu_r", vmin=-vmax_dl, vmax=vmax_dl, interpolation="nearest")
    ax.set_title("PSF*DL − raw (red=over, blue=under)")
    plt.colorbar(im, ax=ax)
    ax.axis("off")

    # Plot 6: DL residual MIP
    ax = axes[1, 2]
    res_mip = mip(residual_dl)
    vmax_res = float(np.percentile(np.abs(res_mip), 99)) or 1.0
    im = ax.imshow(res_mip, cmap="RdBu_r", vmin=-vmax_res, vmax=vmax_res, interpolation="nearest")
    ax.set_title("DL residual (MIP)")
    plt.colorbar(im, ax=ax)
    ax.axis("off")

    fig.suptitle(
        f"raw: {raw.shape}  bg_p1={bg_p1:.1f}  "
        f"ci_rl_NMAE={nmae(recon_ci2, raw):.4f}  DL_NMAE={dl_e_raw:.4f}",
        fontsize=10,
    )
    fig.tight_layout()

    plot_path = OUTPUT_DIR / "residual_convergence.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plot_path}")

    # Save CSV
    csv_path = OUTPUT_DIR / "iteration_errors.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iteration", "nmae_vs_raw", "nmae_vs_raw_bgsub", "callback_bg"])
        for it, e1, e2, bg in zip(iters, errs_raw, errs_bgsub, bg_logged):
            writer.writerow([it, f"{e1:.6f}", f"{e2:.6f}", f"{bg:.4f}"])
    print(f"  Saved: {csv_path}")

    # ------------------------------------------------------------------
    # Diagnosis summary
    # ------------------------------------------------------------------
    print("\n=== DIAGNOSIS SUMMARY ===")
    p99_raw = float(np.percentile(raw, 99))
    threshold = 0.02 * p99_raw  # 2% of p99 as "significant"

    conv_trend = errs_raw[-1] - errs_raw[0] if len(errs_raw) >= 2 else 0.0
    print(f"  RL convergence trend: {errs_raw[0]:.4f} → {errs_raw[-1]:.4f} "
          f"({'decreasing OK' if conv_trend < 0 else 'NOT decreasing!'})")

    mean_diff_ci = diff_ci.mean()
    mean_diff_dl = diff_dl.mean()

    print(f"\n  ci_rl: mean(PSF*ci_rl - raw) = {mean_diff_ci:.4f}")
    if abs(mean_diff_ci) < threshold:
        print("    -> ci_rl reconvolution closely matches raw (expected)")
    elif mean_diff_ci > 0:
        print(f"    -> PSF*ci_rl EXCEEDS raw by {mean_diff_ci:.4f} "
              f"(cyan in inset; scale issue or offset mismatch?)")
    else:
        print(f"    -> PSF*ci_rl BELOW raw by {abs(mean_diff_ci):.4f} "
              f"(red in inset; background not subtracted?)")

    print(f"\n  DL:   mean(PSF*DL - raw)    = {mean_diff_dl:.4f}")
    if abs(mean_diff_dl) < threshold:
        print("    -> DL reconvolution closely matches raw (ideal)")
    elif mean_diff_dl > 0:
        print(
            f"    -> PSF*DL EXCEEDS raw by {mean_diff_dl:.4f} "
            f"(cyan in inset; DL adds signal that wasn't in the raw image)"
        )
    else:
        print(
            f"    -> PSF*DL BELOW raw by {abs(mean_diff_dl):.4f} "
            f"(red in inset; DL reduces total signal)"
        )

    print(
        f"\n  Conclusion: DL changes reconvolution by {mean_diff_dl - mean_diff_ci:+.4f} "
        f"relative to ci_rl baseline"
    )
    print(
        "  If DL is a 'super-resolution' style model that sharpens beyond RL, "
        "PSF*DL will NOT match raw — that is expected physical behavior.\n"
        "  If you want the inset to fade to black, use the 'Raw / estimated' RATIO mode "
        "which auto-normalises per frame, OR do not use a difference inset for the DL frame."
    )


if __name__ == "__main__":
    main()
