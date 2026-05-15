"""Evaluate a trained ci_rl_dl model on synthetic and local real microscopy data.

This is an intentionally heavier quality script, not a normal pytest test.  It
loads an existing trained model, sweeps residual strengths, writes CSV metrics,
and saves montages for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.deconvolve import deconvolve_image
from core.deconvolve_ci_dl import (
    apply_dl_refinement_25d,
    conditioning_vector,
    load_residual_unet_checkpoint,
    reconvolve_same,
    resolve_torch_device,
)


log = logging.getLogger("evaluate_ci_rl_dl_model")


DEFAULT_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
DEFAULT_REAL_FILES = [
    "Dendrites_Crop.ome.tiff",
    "DividingCellcrop.ome.tiff",
    "DNAcrop.ome.tiff",
    "U2OS.ome.tiff",
]


def parse_strengths(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def robust_mip(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim == 3:
        return arr.max(axis=0)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 2-D or 3-D volume, got {arr.shape}")


def display_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo = float(np.percentile(finite, 0.5))
    hi = float(np.percentile(finite, 99.7))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalized_mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    scale = max(float(np.percentile(np.abs(b), 99.5)), 1e-6)
    return float(np.mean(np.abs(a - b)) / scale)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) ** 2)))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = rmse(a, b)
    if err <= 0:
        return float("inf")
    finite = np.asarray(b, dtype=np.float32)
    data_range = max(float(np.percentile(finite, 99.9) - np.percentile(finite, 0.1)), 1e-6)
    return float(20.0 * math.log10(data_range / err))


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    if av.size < 2 or float(np.std(av)) == 0.0 or float(np.std(bv)) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def gradient_energy(volume: np.ndarray) -> float:
    arr = np.asarray(volume, dtype=np.float32)
    grads = np.gradient(arr)
    return float(sum(np.mean(g * g) for g in grads))


def compare_metrics(result: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_nmae": normalized_mae(result, reference),
        f"{prefix}_rmse": rmse(result, reference),
        f"{prefix}_psnr": psnr(result, reference),
        f"{prefix}_corr": corrcoef(result, reference),
    }


def intensity_metrics(result: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    result_f = np.asarray(result, dtype=np.float32)
    base_f = np.asarray(baseline, dtype=np.float32)
    eps = 1e-6
    negative_correction = np.maximum(base_f - result_f, 0)
    return {
        "sum_to_ci_rl": float(result_f.sum() / max(float(base_f.sum()), eps)),
        "mean_to_ci_rl": float(result_f.mean() / max(float(base_f.mean()), eps)),
        "p99_to_ci_rl": float(np.percentile(result_f, 99) / max(float(np.percentile(base_f, 99)), eps)),
        "mean_abs_change_to_ci_rl": normalized_mae(result_f, base_f),
        "negative_correction_fraction": float(negative_correction.sum() / max(float(base_f.sum()), eps)),
        "gradient_energy": gradient_energy(result_f),
    }


def load_checkpoint_settings(model_path: Path) -> dict[str, Any]:
    sidecar = model_path.with_suffix(".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    return {}


def infer_model_settings(metadata: dict[str, Any]) -> tuple[int, int, bool, int]:
    recommended = metadata.get("recommended_inference", {})
    dl_kwargs = recommended.get("dl_kwargs", {})
    z_radius = int(dl_kwargs.get("z_radius", recommended.get("dl_z_context", 2)))
    batch_size = int(dl_kwargs.get("batch_size", recommended.get("dl_batch_size", 8)))
    mixed_precision = bool(dl_kwargs.get("mixed_precision", recommended.get("dl_mixed_precision", True)))
    xy_padding = int(dl_kwargs.get("xy_padding", 0))
    return z_radius, max(batch_size, 1), mixed_precision, max(xy_padding, 0)


def synthetic_sample_dirs(data_dir: Path, num_synthetic: int) -> list[Path]:
    selected: list[Path] = []
    for split in ("test", "val", "train"):
        split_dir = data_dir / split
        if split_dir.exists():
            dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and (p / "gt.tif").exists())
            selected.extend(dirs)
            if len(selected) >= num_synthetic:
                return selected[:num_synthetic]
    return selected


def load_synthetic_sample(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    signal_scale = float(metadata.get("noise", {}).get("signal_scale", 1.0))
    gt_density = tifffile.imread(sample_dir / "gt.tif").astype(np.float32)
    gt = gt_density * signal_scale
    raw = tifffile.imread(sample_dir / "raw.tif").astype(np.float32)
    ci_rl = tifffile.imread(sample_dir / "ci_rl.tif").astype(np.float32)
    psf = tifffile.imread(sample_dir / "psf.tif").astype(np.float32)
    metadata["signal_scale"] = signal_scale
    metadata["gt_density_sum"] = float(gt_density.sum())
    return gt, raw, ci_rl, psf, metadata


def evaluate_synthetic(
    *,
    sample_dirs: list[Path],
    model: Any,
    strengths: list[float],
    z_radius: int,
    batch_size: int,
    mixed_precision: bool,
    xy_padding: int,
    device: str,
    output_dir: Path,
    checkpoint_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    montage_rows: list[tuple[str, list[tuple[str, np.ndarray]]]] = []

    for sample_dir in sample_dirs:
        log.info("Synthetic sample %s", sample_dir.name)
        gt, raw, ci_rl, psf, sample_meta = load_synthetic_sample(sample_dir)
        cond = conditioning_vector(
            psf=psf,
            metadata=sample_meta,
            rl_iterations=int(sample_meta.get("rl_requested_iterations", 0) or 0),
            microscope_type=sample_meta.get("psf_params", {}).get("microscope_type"),
        ) if checkpoint_metadata.get("conditioning_channels") else None
        raw_bgsub = np.clip(raw - np.percentile(raw, 1), 0, None)
        recon_ci = reconvolve_same(ci_rl, psf, device=device)
        baseline_gt = compare_metrics(ci_rl, gt, "gt")
        baseline_recon = normalized_mae(recon_ci, raw_bgsub)

        montage_images: list[tuple[str, np.ndarray]] = [
            ("raw", robust_mip(raw)),
            ("ci_rl", robust_mip(ci_rl)),
            ("gt", robust_mip(gt)),
        ]
        for strength in strengths:
            if strength == 0.0:
                result = ci_rl
            else:
                refined = apply_dl_refinement_25d(
                    raw,
                    ci_rl,
                    model,
                    psf=psf,
                    device=device,
                    z_radius=z_radius,
                    batch_size=batch_size,
                    mixed_precision=mixed_precision,
                    residual_strength=strength,
                    xy_padding=xy_padding,
                    conditioning_values=cond,
                )
                result = refined["result"]
            recon = reconvolve_same(result, psf, device=device)
            row = {
                "sample": sample_dir.name,
                "strength": strength,
                "synthetic_morphology": sample_meta.get("synthetic_morphology", "unknown"),
                "microscope_type": sample_meta.get("psf_params", {}).get("microscope_type", "unknown"),
                "rl_iterations": sample_meta.get("rl_requested_iterations", ""),
                "psf_mismatch_mode": sample_meta.get("psf_mismatch_mode", "none"),
                "gt_signal_scale": float(sample_meta["signal_scale"]),
                "raw_sum_to_gt": float(raw.sum() / max(float(gt.sum()), 1e-6)),
                "ci_rl_sum_to_gt": float(ci_rl.sum() / max(float(gt.sum()), 1e-6)),
                "ci_rl_sum_to_unscaled_gt_density": float(ci_rl.sum() / max(float(sample_meta["gt_density_sum"]), 1e-6)),
                "result_sum_to_gt": float(np.asarray(result, dtype=np.float32).sum() / max(float(gt.sum()), 1e-6)),
                **compare_metrics(result, gt, "gt"),
                **intensity_metrics(result, ci_rl),
                "reconv_nmae_to_raw_bgsub": normalized_mae(recon, raw_bgsub),
                "gt_nmae_delta_vs_ci_rl": compare_metrics(result, gt, "gt")["gt_nmae"] - baseline_gt["gt_nmae"],
                "reconv_delta_vs_ci_rl": normalized_mae(recon, raw_bgsub) - baseline_recon,
            }
            rows.append(row)
            if strength in (0.25, 0.5, 0.75, 1.0):
                montage_images.append((f"dl {strength:g}", robust_mip(result)))
        montage_rows.append((sample_dir.name, montage_images))

    save_strength_montage(output_dir / "synthetic_strength_sweep.png", montage_rows, max_rows=10)
    return rows


def save_strength_montage(path: Path, rows: list[tuple[str, list[tuple[str, np.ndarray]]]], *, max_rows: int = 10) -> None:
    if not rows:
        return
    rows = rows[:max_rows]
    n_cols = max(len(images) for _, images in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.6 * n_cols, 2.4 * len(rows)), squeeze=False)
    for row_idx, (row_name, images) in enumerate(rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            if col_idx >= len(images):
                continue
            title, image = images[col_idx]
            ax.imshow(display_image(image), cmap="gray", interpolation="nearest")
            if row_idx == 0:
                ax.set_title(title, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_real_file(
    *,
    path: Path,
    model: Any,
    strengths: list[float],
    z_radius: int,
    batch_size: int,
    mixed_precision: bool,
    xy_padding: int,
    device: str,
    niter: int,
    output_dir: Path,
    checkpoint_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    log.info("Real file %s: running ci_rl baseline once", path)
    baseline = deconvolve_image(
        path,
        method="ci_rl",
        niter=niter,
        background="auto",
        offset="auto",
        start="observed",
        convergence="fixed",
        two_d_mode="legacy_2d",
        device=device,
    )
    rows: list[dict[str, Any]] = []
    all_montages: list[tuple[str, list[tuple[str, np.ndarray]]]] = []

    metadata = baseline.get("metadata", {})
    for ch_idx, (raw, ci_rl, psf) in enumerate(
        zip(baseline["source_channels"], baseline["channels"], baseline["psfs"], strict=False)
    ):
        raw = np.asarray(raw, dtype=np.float32)
        ci_rl = np.asarray(ci_rl, dtype=np.float32)
        psf = np.asarray(psf, dtype=np.float32)
        if raw.ndim == 2 and psf.ndim == 3:
            psf_for_recon = psf[psf.shape[0] // 2]
        else:
            psf_for_recon = psf
        raw_bgsub = np.clip(raw - np.percentile(raw, 1), 0, None)
        recon_ci = reconvolve_same(ci_rl, psf_for_recon, device=device)
        recon_ci_nmae = normalized_mae(recon_ci, raw_bgsub)
        montage_images: list[tuple[str, np.ndarray]] = [
            ("raw", robust_mip(raw)),
            ("ci_rl", robust_mip(ci_rl)),
        ]
        cond = conditioning_vector(
            psf=psf_for_recon,
            metadata={
                "psf_params": {
                    "pixel_size_xy_nm": metadata.get("pixel_size_x", 0),
                    "pixel_size_z_nm": metadata.get("pixel_size_z", 0),
                    "microscope_type": metadata.get("microscope_type", "unknown"),
                }
            },
            rl_iterations=niter,
            microscope_type=metadata.get("microscope_type", "unknown"),
        ) if checkpoint_metadata.get("conditioning_channels") else None

        for strength in strengths:
            if strength == 0.0:
                result = ci_rl
            else:
                refined = apply_dl_refinement_25d(
                    raw,
                    ci_rl,
                    model,
                    psf=psf_for_recon,
                    device=device,
                    z_radius=z_radius,
                    batch_size=batch_size,
                    mixed_precision=mixed_precision,
                    residual_strength=strength,
                    xy_padding=xy_padding,
                    conditioning_values=cond,
                )
                result = np.asarray(refined["result"], dtype=np.float32)
            recon = reconvolve_same(result, psf_for_recon, device=device)
            recon_nmae = normalized_mae(recon, raw_bgsub)
            row = {
                "file": path.name,
                "channel": ch_idx,
                "strength": strength,
                "microscope_type": metadata.get("microscope_type", "unknown"),
                "rl_iterations": niter,
                "psf_mismatch_mode": "real_unknown",
                **intensity_metrics(result, ci_rl),
                "reconv_nmae_to_raw_bgsub": recon_nmae,
                "reconv_delta_vs_ci_rl": recon_nmae - recon_ci_nmae,
                "corr_to_ci_rl": corrcoef(result, ci_rl),
                "shape": "x".join(str(v) for v in raw.shape),
            }
            rows.append(row)
            if strength in (0.25, 0.5, 0.75, 1.0):
                montage_images.append((f"dl {strength:g}", robust_mip(result)))
        all_montages.append((f"{path.stem} ch{ch_idx}", montage_images))

    save_strength_montage(output_dir / f"{path.stem}_strength_sweep.png", all_montages, max_rows=16)
    return rows


def summarize(rows: Iterable[dict[str, Any]], key_metric: str, group_key: str = "strength") -> list[dict[str, float]]:
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key_metric)
        if value is not None and np.isfinite(float(value)):
            group_value: Any
            try:
                group_value = float(row[group_key])
            except Exception:
                group_value = str(row[group_key])
            grouped[group_value].append(float(value))
    summary = []
    for strength, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float32)
        summary.append(
            {
                group_key: strength,
                f"{key_metric}_median": float(np.median(arr)),
                f"{key_metric}_mean": float(np.mean(arr)),
                "n": int(arr.size),
            }
        )
    return summary


def build_recommendations(synthetic_rows: list[dict[str, Any]], real_rows: list[dict[str, Any]]) -> dict[str, Any]:
    synthetic_by_strength = summarize(synthetic_rows, "gt_nmae")
    synthetic_ci_to_gt = summarize(
        [row for row in synthetic_rows if float(row.get("strength", -1)) == 0.0],
        "ci_rl_sum_to_gt",
    )
    synthetic_ci_to_density = summarize(
        [row for row in synthetic_rows if float(row.get("strength", -1)) == 0.0],
        "ci_rl_sum_to_unscaled_gt_density",
    )
    synthetic_raw_to_gt = summarize(
        [row for row in synthetic_rows if float(row.get("strength", -1)) == 0.0],
        "raw_sum_to_gt",
    )
    real_intensity = summarize(real_rows, "sum_to_ci_rl")
    real_change = summarize(real_rows, "mean_abs_change_to_ci_rl")
    synthetic_grouped = {
        "by_morphology": summarize([row for row in synthetic_rows if float(row.get("strength", -1)) == 1.0], "gt_nmae", "synthetic_morphology"),
        "by_microscope_type": summarize([row for row in synthetic_rows if float(row.get("strength", -1)) == 1.0], "gt_nmae", "microscope_type"),
        "by_rl_iterations": summarize([row for row in synthetic_rows if float(row.get("strength", -1)) == 1.0], "gt_nmae", "rl_iterations"),
        "by_psf_mismatch": summarize([row for row in synthetic_rows if float(row.get("strength", -1)) == 1.0], "gt_nmae", "psf_mismatch_mode"),
    }
    real_grouped = {
        "by_microscope_type": summarize([row for row in real_rows if float(row.get("strength", -1)) == 1.0], "sum_to_ci_rl", "microscope_type"),
        "by_rl_iterations": summarize([row for row in real_rows if float(row.get("strength", -1)) == 1.0], "sum_to_ci_rl", "rl_iterations"),
    }
    best_synth = min(synthetic_by_strength, key=lambda row: row["gt_nmae_median"]) if synthetic_by_strength else None
    safe_real = [
        row for row in real_intensity
        if 0.9 <= row["sum_to_ci_rl_median"] <= 1.05 and row["strength"] > 0.0
    ]
    safe_strength = min(safe_real, key=lambda row: abs(row["sum_to_ci_rl_median"] - 1.0)) if safe_real else None
    return {
        "synthetic_gt_nmae_by_strength": synthetic_by_strength,
        "synthetic_ci_rl_sum_to_gt_at_baseline": synthetic_ci_to_gt,
        "synthetic_ci_rl_sum_to_unscaled_gt_density_at_baseline": synthetic_ci_to_density,
        "synthetic_raw_sum_to_gt_at_baseline": synthetic_raw_to_gt,
        "real_sum_to_ci_rl_by_strength": real_intensity,
        "real_change_to_ci_rl_by_strength": real_change,
        "synthetic_grouped_at_strength_1": synthetic_grouped,
        "real_grouped_at_strength_1": real_grouped,
        "best_synthetic_strength_by_gt_nmae": best_synth,
        "conservative_real_strength_by_intensity_retention": safe_strength,
        "scale_mismatch_warning": _scale_mismatch_warning(synthetic_ci_to_gt, synthetic_ci_to_density),
        "interpretation": (
            "Use synthetic GT metrics to detect whether the model can improve known data. "
            "Use real-data intensity retention and reconvolution metrics as guard rails only; "
            "without real ground truth they cannot prove biological correctness."
        ),
    }


def _scale_mismatch_warning(
    synthetic_ci_to_gt: list[dict[str, float]],
    synthetic_ci_to_density: list[dict[str, float]],
) -> str | None:
    if not synthetic_ci_to_gt:
        return None
    ratio = float(synthetic_ci_to_gt[0].get("ci_rl_sum_to_gt_median", 1.0))
    density_ratio = (
        float(synthetic_ci_to_density[0].get("ci_rl_sum_to_unscaled_gt_density_median", 1.0))
        if synthetic_ci_to_density else 1.0
    )
    if ratio > 10.0:
        return (
            f"Synthetic ci_rl intensity is about {ratio:.1f}x the GT intensity at baseline. "
            "A residual model trained to map ci_rl directly to low-scale GT can learn an almost pure "
            "negative correction and darken real images. Match GT/raw photon scaling or train in a "
            "normalized domain with explicit rescaling back to the ci_rl intensity domain."
        )
    if ratio < 0.1:
        return (
            f"Synthetic ci_rl intensity is about {ratio:.3f}x the GT intensity at baseline. "
            "Check synthetic intensity calibration before trusting residual-strength sweeps."
        )
    if density_ratio > 10.0:
        return (
            f"Saved synthetic GT density files are unscaled ({density_ratio:.1f}x below ci_rl by summed intensity), "
            "but this evaluator and the current training dataset rescale GT by metadata.noise.signal_scale. "
            "This is expected for generator version 4 and newer."
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("training_runs/gui_large_quality_v2_mixed/checkpoints/best_model.pt"))
    parser.add_argument("--training-run", type=Path, default=Path("training_runs/gui_large_quality_v2_mixed"))
    parser.add_argument("--real-dir", type=Path, default=Path("localdata"))
    parser.add_argument("--real-files", nargs="*", default=DEFAULT_REAL_FILES)
    parser.add_argument("--output-dir", type=Path, default=Path("training_runs/gui_large_quality_v2_mixed/evaluation"))
    parser.add_argument("--num-synthetic", type=int, default=10)
    parser.add_argument("--strengths", type=parse_strengths, default=DEFAULT_STRENGTHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--niter", type=int, default=None)
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    device_obj = resolve_torch_device(args.device)
    device = str(device_obj)
    metadata = load_checkpoint_settings(args.model_path)
    z_radius, batch_size, mixed_precision, xy_padding = infer_model_settings(metadata)
    niter = args.niter
    if niter is None:
        niter = int(metadata.get("recommended_inference", {}).get("iterations", 50))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load_residual_unet_checkpoint(args.model_path, device=device)
    log.info(
        "Evaluating %s on %s, strengths=%s, z_radius=%d, batch_size=%d, xy_padding=%d, niter=%d",
        args.model_path,
        device,
        args.strengths,
        z_radius,
        batch_size,
        xy_padding,
        niter,
    )

    synthetic_rows: list[dict[str, Any]] = []
    if not args.skip_synthetic:
        sample_dirs = synthetic_sample_dirs(args.training_run / "data", args.num_synthetic)
        if not sample_dirs:
            log.warning("No synthetic samples found under %s", args.training_run / "data")
        else:
            synthetic_rows = evaluate_synthetic(
                sample_dirs=sample_dirs,
                model=model,
                strengths=args.strengths,
                z_radius=z_radius,
                batch_size=batch_size,
                mixed_precision=mixed_precision,
                xy_padding=xy_padding,
                device=device,
                output_dir=args.output_dir,
                checkpoint_metadata=metadata,
            )
            write_csv(args.output_dir / "synthetic_metrics.csv", synthetic_rows)

    real_rows: list[dict[str, Any]] = []
    if not args.skip_real:
        for name in args.real_files:
            real_path = args.real_dir / name
            if not real_path.exists():
                log.warning("Skipping missing real file %s", real_path)
                continue
            real_rows.extend(
                evaluate_real_file(
                    path=real_path,
                    model=model,
                    strengths=args.strengths,
                    z_radius=z_radius,
                    batch_size=batch_size,
                    mixed_precision=mixed_precision,
                    xy_padding=xy_padding,
                    device=device,
                    niter=niter,
                    output_dir=args.output_dir,
                    checkpoint_metadata=metadata,
                )
            )
        write_csv(args.output_dir / "real_metrics.csv", real_rows)

    summary = build_recommendations(synthetic_rows, real_rows)
    summary.update(
        {
            "model_path": str(args.model_path),
            "training_run": str(args.training_run),
            "strengths": args.strengths,
            "device": device,
            "z_radius": z_radius,
            "batch_size": batch_size,
            "mixed_precision": mixed_precision,
            "xy_padding": xy_padding,
            "niter": niter,
            "num_synthetic_rows": len(synthetic_rows),
            "num_real_rows": len(real_rows),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote evaluation to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
