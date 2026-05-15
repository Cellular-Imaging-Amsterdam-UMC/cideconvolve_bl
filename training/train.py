"""Train an experimental 2.5D residual U-Net for ci_rl_dl."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import math
import multiprocessing
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# Ensure repo root is on sys.path so core.* is importable when this
# script is run directly (python training/train.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import tifffile
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - Pillow fallback covers minimal envs.
    plt = None

from core.deconvolve_ci import ci_generate_psf, ci_rl_deconvolve
from core.deconvolve_ci_dl import (
    CONDITIONING_CHANNELS,
    GatedResidualUNet25D,
    ResidualUNet25D,
    conditioning_vector,
    input_channel_count,
    make_25d_input,
    reconvolve_same,
    resolve_torch_device,
)

log = logging.getLogger(__name__)

PSF_GENERATION_KEYS = {
    "na",
    "wavelength_nm",
    "pixel_size_xy_nm",
    "pixel_size_z_nm",
    "n_xy",
    "n_z",
    "ri_immersion",
    "ri_sample",
    "ri_coverslip",
    "ri_coverslip_design",
    "ri_immersion_design",
    "microscope_type",
    "excitation_nm",
    "pinhole_airy_units",
    "integrate_pixels",
    "n_pupil",
    "device",
}


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_finish_time(seconds_from_now: float | None) -> str:
    if seconds_from_now is None or not math.isfinite(seconds_from_now) or seconds_from_now < 0:
        return "unknown"
    return datetime.fromtimestamp(time.time() + seconds_from_now).strftime("%Y-%m-%d_%H:%M:%S")


@dataclass
class TrainConfig:
    num_volumes: int = 24
    volume_shape: tuple[int, int, int] = (16, 96, 96)
    patch_size: int = 64
    z_context: int = 2
    batch_size: int = 4
    epochs: int = 2
    steps: Optional[int] = None
    learning_rate: float = 1e-3
    output_dir: Path = Path("training_runs")
    device: str = "auto"
    mixed_precision: bool = True
    seed: int = 42
    base_channels: int = 16
    residual_scale: float = 1.0
    rl_iterations: int = 8
    rl_iteration_pool: tuple[int, ...] = ()
    rl_iteration_weights: tuple[float, ...] = ()
    quick_test: bool = False
    reconvolution_weight: float = 0.0
    gradient_weight: float = 0.05
    negative_residual_weight: float = 0.05
    max_negative_residual_fraction: float = 0.25
    intensity_retention_weight: float = 0.0
    intensity_retention_min: float = 0.90
    intensity_retention_max: float = 1.15
    global_intensity_weight: float = 0.0
    global_intensity_min: float = 0.98
    global_intensity_max: float = 1.02
    background_offset_weight: float = 0.0
    training_xy_padding: int = 0
    train_samples_per_epoch: int = 256
    val_samples: int = 64
    synthetic_complexity: str = "standard"
    synthetic_artifact_level: str = "standard"
    super_sample_xy: int = 1
    super_sample_z: int = 1
    synthetic_morphology: str = "mixed"
    microscope_type: str = "widefield"
    psf_mismatch: str = "none"
    psf_mismatch_moderate_fraction: float = 0.0
    model_type: str = "GatedResidualUNet25D"
    use_conditioning: bool = True
    residual_bound_fraction: float = 0.35
    residual_bound_scale: float = 0.05
    num_workers: int = 0
    data_loader_workers: int = 0
    volume_cache_size: int = 8


def parse_volume_shape(value: str) -> tuple[int, int, int]:
    parts = [int(p.strip()) for p in value.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("volume shape must be Z,Y,X, for example 16,128,128")
    return tuple(parts)  # type: ignore[return-value]


def parse_int_pool(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [int(p.strip()) for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        parts = [int(v) for v in value]
    return tuple(max(int(v), 1) for v in parts)


def parse_float_pool(value: str | Sequence[float] | None) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [float(p.strip()) for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        parts = [float(v) for v in value]
    return tuple(max(float(v), 0.0) for v in parts)


def downsample_xy_mean(volume: np.ndarray, factor: int) -> np.ndarray:
    factor = max(int(factor), 1)
    if factor == 1:
        return np.asarray(volume)
    z, y, x = volume.shape
    y_trim = (y // factor) * factor
    x_trim = (x // factor) * factor
    trimmed = np.asarray(volume)[:, :y_trim, :x_trim]
    return trimmed.reshape(z, y_trim // factor, factor, x_trim // factor, factor).mean(axis=(2, 4))


def upsample_xy_torch(volume: np.ndarray, factor: int) -> np.ndarray:
    factor = max(int(factor), 1)
    arr = np.asarray(volume, dtype=np.float32)
    if factor == 1:
        return arr
    with torch.no_grad():
        tensor = torch.from_numpy(arr)[None, None]
        up = torch.nn.functional.interpolate(
            tensor,
            scale_factor=(1, factor, factor),
            mode="trilinear",
            align_corners=False,
        )
    return up[0, 0].cpu().numpy().astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_gaussian_blob(volume: np.ndarray, center: tuple[float, float, float], sigma: tuple[float, float, float], amplitude: float) -> None:
    z, y, x = np.indices(volume.shape, dtype=np.float32)
    cz, cy, cx = center
    sz, sy, sx = sigma
    blob = np.exp(-0.5 * (((z - cz) / sz) ** 2 + ((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2))
    volume += amplitude * blob.astype(np.float32)


def _add_polyline_fiber(
    volume: np.ndarray,
    points: list[tuple[float, float, float]],
    radius: tuple[float, float, float],
    amplitude: float,
    steps_per_segment: int = 18,
) -> None:
    for p0, p1 in zip(points[:-1], points[1:]):
        for t in np.linspace(0.0, 1.0, steps_per_segment):
            center = tuple((1.0 - t) * p0[i] + t * p1[i] for i in range(3))
            _add_gaussian_blob(volume, center, radius, amplitude)


def _add_soft_tube(volume: np.ndarray, points: np.ndarray, radius_px: float, amplitude: float) -> None:
    """Rasterize a soft curved 3-D tube around a polyline."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 2:
        return
    rz = max(float(radius_px) * 0.65, 0.7)
    ryx = max(float(radius_px), 0.9)
    margin_z = int(np.ceil(3.0 * rz))
    margin_yx = int(np.ceil(3.0 * ryx))
    shape_hi = np.asarray(volume.shape, dtype=np.float32) - 1

    for p0, p1 in zip(pts[:-1], pts[1:]):
        p0 = np.clip(p0, 0, shape_hi)
        p1 = np.clip(p1, 0, shape_hi)
        segment = p1 - p0
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq <= 1e-6:
            continue

        z0 = max(int(np.floor(min(p0[0], p1[0]) - margin_z)), 0)
        z1 = min(int(np.ceil(max(p0[0], p1[0]) + margin_z)) + 1, volume.shape[0])
        y0 = max(int(np.floor(min(p0[1], p1[1]) - margin_yx)), 0)
        y1 = min(int(np.ceil(max(p0[1], p1[1]) + margin_yx)) + 1, volume.shape[1])
        x0 = max(int(np.floor(min(p0[2], p1[2]) - margin_yx)), 0)
        x1 = min(int(np.ceil(max(p0[2], p1[2]) + margin_yx)) + 1, volume.shape[2])
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue

        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1].astype(np.float32)
        coords = np.stack((zz, yy, xx), axis=-1)
        rel = coords - p0
        t = np.clip(np.sum(rel * segment, axis=-1) / seg_len_sq, 0.0, 1.0)
        nearest = p0 + t[..., None] * segment
        dz = (coords[..., 0] - nearest[..., 0]) / rz
        dy = (coords[..., 1] - nearest[..., 1]) / ryx
        dx = (coords[..., 2] - nearest[..., 2]) / ryx
        tube = np.exp(-0.5 * (dz * dz + dy * dy + dx * dx)).astype(np.float32)
        volume[z0:z1, y0:y1, x0:x1] += float(amplitude) * tube


def _add_tapered_soft_tube(
    volume: np.ndarray,
    points: np.ndarray,
    radius_start: float,
    radius_end: float,
    amplitude: float,
) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 2:
        return
    n_segments = max(pts.shape[0] - 1, 1)
    for idx in range(n_segments):
        t = idx / max(n_segments - 1, 1)
        radius = (1.0 - t) * float(radius_start) + t * float(radius_end)
        local_amp = float(amplitude) * (0.75 + 0.25 * math.sin(math.pi * t))
        _add_soft_tube(volume, pts[idx:idx + 2], radius, local_amp)


def _orthonormal_vectors(axis: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis) + 1e-6
    other = rng.normal(size=3).astype(np.float32)
    other -= axis * float(np.dot(axis, other))
    if np.linalg.norm(other) < 1e-4:
        other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        other -= axis * float(np.dot(axis, other))
    other /= np.linalg.norm(other) + 1e-6
    third = np.cross(axis, other).astype(np.float32)
    third /= np.linalg.norm(third) + 1e-6
    return other, third


def _random_curved_path(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    *,
    n_points: int = 48,
    length_fraction: tuple[float, float] = (0.18, 0.70),
    z_weight: float = 0.22,
) -> np.ndarray:
    z, y, x = shape
    center = np.array(
        [
            rng.uniform(z * 0.18, z * 0.82),
            rng.uniform(y * 0.18, y * 0.82),
            rng.uniform(x * 0.18, x * 0.82),
        ],
        dtype=np.float32,
    )
    axis = rng.normal(size=3).astype(np.float32)
    axis[0] *= z_weight
    axis /= np.linalg.norm(axis) + 1e-6
    side_a, side_b = _orthonormal_vectors(axis, rng)
    span = rng.uniform(*length_fraction) * min(y, x)
    curve_a = rng.uniform(0.04, 0.18) * min(y, x)
    curve_b = rng.uniform(0.02, 0.12) * min(y, x)
    z_curve = rng.uniform(0.04, 0.22) * z
    phase_a = rng.uniform(0, 2 * math.pi)
    phase_b = rng.uniform(0, 2 * math.pi)
    t_values = np.linspace(-0.5, 0.5, n_points, dtype=np.float32)
    points = []
    for t in t_values:
        drift = axis * (span * t)
        bend = side_a * (curve_a * math.sin(2 * math.pi * (t + 0.5) + phase_a))
        bend += side_b * (curve_b * math.sin(4 * math.pi * (t + 0.5) + phase_b))
        bend[0] += z_curve * math.sin(2 * math.pi * (t + 0.5) + phase_b)
        points.append(center + drift + bend)
    return np.clip(np.stack(points, axis=0), [0, 0, 0], np.asarray(shape, dtype=np.float32) - 1)


def _curved_path_between(
    start: np.ndarray,
    end: np.ndarray,
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    *,
    n_points: int = 40,
    wobble_scale: float = 0.05,
) -> np.ndarray:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    axis = end - start
    side_a, side_b = _orthonormal_vectors(axis, rng)
    span = max(float(np.linalg.norm(axis)), 1.0)
    phase = rng.uniform(0, 2 * math.pi)
    points = []
    for t in np.linspace(0.0, 1.0, n_points, dtype=np.float32):
        base = (1.0 - t) * start + t * end
        envelope = math.sin(math.pi * float(t))
        wobble = side_a * (span * wobble_scale * envelope * math.sin(2 * math.pi * t + phase))
        wobble += side_b * (span * wobble_scale * 0.65 * envelope * math.sin(3 * math.pi * t + 0.5 * phase))
        points.append(base + wobble)
    return np.clip(np.stack(points, axis=0), [0, 0, 0], np.asarray(shape, dtype=np.float32) - 1)


def _add_nucleus_like_object(volume: np.ndarray, rng: np.random.Generator) -> None:
    z, y, x = volume.shape
    center = (rng.uniform(z * 0.2, z * 0.8), rng.uniform(y * 0.25, y * 0.75), rng.uniform(x * 0.25, x * 0.75))
    sigma = (rng.uniform(z * 0.08, z * 0.22), rng.uniform(y * 0.10, y * 0.23), rng.uniform(x * 0.10, x * 0.23))
    _add_gaussian_blob(volume, center, sigma, rng.uniform(0.10, 0.35))
    _add_gaussian_blob(volume, center, tuple(max(s * 0.55, 1.0) for s in sigma), -rng.uniform(0.02, 0.10))
    for _ in range(rng.integers(12, 35)):
        offset = rng.normal(0, 0.45, size=3)
        spot_center = tuple(np.clip(center[i] + offset[i] * sigma[i], 0, volume.shape[i] - 1) for i in range(3))
        spot_sigma = (rng.uniform(0.5, 1.4), rng.uniform(1.0, 3.2), rng.uniform(1.0, 3.2))
        _add_gaussian_blob(volume, spot_center, spot_sigma, rng.uniform(0.06, 0.22))


def _add_irregular_chromatin_nucleus(volume: np.ndarray, rng: np.random.Generator, *, mitotic: bool = False) -> None:
    z, y, x = volume.shape
    zz, yy, xx = np.indices(volume.shape, dtype=np.float32)
    center = np.array(
        [rng.uniform(z * 0.25, z * 0.75), rng.uniform(y * 0.25, y * 0.75), rng.uniform(x * 0.25, x * 0.75)],
        dtype=np.float32,
    )
    radii = np.array(
        [rng.uniform(z * 0.10, z * 0.28), rng.uniform(y * 0.12, y * 0.28), rng.uniform(x * 0.12, x * 0.28)],
        dtype=np.float32,
    )
    coords = np.stack((zz - center[0], yy - center[1], xx - center[2]), axis=0)
    r = np.sqrt(np.sum((coords / radii[:, None, None, None]) ** 2, axis=0))
    warp = np.zeros_like(volume, dtype=np.float32)
    for _ in range(rng.integers(8, 16)):
        blob_center = (
            rng.uniform(0, z - 1),
            rng.uniform(0, y - 1),
            rng.uniform(0, x - 1),
        )
        blob_sigma = (
            rng.uniform(z * 0.08, z * 0.22),
            rng.uniform(y * 0.12, y * 0.30),
            rng.uniform(x * 0.12, x * 0.30),
        )
        _add_gaussian_blob(warp, blob_center, blob_sigma, rng.uniform(-0.18, 0.18))
    mask_arg = np.clip((r + warp - 1.0) / rng.uniform(0.035, 0.075), -80.0, 80.0)
    mask = 1.0 / (1.0 + np.exp(mask_arg))
    texture = np.zeros_like(volume, dtype=np.float32)
    for _ in range(rng.integers(25, 55)):
        offset = rng.normal(0, 0.45, size=3)
        clump_center = tuple(np.clip(center[i] + offset[i] * radii[i], 0, volume.shape[i] - 1) for i in range(3))
        clump_sigma = (rng.uniform(0.45, 1.5), rng.uniform(1.2, 5.0), rng.uniform(1.2, 5.0))
        _add_gaussian_blob(texture, clump_center, clump_sigma, rng.uniform(0.12, 0.75))
    for _ in range(rng.integers(1, 4)):
        hole_center = tuple(np.clip(center[i] + rng.normal(0, 0.25) * radii[i], 0, volume.shape[i] - 1) for i in range(3))
        hole_sigma = (rng.uniform(0.8, 2.5), rng.uniform(3.0, 8.0), rng.uniform(3.0, 8.0))
        _add_gaussian_blob(texture, hole_center, hole_sigma, -rng.uniform(0.15, 0.45))
    volume += (mask * (rng.uniform(0.06, 0.22) + texture)).astype(np.float32)

    if mitotic:
        axis = rng.normal(size=3).astype(np.float32)
        axis[0] *= 0.3
        axis /= np.linalg.norm(axis) + 1e-6
        for sign in (-1.0, 1.0):
            cluster_center = center + sign * axis * rng.uniform(0.15, 0.35) * min(y, x)
            for _ in range(rng.integers(14, 30)):
                chrom_center = cluster_center + rng.normal(0, [z * 0.035, y * 0.045, x * 0.045])
                _add_gaussian_blob(
                    volume,
                    tuple(np.clip(chrom_center, [0, 0, 0], np.asarray(volume.shape) - 1)),
                    (rng.uniform(0.5, 1.4), rng.uniform(1.4, 4.5), rng.uniform(1.4, 4.5)),
                    rng.uniform(0.25, 0.95),
                )


def _add_mitotic_spindle(volume: np.ndarray, rng: np.random.Generator) -> None:
    z, y, x = volume.shape
    center = np.array([rng.uniform(z * 0.25, z * 0.75), rng.uniform(y * 0.35, y * 0.65), rng.uniform(x * 0.35, x * 0.65)], dtype=np.float32)
    angle = rng.uniform(0, 2 * math.pi)
    axis = np.array([rng.normal(0, 0.08), math.sin(angle), math.cos(angle)], dtype=np.float32)
    axis /= np.linalg.norm(axis) + 1e-6
    half_len = rng.uniform(min(y, x) * 0.12, min(y, x) * 0.28)
    pole_a = center - axis * half_len
    pole_b = center + axis * half_len
    for pole in (pole_a, pole_b):
        _add_gaussian_blob(volume, tuple(np.clip(pole, [0, 0, 0], np.array(volume.shape) - 1)), (0.7, 1.5, 1.5), rng.uniform(0.3, 0.9))
    for _ in range(rng.integers(18, 40)):
        mid = center + rng.normal(0, [z * 0.04, y * 0.06, x * 0.06])
        first = _curved_path_between(
            pole_a + rng.normal(0, [0.4, 1.8, 1.8]),
            mid,
            volume.shape,
            rng,
            n_points=20,
            wobble_scale=0.035,
        )
        second = _curved_path_between(
            mid,
            pole_b + rng.normal(0, [0.4, 1.8, 1.8]),
            volume.shape,
            rng,
            n_points=20,
            wobble_scale=0.035,
        )
        _add_soft_tube(volume, np.concatenate([first, second[1:]], axis=0), rng.uniform(0.7, 1.2), rng.uniform(0.025, 0.09))
    for _ in range(rng.integers(8, 18)):
        chrom_center = center + rng.normal(0, [z * 0.035, y * 0.045, x * 0.045])
        _add_gaussian_blob(
            volume,
            tuple(np.clip(chrom_center, [0, 0, 0], np.array(volume.shape) - 1)),
            (rng.uniform(0.7, 1.5), rng.uniform(2.0, 5.0), rng.uniform(2.0, 5.0)),
            rng.uniform(0.12, 0.45),
        )


def _add_cell_membrane_and_cytoplasm(volume: np.ndarray, rng: np.random.Generator) -> None:
    z, y, x = volume.shape
    zz, yy, xx = np.indices(volume.shape, dtype=np.float32)
    center = (rng.uniform(z * 0.25, z * 0.75), rng.uniform(y * 0.35, y * 0.65), rng.uniform(x * 0.35, x * 0.65))
    rz = rng.uniform(z * 0.18, z * 0.42)
    ry = rng.uniform(y * 0.22, y * 0.48)
    rx = rng.uniform(x * 0.22, x * 0.48)
    r = ((zz - center[0]) / rz) ** 2 + ((yy - center[1]) / ry) ** 2 + ((xx - center[2]) / rx) ** 2
    cytoplasm = np.exp(-0.5 * r / rng.uniform(0.45, 0.9))
    shell = np.exp(-0.5 * ((np.sqrt(np.maximum(r, 1e-8)) - 1.0) / rng.uniform(0.035, 0.09)) ** 2)
    volume += (cytoplasm * rng.uniform(0.025, 0.12) + shell * rng.uniform(0.04, 0.25)).astype(np.float32)


def _add_partial_membrane_surface(volume: np.ndarray, rng: np.random.Generator) -> None:
    z, y, x = volume.shape
    zz, yy, xx = np.indices(volume.shape, dtype=np.float32)
    center = np.array([rng.uniform(z * 0.20, z * 0.80), rng.uniform(y * 0.15, y * 0.85), rng.uniform(x * 0.15, x * 0.85)], dtype=np.float32)
    radii = np.array([rng.uniform(z * 0.18, z * 0.50), rng.uniform(y * 0.24, y * 0.60), rng.uniform(x * 0.24, x * 0.60)], dtype=np.float32)
    r = np.sqrt(((zz - center[0]) / radii[0]) ** 2 + ((yy - center[1]) / radii[1]) ** 2 + ((xx - center[2]) / radii[2]) ** 2)
    angle = np.arctan2(yy - center[1], xx - center[2])
    wrinkle = 0.035 * np.sin(angle * rng.uniform(2.0, 5.0) + rng.uniform(0, 2 * math.pi))
    wrinkle += 0.025 * np.sin((zz - center[0]) / max(radii[0], 1.0) * math.pi * rng.uniform(1.0, 3.5))
    shell = np.exp(-0.5 * ((r + wrinkle - 1.0) / rng.uniform(0.035, 0.085)) ** 2)
    gate_axis = rng.uniform(0, 2 * math.pi)
    gate = 1.0 / (1.0 + np.exp(-rng.uniform(3.0, 7.0) * np.cos(angle - gate_axis)))
    cytoplasm = np.exp(-0.5 * (r / rng.uniform(0.50, 0.90)) ** 2)
    volume += (shell * gate * rng.uniform(0.10, 0.38) + cytoplasm * gate * rng.uniform(0.015, 0.08)).astype(np.float32)


def _add_actin_fiber_field(volume: np.ndarray, rng: np.random.Generator) -> None:
    z, y, x = volume.shape
    for _ in range(rng.integers(12, 36)):
        points = _random_curved_path(
            volume.shape,
            rng,
            n_points=int(rng.integers(32, 72)),
            length_fraction=(0.22, 0.85),
            z_weight=0.18,
        )
        _add_soft_tube(volume, points, rng.uniform(0.75, 1.6), rng.uniform(0.025, 0.12))


def _add_branching_dendrite_tree(volume: np.ndarray, rng: np.random.Generator) -> None:
    main = _random_curved_path(
        volume.shape,
        rng,
        n_points=int(rng.integers(56, 110)),
        length_fraction=(0.50, 1.05),
        z_weight=0.18,
    )
    _add_tapered_soft_tube(volume, main, rng.uniform(1.8, 3.2), rng.uniform(0.8, 1.4), rng.uniform(0.08, 0.24))
    branch_count = int(rng.integers(4, 12))
    for _ in range(branch_count):
        idx = int(rng.integers(max(main.shape[0] // 8, 1), max(main.shape[0] - 3, 2)))
        start = main[idx]
        direction = rng.normal(size=3).astype(np.float32)
        direction[0] *= 0.25
        direction /= np.linalg.norm(direction) + 1e-6
        length = rng.uniform(0.12, 0.42) * min(volume.shape[1], volume.shape[2])
        end = start + direction * length
        branch = _curved_path_between(start, end, volume.shape, rng, n_points=int(rng.integers(24, 58)), wobble_scale=0.09)
        _add_tapered_soft_tube(volume, branch, rng.uniform(0.9, 1.8), rng.uniform(0.35, 0.9), rng.uniform(0.04, 0.16))
        for spine_idx in rng.choice(branch.shape[0], size=int(rng.integers(2, 7)), replace=False):
            spine_center = branch[int(spine_idx)] + rng.normal(0, [0.4, 1.2, 1.2])
            _add_gaussian_blob(
                volume,
                tuple(np.clip(spine_center, [0, 0, 0], np.asarray(volume.shape) - 1)),
                (rng.uniform(0.35, 0.9), rng.uniform(0.7, 1.8), rng.uniform(0.7, 1.8)),
                rng.uniform(0.05, 0.22),
            )


def _add_puncta_field(volume: np.ndarray, rng: np.random.Generator, *, dense: bool = False) -> None:
    z, y, x = volume.shape
    count = int(rng.integers(120 if dense else 35, 360 if dense else 110))
    for _ in range(count):
        center = (rng.uniform(0, z - 1), rng.uniform(0, y - 1), rng.uniform(0, x - 1))
        sigma = (rng.uniform(0.35, 1.0), rng.uniform(0.45, 1.6), rng.uniform(0.45, 1.6))
        _add_gaussian_blob(volume, center, sigma, rng.uniform(0.08, 0.75 if dense else 1.3))


SYNTHETIC_MORPHOLOGIES = ("mixed", "generic", "dna", "mitotic", "membrane", "actin", "dendrite", "puncta")


def choose_synthetic_morphology(requested: str, rng: np.random.Generator) -> str:
    requested = str(requested).strip().lower()
    if requested and requested != "mixed":
        return requested
    morphologies = ["generic", "dna", "mitotic", "membrane", "actin", "dendrite", "puncta"]
    weights = np.asarray([0.16, 0.18, 0.10, 0.14, 0.16, 0.14, 0.12], dtype=np.float32)
    weights /= weights.sum()
    return str(rng.choice(morphologies, p=weights))


def _add_generic_structures(vol: np.ndarray, rng: np.random.Generator, *, full: bool) -> None:
    z, y, x = vol.shape
    for _ in range(rng.integers(35 if full else 20, 95 if full else 55)):
        center = (rng.uniform(0, z - 1), rng.uniform(0, y - 1), rng.uniform(0, x - 1))
        sigma = (rng.uniform(0.45, 1.4), rng.uniform(0.7, 2.2), rng.uniform(0.7, 2.2))
        _add_gaussian_blob(vol, center, sigma, rng.uniform(0.25, 1.4))

    for _ in range(rng.integers(4 if full else 2, 10 if full else 6)):
        center = (rng.uniform(0, z - 1), rng.uniform(0, y - 1), rng.uniform(0, x - 1))
        sigma = (rng.uniform(1.0, 3.0), rng.uniform(4.0, 12.0), rng.uniform(4.0, 12.0))
        _add_gaussian_blob(vol, center, sigma, rng.uniform(0.05, 0.25))


def _add_vesicles(vol: np.ndarray, rng: np.random.Generator, *, full: bool) -> None:
    z, y, x = vol.shape
    for _ in range(rng.integers(4 if full else 2, 12 if full else 7)):
        center = (rng.uniform(0, z - 1), rng.uniform(0, y - 1), rng.uniform(0, x - 1))
        outer = (rng.uniform(0.8, 2.0), rng.uniform(3.0, 9.0), rng.uniform(3.0, 9.0))
        inner = tuple(max(s * rng.uniform(0.45, 0.7), 0.5) for s in outer)
        amp = rng.uniform(0.12, 0.5)
        _add_gaussian_blob(vol, center, outer, amp)
        _add_gaussian_blob(vol, center, inner, -amp * rng.uniform(0.65, 0.9))


def _add_soft_sheets(vol: np.ndarray, rng: np.random.Generator, *, full: bool) -> None:
    z, y, x = vol.shape
    for _ in range(rng.integers(2 if full else 1, 7 if full else 4)):
        z0 = rng.uniform(0, z - 1)
        y0 = rng.uniform(0, y - 1)
        x0 = rng.uniform(0, x - 1)
        zz, yy, xx = np.indices(vol.shape, dtype=np.float32)
        normal = rng.normal(size=3).astype(np.float32)
        normal /= np.linalg.norm(normal) + 1e-6
        dist = normal[0] * (zz - z0) + normal[1] * (yy - y0) + normal[2] * (xx - x0)
        sheet = np.exp(-0.5 * (dist / rng.uniform(0.7, 1.8)) ** 2)
        window = np.exp(-0.5 * (((yy - y0) / rng.uniform(y * 0.25, y * 0.7)) ** 2 + ((xx - x0) / rng.uniform(x * 0.25, x * 0.7)) ** 2))
        vol += (sheet * window * rng.uniform(0.03, 0.18)).astype(np.float32)


def generate_synthetic_gt(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    complexity: str = "standard",
    morphology: str = "mixed",
) -> np.ndarray:
    """Create a small microscopy-like synthetic ground-truth volume."""
    z, y, x = shape
    full = str(complexity).lower() == "full"
    morphology = choose_synthetic_morphology(morphology, rng)
    vol = np.zeros(shape, dtype=np.float32)
    vol += rng.uniform(0.0, 0.02)

    if morphology in ("generic", "puncta"):
        _add_generic_structures(vol, rng, full=full)
    if morphology == "puncta":
        _add_puncta_field(vol, rng, dense=True)

    if morphology in ("generic", "actin"):
        for _ in range(rng.integers(6 if full else 3, 16 if full else 8)):
            points = _random_curved_path(
                shape,
                rng,
                n_points=int(rng.integers(28, 72)),
                length_fraction=(0.15, 0.62),
                z_weight=0.25,
            )
            _add_soft_tube(vol, points, rng.uniform(0.9, 2.0), rng.uniform(0.04, 0.22))

    if morphology == "actin":
        _add_actin_fiber_field(vol, rng)
        _add_puncta_field(vol, rng, dense=False)

    if morphology == "dendrite":
        for _ in range(rng.integers(1, 4 if full else 3)):
            _add_branching_dendrite_tree(vol, rng)
        _add_puncta_field(vol, rng, dense=False)

    if morphology in ("dna", "mitotic"):
        for _ in range(rng.integers(1, 4 if full else 3)):
            _add_irregular_chromatin_nucleus(vol, rng, mitotic=(morphology == "mitotic"))
        if morphology == "mitotic" or rng.random() < 0.25:
            _add_mitotic_spindle(vol, rng)

    if morphology == "membrane":
        for _ in range(rng.integers(1, 5 if full else 3)):
            if rng.random() < 0.45:
                _add_cell_membrane_and_cytoplasm(vol, rng)
            _add_partial_membrane_surface(vol, rng)
        _add_soft_sheets(vol, rng, full=full)

    if morphology == "generic":
        _add_vesicles(vol, rng, full=full)
        _add_soft_sheets(vol, rng, full=full)

    if full and morphology == "generic":
        for _ in range(rng.integers(1, 3)):
            _add_nucleus_like_object(vol, rng)
        if rng.random() < 0.30:
            _add_mitotic_spindle(vol, rng)
        if rng.random() < 0.55:
            _add_cell_membrane_and_cytoplasm(vol, rng)
        if rng.random() < 0.65:
            _add_actin_fiber_field(vol, rng)

    if np.percentile(vol, 99.5) <= 0.025:
        _add_generic_structures(vol, rng, full=False)
    for _ in range(rng.integers(0 if full else 0, 3 if full else 2)):
        points = _random_curved_path(
            shape,
            rng,
            n_points=int(rng.integers(24, 56)),
            length_fraction=(0.12, 0.45),
            z_weight=0.25,
        )
        _add_soft_tube(vol, points, rng.uniform(0.6, 1.3), rng.uniform(0.01, 0.05))

    vol -= vol.min()
    p = np.percentile(vol, 99.8)
    if p > 0:
        vol /= p
    return np.clip(vol, 0, 1.5).astype(np.float32)


def make_small_psf(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    device: str = "cpu",
    microscope_type: str = "widefield",
    super_sample_xy: int = 1,
    super_sample_z: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    micro = str(microscope_type).strip().lower()
    if micro == "mixed":
        micro = "confocal" if rng.random() < 0.5 else "widefield"
    ss_xy = max(int(super_sample_xy), 1)
    ss_z = max(int(super_sample_z), 1)
    psf_z = min(max(5, shape[0] | 1), 15)
    psf_xy = 15
    emission = float(rng.uniform(500, 620))
    camera_pixel_xy = float(rng.uniform(75, 130))
    camera_pixel_z = float(rng.uniform(220, 380))
    params = {
        "na": float(rng.uniform(1.1, 1.4)),
        "wavelength_nm": emission,
        "pixel_size_xy_nm": camera_pixel_xy / ss_xy,
        "pixel_size_z_nm": camera_pixel_z / ss_z,
        "camera_pixel_size_xy_nm": camera_pixel_xy,
        "camera_pixel_size_z_nm": camera_pixel_z,
        "super_sample_xy": ss_xy,
        "super_sample_z": ss_z,
        "n_xy": psf_xy,
        "n_z": psf_z,
        "ri_sample": float(rng.uniform(1.33, 1.47)),
        "microscope_type": micro,
        "excitation_nm": float(max(emission - rng.uniform(25, 80), 350)) if micro == "confocal" else None,
        "pinhole_airy_units": float(rng.uniform(0.8, 1.4)) if micro == "confocal" else 1.0,
        "integrate_pixels": False,
        "n_pupil": 33,
        "device": device,
    }
    psf_kwargs = {key: value for key, value in params.items() if key in PSF_GENERATION_KEYS}
    psf = ci_generate_psf(**psf_kwargs).astype(np.float32)
    return psf, params


def generate_psf_from_params(params: dict[str, Any], device: str = "cpu") -> np.ndarray:
    psf_params = {key: value for key, value in dict(params).items() if key in PSF_GENERATION_KEYS}
    psf_params["device"] = device
    return ci_generate_psf(**psf_params).astype(np.float32)


def perturb_psf_params(params: dict[str, Any], rng: np.random.Generator, mode: str) -> dict[str, Any]:
    mode_l = str(mode).strip().lower()
    if mode_l == "none":
        return dict(params)
    scale = 0.045 if mode_l == "mild" else 0.12
    out = dict(params)
    for key in ("na", "wavelength_nm", "pixel_size_xy_nm", "pixel_size_z_nm", "ri_sample"):
        if key in out and out[key] is not None:
            out[key] = float(out[key]) * float(np.clip(1.0 + rng.normal(0.0, scale), 0.65, 1.35))
    if out.get("pinhole_airy_units") is not None:
        out["pinhole_airy_units"] = float(out["pinhole_airy_units"]) * float(np.clip(1.0 + rng.normal(0.0, scale), 0.65, 1.50))
    return out


def add_psf_aberration(psf: np.ndarray, rng: np.random.Generator, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Add lightweight coma/astigmatism-like PSF error for synthetic robustness."""
    mode_l = str(mode).strip().lower()
    if mode_l == "none":
        return psf.astype(np.float32), {"enabled": False, "mode": "none"}
    strength = 0.04 if mode_l == "mild" else 0.10
    out = np.asarray(psf, dtype=np.float32)
    shift_y = int(rng.choice([-2, -1, 1, 2]))
    shift_x = int(rng.choice([-2, -1, 1, 2]))
    if mode_l == "mild":
        shift_y = int(np.sign(shift_y))
        shift_x = int(np.sign(shift_x))
    coma = np.roll(out, shift=(0, shift_y, shift_x), axis=(0, 1, 2))
    astig = 0.5 * (
        np.roll(out, shift=(0, shift_y, 0), axis=(0, 1, 2))
        + np.roll(out, shift=(0, 0, shift_x), axis=(0, 1, 2))
    )
    out = (1.0 - strength) * out + strength * (0.65 * coma + 0.35 * astig)
    axial_shift = int(rng.choice([-1, 1])) if out.shape[0] > 3 and rng.random() < (0.25 if mode_l == "mild" else 0.55) else 0
    if axial_shift:
        out = 0.92 * out + 0.08 * np.roll(out, shift=axial_shift, axis=0)
    out = np.clip(out, 0, None)
    out /= max(float(out.sum()), 1e-12)
    return out.astype(np.float32), {
        "enabled": True,
        "mode": mode_l,
        "strength": float(strength),
        "coma_shift_yx": [int(shift_y), int(shift_x)],
        "axial_shift": int(axial_shift),
    }


def choose_psf_mismatch_mode(config_mode: str, rng: np.random.Generator, moderate_fraction: float) -> str:
    mode = str(config_mode).strip().lower()
    if mode == "mild" and rng.random() < max(float(moderate_fraction), 0.0):
        return "moderate"
    if mode not in {"none", "mild", "moderate"}:
        return "none"
    return mode


def synthetic_illumination_field(shape: tuple[int, int, int], rng: np.random.Generator, strength: float) -> np.ndarray:
    z, y, x = shape
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    yy = (yy / max(y - 1, 1)) - 0.5
    xx = (xx / max(x - 1, 1)) - 0.5
    zz = (zz / max(z - 1, 1)) - 0.5
    angle = rng.uniform(0, 2 * math.pi)
    ramp = math.cos(angle) * xx + math.sin(angle) * yy + rng.normal(0, 0.25) * zz
    field = 1.0 + strength * ramp
    for _ in range(2):
        cy = rng.uniform(-0.45, 0.45)
        cx = rng.uniform(-0.45, 0.45)
        sigma = rng.uniform(0.18, 0.45)
        amp = rng.uniform(-0.8, 1.0) * strength
        field += amp * np.exp(-0.5 * (((yy - cy) / sigma) ** 2 + ((xx - cx) / sigma) ** 2))
    return np.clip(field, 0.35, 1.9).astype(np.float32)


def add_microscopy_noise(
    blurred: np.ndarray,
    rng: np.random.Generator,
    *,
    microscope_type: str,
    artifact_level: str = "standard",
) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    micro = str(microscope_type).strip().lower()
    artifacts = str(artifact_level).strip().lower()
    strong = artifacts in {"strong", "restoration", "aggressive"}
    if micro == "confocal":
        signal_scale = float(rng.uniform(20, 300) if strong else rng.uniform(50, 450))
        optical_background = float(rng.uniform(1.0, 20.0) if strong else rng.uniform(0.5, 8.0))
        camera_offset = float(rng.uniform(0.0, 8.0))
        read_sigma = float(rng.uniform(0.8, 5.0) if strong else rng.uniform(0.4, 2.0))
        hot_probability = 0.30 if strong else 0.15
        hot_fraction = (5e-6, 8e-5) if strong else (2e-6, 2e-5)
        hot_amplitude = (50, 1_000) if strong else (30, 300)
        clip_probability = 0.12 if strong else 0.05
    else:
        signal_scale = float(rng.uniform(500, 20_000) if strong else rng.uniform(15_000, 60_000))
        optical_background = float(rng.uniform(100, 2_500) if strong else rng.uniform(250, 1_500))
        camera_offset = float(rng.uniform(85, 125)) if strong else 100.0
        read_sigma = float(rng.uniform(1.5, 9.0) if strong else rng.uniform(1.0, 4.0))
        hot_probability = 0.55 if strong else 0.35
        hot_fraction = (1e-5, 2.5e-4) if strong else (5e-6, 8e-5)
        hot_amplitude = (300, 12_000) if strong else (500, 6_000)
        clip_probability = 0.35 if strong else 0.20

    illumination_strength = float(rng.uniform(0.08, 0.32) if strong else rng.uniform(0.0, 0.08))
    illumination = synthetic_illumination_field(blurred.shape, rng, illumination_strength)
    haze_strength = float(rng.uniform(0.01, 0.10) if strong else rng.uniform(0.0, 0.025))
    haze = synthetic_illumination_field(blurred.shape, rng, haze_strength) - 1.0
    photons = np.clip(blurred, 0, None) * signal_scale * illumination + optical_background * (1.0 + haze)
    raw = rng.poisson(photons).astype(np.float32)
    raw += camera_offset
    raw += rng.normal(0, read_sigma, size=raw.shape).astype(np.float32)
    row_banding = False
    if strong and rng.random() < 0.60:
        row_banding = True
        rows = rng.normal(0, read_sigma * rng.uniform(0.25, 1.1), size=(1, raw.shape[1], 1)).astype(np.float32)
        cols = rng.normal(0, read_sigma * rng.uniform(0.08, 0.45), size=(1, 1, raw.shape[2])).astype(np.float32)
        phase = rng.uniform(0, 2 * math.pi)
        yy = np.arange(raw.shape[1], dtype=np.float32)[None, :, None]
        periodic = np.sin(yy / rng.uniform(6.0, 24.0) + phase) * read_sigma * rng.uniform(0.2, 1.0)
        raw += rows + cols + periodic
    outlier_noise = False
    if strong and rng.random() < 0.45:
        outlier_noise = True
        mask = rng.random(raw.shape) < rng.uniform(1e-5, 2e-4)
        raw[mask] += rng.normal(0, read_sigma * rng.uniform(8.0, 30.0), size=int(mask.sum())).astype(np.float32)
    hot_pixels = False
    if rng.random() < hot_probability:
        hot_pixels = True
        n_hot = max(int(raw.size * rng.uniform(*hot_fraction)), 1)
        flat = raw.reshape(-1)
        idx = rng.choice(flat.size, size=n_hot, replace=False)
        flat[idx] += rng.uniform(*hot_amplitude, size=n_hot).astype(np.float32)
    clipped = False
    if rng.random() < clip_probability:
        clipped = True
        clip = np.percentile(raw, rng.uniform(99.85, 99.98))
        raw = np.clip(raw, 0, clip)
    raw_u16 = np.clip(np.rint(raw), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return raw_u16, {
        "microscope_type": micro,
        "signal_scale": signal_scale,
        "optical_background": optical_background,
        "camera_offset": camera_offset,
        "read_sigma": read_sigma,
        "hot_pixels": hot_pixels,
        "row_banding": row_banding,
        "outlier_noise": outlier_noise,
        "clipped": clipped,
        "synthetic_artifact_level": artifacts,
        "illumination_strength": illumination_strength,
        "haze_strength": haze_strength,
        "raw_dtype": "uint16",
        "raw_min": float(raw_u16.min()),
        "raw_max": float(raw_u16.max()),
        "raw_p01": float(np.percentile(raw_u16, 1)),
        "raw_p50": float(np.percentile(raw_u16, 50)),
        "raw_p995": float(np.percentile(raw_u16, 99.5)),
    }


def split_name(index: int, total: int) -> str:
    train_cut = max(int(total * 0.8), 1)
    val_cut = max(int(total * 0.9), train_cut + 1) if total >= 3 else train_cut
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def save_volume_sample(sample_dir: Path, gt: np.ndarray, raw: np.ndarray, ci_rl: np.ndarray, psf: np.ndarray, metadata: dict[str, Any]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(sample_dir / "gt.tif", gt.astype(np.float32))
    tifffile.imwrite(sample_dir / "raw.tif", raw.astype(np.uint16) if raw.dtype == np.uint16 else raw.astype(np.float32))
    tifffile.imwrite(sample_dir / "ci_rl.tif", ci_rl.astype(np.float32))
    tifffile.imwrite(sample_dir / "psf.tif", psf.astype(np.float32))
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_raw_sample(sample_dir: Path, gt: np.ndarray, raw: np.ndarray, psf: np.ndarray, metadata: dict[str, Any]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(sample_dir / "gt.tif", gt.astype(np.float32))
    tifffile.imwrite(sample_dir / "raw.tif", raw.astype(np.uint16) if raw.dtype == np.uint16 else raw.astype(np.float32))
    tifffile.imwrite(sample_dir / "psf.tif", psf.astype(np.float32))
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def update_sample_metadata(sample_dir: Path, updates: dict[str, Any]) -> None:
    metadata_path = sample_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(updates)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_sample_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def raw_data_signature(payload: dict[str, Any], split: str) -> dict[str, Any]:
    idx = int(payload["index"])
    seed = int(payload["seed"])
    return {
        "index": idx,
        "split": split,
        "num_volumes": int(payload["num_volumes"]),
        "volume_shape": [int(v) for v in payload["volume_shape"]],
        "synthetic_complexity": str(payload["synthetic_complexity"]),
        "synthetic_generator_version": 8,
        "synthetic_morphology": str(payload.get("synthetic_morphology", "mixed")).strip().lower(),
        "synthetic_artifact_level": str(payload.get("synthetic_artifact_level", "standard")).strip().lower(),
        "super_sample_xy": int(payload.get("super_sample_xy", 1)),
        "super_sample_z": int(payload.get("super_sample_z", 1)),
        "microscope_type": str(payload["microscope_type"]).strip().lower(),
        "psf_mismatch": str(payload.get("psf_mismatch", "none")).strip().lower(),
        "psf_mismatch_moderate_fraction": float(payload.get("psf_mismatch_moderate_fraction", 0.0)),
        "raw_generation_seed": seed + idx * 104729,
    }


def raw_sample_matches(sample_dir: Path, signature: dict[str, Any]) -> bool:
    required = ("gt.tif", "raw.tif", "psf.tif", "forward_psf.tif", "deconv_psf.tif")
    if not all((sample_dir / name).exists() for name in required):
        return False
    metadata = read_sample_metadata(sample_dir)
    return metadata.get("raw_data_signature") == signature


def ci_rl_path(sample_dir: Path, rl_iterations: int) -> Path:
    return sample_dir / f"ci_rl_iter_{int(rl_iterations):03d}.tif"


def ci_rl_matches(sample_dir: Path, rl_iterations: int) -> bool:
    if not ci_rl_path(sample_dir, rl_iterations).exists() and not (sample_dir / "ci_rl.tif").exists():
        return False
    metadata = read_sample_metadata(sample_dir)
    return int(metadata.get("rl_requested_iterations", -1)) == int(rl_iterations)


def _to_uint8_plane(arr: np.ndarray) -> np.ndarray:
    plane = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(plane, [1, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((plane - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def save_montage(path: Path, panels: dict[str, np.ndarray]) -> None:
    tiles = []
    labels = []
    for label, vol in panels.items():
        plane = vol[vol.shape[0] // 2] if vol.ndim == 3 else vol
        tiles.append(Image.fromarray(_to_uint8_plane(plane)).convert("L"))
        labels.append(label)
    w, h = tiles[0].size
    label_h = 18
    canvas = Image.new("L", (w * len(tiles), h + label_h), color=0)
    draw = ImageDraw.Draw(canvas)
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * w, label_h))
        draw.text((i * w + 4, 3), labels[i], fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def generation_worker_count(num_workers: int) -> int:
    if num_workers < 0:
        return 1
    if num_workers == 0:
        return max((os.cpu_count() or 2) - 1, 1)
    return max(int(num_workers), 1)


def data_loader_worker_count(num_workers: int, device: torch.device) -> int:
    if num_workers < 0:
        return 0
    if num_workers > 0:
        return int(num_workers)
    if device.type != "cuda":
        return 0
    return min(max((os.cpu_count() or 4) // 2, 1), 8)


def effective_rl_iteration_pool(config: TrainConfig) -> tuple[int, ...]:
    pool = parse_int_pool(config.rl_iteration_pool)
    if pool:
        return pool
    return (max(int(config.rl_iterations), 1),)


def select_rl_iterations(sample_dir: Path, config: TrainConfig) -> int:
    pool = effective_rl_iteration_pool(config)
    metadata = read_sample_metadata(sample_dir)
    idx = int(metadata.get("index", 0))
    rng = np.random.default_rng(int(config.seed) + idx * 8191 + 17)
    weights = parse_float_pool(config.rl_iteration_weights)
    probabilities = None
    if weights and len(weights) == len(pool) and sum(weights) > 0:
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
    return int(rng.choice(np.asarray(pool, dtype=np.int32), p=probabilities))


def _generate_raw_sample_worker(payload: dict[str, Any]) -> str:
    # Avoid severe oversubscription when many PSF/convolution workers run.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    idx = int(payload["index"])
    num_volumes = int(payload["num_volumes"])
    run_dir = Path(payload["run_dir"])
    camera_shape = tuple(int(v) for v in payload["volume_shape"])
    ss_xy = max(int(payload.get("super_sample_xy", 1)), 1)
    ss_z = max(int(payload.get("super_sample_z", 1)), 1)
    if ss_z != 1:
        raise ValueError("Only XY supersampling is currently implemented; keep super_sample_z=1")
    shape = (camera_shape[0] * ss_z, camera_shape[1] * ss_xy, camera_shape[2] * ss_xy)
    seed = int(payload["seed"])
    rng = np.random.default_rng(seed + idx * 104729)
    split = split_name(idx, num_volumes)
    sample_dir = run_dir / "data" / split / f"sample_{idx:06d}"
    signature = raw_data_signature(payload, split)

    if raw_sample_matches(sample_dir, signature):
        return str(sample_dir)

    morphology = choose_synthetic_morphology(str(payload.get("synthetic_morphology", "mixed")), rng)
    gt = generate_synthetic_gt(
        shape,
        rng,
        complexity=str(payload["synthetic_complexity"]),
        morphology=morphology,
    )
    psf, psf_params = make_small_psf(
        shape,
        rng,
        device="cpu",
        microscope_type=str(payload["microscope_type"]),
        super_sample_xy=ss_xy,
        super_sample_z=ss_z,
    )
    mismatch_mode = choose_psf_mismatch_mode(
        str(payload.get("psf_mismatch", "none")),
        rng,
        float(payload.get("psf_mismatch_moderate_fraction", 0.0)),
    )
    forward_psf, aberration_meta = add_psf_aberration(psf, rng, mismatch_mode)
    deconv_psf_params = perturb_psf_params(psf_params, rng, mismatch_mode)
    deconv_psf = psf.copy() if mismatch_mode == "none" else generate_psf_from_params(deconv_psf_params, device="cpu")
    blurred = reconvolve_same(gt, forward_psf, device="cpu")
    blurred_for_camera = downsample_xy_mean(blurred, ss_xy)
    raw, noise_meta = add_microscopy_noise(
        blurred_for_camera,
        rng,
        microscope_type=str(psf_params.get("microscope_type", payload["microscope_type"])),
        artifact_level=str(payload.get("synthetic_artifact_level", "standard")),
    )
    raw_observed = raw
    raw_for_deconv = upsample_xy_torch(raw_observed, ss_xy)
    noise_meta["raw_observed_shape"] = [int(v) for v in raw_observed.shape]
    noise_meta["raw_deconvolution_shape"] = [int(v) for v in raw_for_deconv.shape]
    noise_meta["raw_deconvolution_dtype"] = "float32" if ss_xy > 1 or ss_z > 1 else "uint16"
    noise_meta["super_sample_xy"] = int(ss_xy)
    noise_meta["super_sample_z"] = int(ss_z)
    metadata = {
        "index": idx,
        "split": split,
        "raw_data_signature": signature,
        "psf_params": psf_params,
        "forward_psf_params": psf_params,
        "deconv_psf_params": deconv_psf_params,
        "psf_mismatch_mode": mismatch_mode,
        "psf_aberration": aberration_meta,
        "noise": noise_meta,
        "camera_volume_shape": [int(v) for v in camera_shape],
        "training_volume_shape": [int(v) for v in shape],
        "super_sample_xy": int(ss_xy),
        "super_sample_z": int(ss_z),
        "synthetic_morphology": morphology,
        "raw_generation_seed": seed + idx * 104729,
    }
    save_raw_sample(sample_dir, gt, raw_for_deconv, deconv_psf, metadata)
    if ss_xy > 1 or ss_z > 1:
        tifffile.imwrite(sample_dir / "raw_observed.tif", raw_observed.astype(np.uint16))
    tifffile.imwrite(sample_dir / "forward_psf.tif", forward_psf.astype(np.float32))
    tifffile.imwrite(sample_dir / "deconv_psf.tif", deconv_psf.astype(np.float32))
    return str(sample_dir)


def generate_training_data(config: TrainConfig, run_dir: Path, progress: Optional[Callable[[str], None]] = None) -> list[Path]:
    sample_dirs: list[Path] = []
    workers = generation_worker_count(config.num_workers)
    payloads = [
        {
            "index": idx,
            "num_volumes": config.num_volumes,
            "run_dir": str(run_dir),
            "volume_shape": list(config.volume_shape),
            "seed": config.seed,
            "synthetic_complexity": config.synthetic_complexity,
            "synthetic_artifact_level": config.synthetic_artifact_level,
            "super_sample_xy": int(config.super_sample_xy),
            "super_sample_z": int(config.super_sample_z),
            "synthetic_morphology": config.synthetic_morphology,
            "microscope_type": config.microscope_type,
            "psf_mismatch": config.psf_mismatch,
            "psf_mismatch_moderate_fraction": config.psf_mismatch_moderate_fraction,
            "rl_iteration_pool": list(effective_rl_iteration_pool(config)),
            "rl_iteration_weights": list(parse_float_pool(config.rl_iteration_weights)),
        }
        for idx in range(config.num_volumes)
    ]

    msg = f"Generating {config.num_volumes} synthetic raw volumes with {workers} CPU worker(s)"
    log.info(msg)
    if progress:
        progress(msg)
    if workers == 1:
        for idx, payload in enumerate(payloads):
            if progress:
                progress(f"Generating synthetic raw volume {idx + 1}/{config.num_volumes}")
            sample_dirs.append(Path(_generate_raw_sample_worker(payload)))
    else:
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_raw_sample_worker, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                sample_dirs.append(Path(future.result()))
                completed += 1
                if progress:
                    progress(f"Generated synthetic raw volume {completed}/{config.num_volumes}")

    sample_dirs = sorted(sample_dirs)
    for idx, sample_dir in enumerate(sample_dirs):
        selected_iterations = select_rl_iterations(sample_dir, config)
        ci_path = ci_rl_path(sample_dir, selected_iterations)
        if ci_rl_matches(sample_dir, selected_iterations):
            if progress:
                progress(f"ci_rl already exists for volume {idx + 1}/{len(sample_dirs)} at {selected_iterations} iterations")
            continue
        if progress:
            progress(f"Running ci_rl preprocessing {idx + 1}/{len(sample_dirs)} at {selected_iterations} iterations")
        raw = tifffile.imread(sample_dir / "raw.tif").astype(np.float32)
        psf = tifffile.imread(sample_dir / "psf.tif").astype(np.float32)
        sample_metadata = read_sample_metadata(sample_dir)
        sample_microscope_type = str(
            sample_metadata.get("psf_params", {}).get("microscope_type", config.microscope_type)
        )
        rl_out = ci_rl_deconvolve(
            raw,
            psf,
            niter=selected_iterations,
            convergence="fixed",
            background="auto",
            offset="auto",
            start="observed",
            microscope_type=sample_microscope_type,
            two_d_mode="legacy_2d",
            device="cpu" if str(config.device).lower() == "cpu" else None,
            tiling="none",
        )
        ci_rl = np.asarray(rl_out["result"], dtype=np.float32)
        tifffile.imwrite(ci_path, ci_rl.astype(np.float32))
        tifffile.imwrite(sample_dir / "ci_rl.tif", ci_rl.astype(np.float32))
        update_sample_metadata(sample_dir, {
            "rl_requested_iterations": int(selected_iterations),
            "rl_iteration_pool": list(effective_rl_iteration_pool(config)),
            "rl_iteration_weights": list(parse_float_pool(config.rl_iteration_weights)),
            "rl_iterations_used": rl_out.get("iterations_used"),
            "rl_convergence": rl_out.get("convergence", []),
        })
        if idx == 0:
            gt = tifffile.imread(sample_dir / "gt.tif").astype(np.float32)
            metadata = read_sample_metadata(sample_dir)
            signal_scale = float(metadata.get("noise", {}).get("signal_scale", 1.0))
            save_montage(run_dir / "figures" / "synthetic_example.png", {"gt_scaled": gt * signal_scale, "raw": raw, "ci_rl": ci_rl})
    summary = {
        "num_volumes": config.num_volumes,
        "volume_shape": list(config.volume_shape),
        "synthetic_complexity": config.synthetic_complexity,
        "synthetic_artifact_level": config.synthetic_artifact_level,
        "super_sample_xy": int(config.super_sample_xy),
        "super_sample_z": int(config.super_sample_z),
        "camera_volume_shape": list(config.volume_shape),
        "training_volume_shape": [
            int(config.volume_shape[0]) * max(int(config.super_sample_z), 1),
            int(config.volume_shape[1]) * max(int(config.super_sample_xy), 1),
            int(config.volume_shape[2]) * max(int(config.super_sample_xy), 1),
        ],
        "synthetic_morphology": config.synthetic_morphology,
        "microscope_type": config.microscope_type,
        "psf_mismatch": config.psf_mismatch,
        "psf_mismatch_moderate_fraction": float(config.psf_mismatch_moderate_fraction),
        "rl_iteration_pool": list(effective_rl_iteration_pool(config)),
        "rl_iteration_weights": list(parse_float_pool(config.rl_iteration_weights)),
        "rl_iteration_counts": {
            str(iteration): sum(1 for sample_dir in sample_dirs if read_sample_metadata(sample_dir).get("rl_requested_iterations") == iteration)
            for iteration in effective_rl_iteration_pool(config)
        },
        "morphologies": {
            name: sum(1 for sample_dir in sample_dirs if read_sample_metadata(sample_dir).get("synthetic_morphology") == name)
            for name in SYNTHETIC_MORPHOLOGIES
            if name != "mixed"
        },
        "splits": {
            split: len(list((run_dir / "data" / split).glob("sample_*")))
            for split in ("train", "val", "test")
        },
        "notes": [
            "Synthetic structures include spots, blobs, filaments, rings/vesicles, sheets, DAPI-like nuclei/chromatin, spindle/actin-like fibers, membranes/cytoplasm, diffuse background, Poisson noise, read noise, hot pixels, and optional clipping.",
            "Strong artifact training adds PSF coma/astigmatism-like mismatch, uneven illumination, haze, row/column banding, read-noise outliers, stronger hot pixels, and mild clipping so DL learns real post-RL failure modes.",
            "XY supersampling generates clean GT and PSFs on a finer grid, forms the camera raw image by downsampling the blurred high-resolution volume, then upsamples the noisy camera raw before high-resolution ci_rl preprocessing.",
            "Filament-like structures are generated as curved 3-D soft tubes rather than straight 2-D line segments.",
            "Synthetic morphology modes include DNA/chromatin, mitotic, membrane, actin, dendrite/branching neurites, puncta, and generic mixed organelle-like structures.",
            "GT files store normalized object density; training rescales GT by the stored synthetic signal_scale so residual targets are in the same photon/count domain as ci_rl.",
            "Volumes are split before patch extraction so validation/test patches do not come from training volumes.",
        ],
    }
    (run_dir / "data" / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return sample_dirs


class SyntheticVolumeDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        *,
        patch_size: int,
        z_radius: int,
        samples_per_epoch: int,
        seed: int,
        use_residual_channel: bool = True,
        cache_size: int = 8,
        use_conditioning: bool = False,
        xy_padding: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sample_dirs = sorted((self.root / "data" / split).glob("sample_*"))
        if not self.sample_dirs:
            raise ValueError(f"No samples found for split '{split}' under {self.root / 'data' / split}")
        self.patch_size = int(patch_size)
        self.z_radius = int(z_radius)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.use_residual_channel = bool(use_residual_channel)
        self.use_conditioning = bool(use_conditioning)
        self.xy_padding = max(int(xy_padding), 0)
        self.cache_size = max(int(cache_size), 0)
        self._cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = OrderedDict()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load(self, sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if self.cache_size > 0 and sample_dir in self._cache:
            cached = self._cache.pop(sample_dir)
            self._cache[sample_dir] = cached
            return cached
        gt = tifffile.imread(sample_dir / "gt.tif").astype(np.float32)
        raw = tifffile.imread(sample_dir / "raw.tif").astype(np.float32)
        ci_rl = tifffile.imread(sample_dir / "ci_rl.tif").astype(np.float32)
        psf = tifffile.imread(sample_dir / "psf.tif").astype(np.float32)
        metadata = read_sample_metadata(sample_dir)
        signal_scale = float(metadata.get("noise", {}).get("signal_scale", 1.0))
        gt = gt * signal_scale
        loaded = (gt, raw, ci_rl, psf, metadata)
        if self.cache_size > 0:
            self._cache[sample_dir] = loaded
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return loaded

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        sample_dir = self.sample_dirs[int(rng.integers(0, len(self.sample_dirs)))]
        gt, raw, ci_rl, psf, metadata = self._load(sample_dir)
        z, h, w = gt.shape
        patch = min(self.patch_size, h, w)
        z_idx = int(rng.integers(0, z))
        y0 = int(rng.integers(0, h - patch + 1)) if h > patch else 0
        x0 = int(rng.integers(0, w - patch + 1)) if w > patch else 0
        yx = (slice(y0, y0 + patch), slice(x0, x0 + patch))
        pad = self.xy_padding
        if pad > 0:
            raw_context = np.pad(raw, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
            ci_context = np.pad(ci_rl, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
            context_yx = (
                slice(y0, y0 + patch + 2 * pad),
                slice(x0, x0 + patch + 2 * pad),
            )
            raw_p = raw_context[:, context_yx[0], context_yx[1]]
            ci_p = ci_context[:, context_yx[0], context_yx[1]]
        else:
            raw_p = raw[:, yx[0], yx[1]]
            ci_p = ci_rl[:, yx[0], yx[1]]
        gt_p = gt[:, yx[0], yx[1]]
        ci_target = ci_rl[:, yx[0], yx[1]]
        x_in, scale = make_25d_input(
            raw_p,
            ci_p,
            z_idx,
            z_radius=self.z_radius,
            use_residual_channel=self.use_residual_channel,
            conditioning_values=conditioning_vector(
                psf=psf,
                metadata=metadata,
                rl_iterations=int(metadata.get("rl_requested_iterations", 0) or 0),
                microscope_type=metadata.get("psf_params", {}).get("microscope_type"),
            ) if self.use_conditioning else None,
        )
        target = ((gt_p[z_idx] - ci_target[z_idx]) / scale).astype(np.float32)[np.newaxis, ...]
        return {
            "input": torch.from_numpy(x_in),
            "target_residual": torch.from_numpy(target),
            "ci": torch.from_numpy((ci_target[z_idx] / scale).astype(np.float32)[np.newaxis, ...]),
            "gt": torch.from_numpy((gt_p[z_idx] / scale).astype(np.float32)[np.newaxis, ...]),
            "psf": torch.from_numpy(psf.astype(np.float32)),
            "bucket": metadata.get("synthetic_morphology", "unknown"),
            "microscope_type": metadata.get("psf_params", {}).get("microscope_type", "unknown"),
            "rl_iterations": torch.tensor(int(metadata.get("rl_requested_iterations", 0) or 0), dtype=torch.int32),
            "psf_mismatch_mode": metadata.get("psf_mismatch_mode", "none"),
        }


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps * eps))


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    tgt_dx = target[..., :, 1:] - target[..., :, :-1]
    tgt_dy = target[..., 1:, :] - target[..., :-1, :]
    return charbonnier(pred_dx, tgt_dx) + charbonnier(pred_dy, tgt_dy)


def crop_like(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Center-crop a fully-convolutional prediction to match the loss target."""
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    y_extra = pred.shape[-2] - target.shape[-2]
    x_extra = pred.shape[-1] - target.shape[-1]
    if y_extra < 0 or x_extra < 0:
        raise ValueError(f"Prediction is smaller than target: {tuple(pred.shape)} vs {tuple(target.shape)}")
    y0 = y_extra // 2
    x0 = x_extra // 2
    return pred[..., y0:y0 + target.shape[-2], x0:x0 + target.shape[-1]]


def negative_residual_guard_loss(
    pred: torch.Tensor,
    ci_plane: torch.Tensor,
    gt_plane: torch.Tensor,
    *,
    max_fraction: float = 0.25,
) -> torch.Tensor:
    """Discourage subtracting too much signal in likely biological structures."""
    ci = ci_plane.clamp(min=0)
    gt = gt_plane.clamp(min=0)
    signal = torch.maximum(ci, gt)
    flat = signal.detach().flatten(start_dim=1)
    threshold = torch.quantile(flat, 0.35, dim=1).view(-1, 1, 1, 1)
    signal_mask = (signal > threshold).to(pred.dtype)
    allowed_negative = max(float(max_fraction), 0.0) * ci
    excess_negative = torch.relu(-pred - allowed_negative)
    return torch.mean(excess_negative * signal_mask)


def intensity_retention_loss(
    pred: torch.Tensor,
    ci_plane: torch.Tensor,
    gt_plane: torch.Tensor,
    *,
    min_ratio: float = 0.90,
    max_ratio: float = 1.15,
) -> torch.Tensor:
    """Limit strong DL corrections from removing or adding too much foreground signal."""
    ci = ci_plane.clamp(min=0)
    gt = gt_plane.clamp(min=0)
    signal = torch.maximum(ci.detach(), gt.detach())
    threshold = torch.quantile(signal.flatten(start_dim=1), 0.50, dim=1).view(-1, 1, 1, 1)
    mask = (signal > threshold).to(ci.dtype)
    final = (ci + pred).clamp(min=0)
    eps = torch.as_tensor(1e-6, dtype=ci.dtype, device=ci.device)
    ci_sum = torch.sum(ci * mask, dim=(1, 2, 3)) + eps
    final_sum = torch.sum(final * mask, dim=(1, 2, 3))
    ratio = final_sum / ci_sum
    low = torch.relu(float(min_ratio) - ratio)
    high = torch.relu(ratio - float(max_ratio))
    return torch.mean(low * low + high * high)


def global_intensity_ratio_loss(
    pred: torch.Tensor,
    ci_plane: torch.Tensor,
    *,
    min_ratio: float = 0.98,
    max_ratio: float = 1.02,
) -> torch.Tensor:
    """Keep total refined image energy near the ci_rl image energy."""
    ci = ci_plane.clamp(min=0)
    final = (ci + pred).clamp(min=0)
    eps = torch.as_tensor(1e-6, dtype=ci.dtype, device=ci.device)
    ratio = torch.sum(final, dim=(1, 2, 3)) / (torch.sum(ci, dim=(1, 2, 3)) + eps)
    low = torch.relu(float(min_ratio) - ratio)
    high = torch.relu(ratio - float(max_ratio))
    return torch.mean(low * low + high * high)


def background_offset_loss(
    pred: torch.Tensor,
    ci_plane: torch.Tensor,
    gt_plane: torch.Tensor,
) -> torch.Tensor:
    """Discourage the residual from adding/removing a constant background floor."""
    ci = ci_plane.clamp(min=0)
    gt = gt_plane.clamp(min=0)
    signal = torch.maximum(ci.detach(), gt.detach())
    threshold = torch.quantile(signal.flatten(start_dim=1), 0.20, dim=1).view(-1, 1, 1, 1)
    bg_mask = (signal <= threshold).to(pred.dtype)
    denom = torch.sum(bg_mask, dim=(1, 2, 3)).clamp(min=1.0)
    offset = torch.sum(pred * bg_mask, dim=(1, 2, 3)) / denom
    return torch.mean(offset * offset)


def central_plane_reconvolution_loss(
    final_plane: torch.Tensor,
    raw_central: torch.Tensor,
    psf_batch: torch.Tensor,
) -> torch.Tensor:
    """Approximate data consistency using each sample's central PSF plane."""
    losses = []
    for idx in range(final_plane.shape[0]):
        psf = psf_batch[idx]
        if psf.ndim == 3:
            psf = psf[psf.shape[0] // 2]
        psf = psf / psf.sum().clamp(min=1e-12)
        kernel = psf.to(final_plane.device, dtype=final_plane.dtype)[None, None]
        pred_raw = torch.nn.functional.conv2d(
            final_plane[idx:idx + 1],
            kernel,
            padding=(kernel.shape[-2] // 2, kernel.shape[-1] // 2),
        )
        losses.append(charbonnier(pred_raw, raw_central[idx:idx + 1]))
    return torch.stack(losses).mean()


def save_loss_curve(path: Path, history: list[dict[str, float]]) -> None:
    if plt is not None and history:
        fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=120)
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["train_loss"] for row in history], label="train Charbonnier")
        ax.plot(epochs, [row["val_loss"] for row in history], label="val Charbonnier")
        if any("train_total_loss" in row for row in history):
            ax.plot(
                epochs,
                [row.get("train_total_loss", row["train_loss"]) for row in history],
                linestyle="--",
                alpha=0.65,
                label="train total objective",
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("ci_rl_dl training loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    w, h = 640, 360
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    margin = 45
    draw.rectangle((margin, margin, w - margin, h - margin), outline=(0, 0, 0))
    if history:
        values = (
            [r["train_loss"] for r in history]
            + [r["val_loss"] for r in history if math.isfinite(r["val_loss"])]
            + [r["train_total_loss"] for r in history if math.isfinite(r.get("train_total_loss", float("nan")))]
        )
        vmin, vmax = min(values), max(values)
        if vmax <= vmin:
            vmax = vmin + 1.0
        def point(i: int, key: str) -> tuple[int, int]:
            x = margin + int((w - 2 * margin) * i / max(len(history) - 1, 1))
            v = history[i].get(key, history[i]["train_loss"])
            y = h - margin - int((h - 2 * margin) * (v - vmin) / (vmax - vmin))
            return x, y
        for key, color in (("train_loss", (20, 90, 180)), ("val_loss", (180, 60, 20)), ("train_total_loss", (90, 90, 90))):
            pts = [point(i, key) for i in range(len(history)) if math.isfinite(history[i].get(key, float("nan")))]
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)
        draw.text((margin, 12), "ci_rl_dl training loss", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    pin_memory = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device, non_blocking=pin_memory)
            target = batch["target_residual"].to(device, non_blocking=pin_memory)
            pred = crop_like(model(x), target)
            losses.append(float(charbonnier(pred, target).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_by_bucket(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    pin_memory = device.type == "cuda"
    buckets: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device, non_blocking=pin_memory)
            target = batch["target_residual"].to(device, non_blocking=pin_memory)
            pred = crop_like(model(x), target)
            per_sample = torch.sqrt((pred - target) ** 2 + 1e-6).mean(dim=(1, 2, 3)).detach().cpu().numpy()
            batch_buckets = batch.get("bucket", ["unknown"] * len(per_sample))
            for bucket, value in zip(batch_buckets, per_sample, strict=False):
                buckets.setdefault(str(bucket), []).append(float(value))
    return {bucket: float(np.mean(values)) for bucket, values in sorted(buckets.items()) if values}


def save_example_prediction(model: torch.nn.Module, dataset: SyntheticVolumeDataset, device: torch.device, path: Path) -> None:
    model.eval()
    item = dataset[0]
    with torch.no_grad():
        target = item["target_residual"].unsqueeze(0).to(device)
        pred = crop_like(model(item["input"].unsqueeze(0).to(device)), target).cpu()[0]
    ci = item["ci"].numpy()[0]
    gt = item["gt"].numpy()[0]
    refined = np.clip(ci + pred.numpy()[0], 0, None)
    save_montage(path, {"ci_rl": ci, "refined": refined, "gt": gt, "residual": pred.numpy()[0]})


def save_bucket_prediction_examples(model: torch.nn.Module, dataset: SyntheticVolumeDataset, device: torch.device, out_dir: Path) -> None:
    seen: set[str] = set()
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample_dir in dataset.sample_dirs:
        metadata = read_sample_metadata(sample_dir)
        bucket = str(metadata.get("synthetic_morphology", "unknown"))
        if bucket in seen:
            continue
        gt, raw, ci_rl, psf, metadata = dataset._load(sample_dir)
        z_idx = gt.shape[0] // 2
        cond = conditioning_vector(
            psf=psf,
            metadata=metadata,
            rl_iterations=int(metadata.get("rl_requested_iterations", 0) or 0),
            microscope_type=metadata.get("psf_params", {}).get("microscope_type"),
        ) if dataset.use_conditioning else None
        x_in, scale = make_25d_input(raw, ci_rl, z_idx, z_radius=dataset.z_radius, conditioning_values=cond)
        with torch.no_grad():
            pred = model(torch.from_numpy(x_in).unsqueeze(0).to(device)).cpu().numpy()[0, 0] * scale
        refined = np.clip(ci_rl[z_idx] + pred, 0, None)
        save_montage(out_dir / f"{bucket}.png", {"ci_rl": ci_rl[z_idx], "refined": refined, "gt": gt[z_idx], "residual": pred})
        seen.add(bucket)


def checkpoint_metadata(
    config: TrainConfig,
    model_kwargs: dict[str, Any],
    history: list[dict[str, float]],
) -> dict[str, Any]:
    best = min(history, key=lambda row: row["val_loss"]) if history else None
    return {
        "model_type": str(config.model_type),
        "version": 1,
        "model_kwargs": model_kwargs,
        "normalization": {"mode": "per_sample_p99"},
        "target": "residual",
        "input_channels": model_kwargs["input_channels"],
        "use_residual_channel": True,
        "conditioning_channels": CONDITIONING_CHANNELS if config.use_conditioning else [],
        "residual_bound": {
            "fraction": float(config.residual_bound_fraction),
            "scale": float(config.residual_bound_scale),
        },
        "training_config": {**asdict(config), "output_dir": str(config.output_dir)},
        "recommended_inference": {
            "method": "ci_rl_dl",
            "iterations": int(config.rl_iterations),
            "rl_iteration_pool": list(effective_rl_iteration_pool(config)),
            "rl_kwargs": {
                "niter": int(config.rl_iterations),
                "start": "observed",
                "convergence": "fixed",
                "background": "auto",
                "offset": "auto",
                "two_d_mode": "legacy_2d",
            },
            "dl_kwargs": {
                "z_radius": int(config.z_context),
                "batch_size": max(int(config.batch_size), 1),
                "mixed_precision": bool(config.mixed_precision),
                "xy_padding": int(config.training_xy_padding),
            },
            "dl_z_context": int(config.z_context),
            "dl_batch_size": max(int(config.batch_size), 1),
            "dl_mixed_precision": bool(config.mixed_precision),
        },
        "training_history": history,
        "training_domain": {
            "microscope_type": config.microscope_type,
            "synthetic_complexity": config.synthetic_complexity,
            "synthetic_morphology": config.synthetic_morphology,
            "synthetic_artifact_level": config.synthetic_artifact_level,
            "super_sample_xy": int(config.super_sample_xy),
            "super_sample_z": int(config.super_sample_z),
            "training_xy_padding": int(config.training_xy_padding),
        },
        "best_epoch": best,
    }


def train(
    config: TrainConfig,
    progress: Optional[Callable[[str], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> Path:
    set_seed(config.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir
    if run_dir.name == "training_runs":
        run_dir = run_dir / f"run_{timestamp}"
    for sub in ("data/train", "data/val", "data/test", "checkpoints", "logs", "figures", "examples"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**asdict(config), "output_dir": str(config.output_dir)}, indent=2), encoding="utf-8")

    generate_training_data(config, run_dir, progress=progress)
    device = resolve_torch_device(config.device)
    loader_workers = data_loader_worker_count(config.data_loader_workers, device)
    pin_memory = device.type == "cuda"
    if progress:
        progress(
            f"Training data loader: workers={loader_workers} "
            f"pin_memory={pin_memory} volume_cache_size={config.volume_cache_size}"
        )
    train_ds = SyntheticVolumeDataset(
        run_dir,
        "train",
        patch_size=config.patch_size,
        z_radius=config.z_context,
        samples_per_epoch=config.steps * config.batch_size if config.steps else config.train_samples_per_epoch,
        seed=config.seed + 1000,
        cache_size=config.volume_cache_size,
        use_conditioning=config.use_conditioning,
        xy_padding=config.training_xy_padding,
    )
    val_ds = SyntheticVolumeDataset(
        run_dir,
        "val" if (run_dir / "data" / "val").exists() and list((run_dir / "data" / "val").glob("sample_*")) else "train",
        patch_size=config.patch_size,
        z_radius=config.z_context,
        samples_per_epoch=config.val_samples,
        seed=config.seed + 2000,
        cache_size=config.volume_cache_size,
        use_conditioning=config.use_conditioning,
        xy_padding=config.training_xy_padding,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": loader_workers,
        "pin_memory": pin_memory,
        "persistent_workers": loader_workers > 0,
    }
    if loader_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    conditioning_channels = CONDITIONING_CHANNELS if config.use_conditioning else []
    model_kwargs = {
        "input_channels": input_channel_count(config.z_context, True, conditioning_channels),
        "base_channels": config.base_channels,
        "residual_scale": float(config.residual_scale),
    }
    model_type = str(config.model_type)
    if model_type == "GatedResidualUNet25D":
        model_kwargs.update({
            "z_radius": int(config.z_context),
            "use_residual_channel": True,
            "residual_bound_fraction": float(config.residual_bound_fraction),
            "residual_bound_scale": float(config.residual_bound_scale),
        })
        model = GatedResidualUNet25D(**model_kwargs).to(device)
    else:
        model_type = "ResidualUNet25D"
        model = ResidualUNet25D(**model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.mixed_precision and device.type == "cuda"))
    history: list[dict[str, float]] = []
    max_epochs = config.epochs
    steps_done = 0
    total_steps = int(config.steps) if config.steps else max_epochs * len(train_loader)
    best_val_loss = float("inf")
    training_started_at = time.monotonic()
    step_timepoints: list[tuple[int, float]] = []
    epoch_timepoints: list[tuple[int, float]] = []
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        if stop_requested and stop_requested():
            stopped_early = True
            if progress:
                progress("Stop requested before next epoch; saving current training artifacts.")
            break
        epoch_started_at = time.monotonic()
        model.train()
        train_total_losses: list[float] = []
        train_charbonnier_losses: list[float] = []
        for batch in train_loader:
            if stop_requested and stop_requested() and train_total_losses:
                stopped_early = True
                if progress:
                    progress("Stop requested; finishing current epoch summary and saving artifacts.")
                break
            x = batch["input"].to(device, non_blocking=pin_memory)
            target = batch["target_residual"].to(device, non_blocking=pin_memory)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(config.mixed_precision and device.type == "cuda")):
                pred = crop_like(model(x), target)
                charbonnier_loss = charbonnier(pred, target)
                loss = charbonnier_loss
                ci_plane = batch["ci"].to(device, non_blocking=pin_memory)
                gt_plane = batch["gt"].to(device, non_blocking=pin_memory)
                if config.gradient_weight > 0:
                    loss = loss + config.gradient_weight * gradient_loss(pred, target)
                if config.negative_residual_weight > 0:
                    loss = loss + config.negative_residual_weight * negative_residual_guard_loss(
                        pred,
                        ci_plane,
                        gt_plane,
                        max_fraction=config.max_negative_residual_fraction,
                    )
                if config.intensity_retention_weight > 0:
                    loss = loss + config.intensity_retention_weight * intensity_retention_loss(
                        pred,
                        ci_plane,
                        gt_plane,
                        min_ratio=config.intensity_retention_min,
                        max_ratio=config.intensity_retention_max,
                    )
                if config.global_intensity_weight > 0:
                    loss = loss + config.global_intensity_weight * global_intensity_ratio_loss(
                        pred,
                        ci_plane,
                        min_ratio=config.global_intensity_min,
                        max_ratio=config.global_intensity_max,
                    )
                if config.background_offset_weight > 0:
                    loss = loss + config.background_offset_weight * background_offset_loss(
                        pred,
                        ci_plane,
                        gt_plane,
                    )
                if config.reconvolution_weight > 0:
                    final_plane = (ci_plane + pred).clamp(min=0)
                    raw_central = crop_like(x[:, config.z_context:config.z_context + 1], pred)
                    loss = loss + config.reconvolution_weight * central_plane_reconvolution_loss(
                        final_plane,
                        raw_central,
                        batch["psf"].to(device, non_blocking=pin_memory),
                    )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_total_losses.append(float(loss.detach().cpu()))
            train_charbonnier_losses.append(float(charbonnier_loss.detach().cpu()))
            steps_done += 1
            if steps_done % 5 == 0:
                elapsed = time.monotonic() - training_started_at
                step_timepoints.append((steps_done, elapsed))
                eta_seconds = None
                if len(step_timepoints) >= 3 and total_steps > steps_done:
                    first_step, first_elapsed = step_timepoints[0]
                    step_delta = max(steps_done - first_step, 1)
                    time_delta = max(elapsed - first_elapsed, 1e-6)
                    eta_seconds = (total_steps - steps_done) * (time_delta / step_delta)
                step_message = (
                    f"{timestamp_now()} Training step {steps_done}/{total_steps}: "
                    f"loss={train_charbonnier_losses[-1]:.5f} total={train_total_losses[-1]:.5f} "
                    f"elapsed={format_duration(elapsed)} "
                    f"eta={format_duration(eta_seconds)} finish={format_finish_time(eta_seconds)}"
                )
                log.info(step_message)
                if progress:
                    progress(step_message)
            if config.steps and steps_done >= config.steps:
                break
            if stop_requested and stop_requested():
                stopped_early = True
                if progress:
                    progress("Stop requested; finishing current epoch summary and saving artifacts.")
                break
        if not train_total_losses:
            break
        val_loss = evaluate(model, val_loader, device)
        bucket_losses = evaluate_by_bucket(model, val_loader, device)
        row = {
            "epoch": float(epoch),
            "step": float(steps_done),
            "train_loss": float(np.mean(train_charbonnier_losses)),
            "train_total_loss": float(np.mean(train_total_losses)),
            "val_loss": val_loss,
            **{f"val_{bucket}_loss": value for bucket, value in bucket_losses.items()},
        }
        history.append(row)
        elapsed = time.monotonic() - training_started_at
        epoch_timepoints.append((epoch, elapsed))
        eta_seconds = None
        effective_total_epochs = epoch if config.steps and steps_done >= config.steps else max_epochs
        if len(epoch_timepoints) >= 2 and effective_total_epochs > epoch:
            first_epoch, first_elapsed = epoch_timepoints[0]
            epoch_delta = max(epoch - first_epoch, 1)
            time_delta = max(elapsed - first_elapsed, 1e-6)
            eta_seconds = (effective_total_epochs - epoch) * (time_delta / epoch_delta)
        epoch_message = (
            f"{timestamp_now()} Epoch {epoch}/{effective_total_epochs}: "
            f"train={row['train_loss']:.5f} val={val_loss:.5f} total={row['train_total_loss']:.5f} "
            f"epoch_time={format_duration(time.monotonic() - epoch_started_at)} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta_seconds)} "
            f"finish={format_finish_time(eta_seconds)}"
        )
        log.info(epoch_message)
        if progress:
            progress(epoch_message)
        metadata = checkpoint_metadata(config, model_kwargs, history)
        checkpoint = {
            **metadata,
            "state_dict": model.state_dict(),
            "z_radius": config.z_context,
            "config": {**asdict(config), "output_dir": str(config.output_dir)},
        }
        torch.save(checkpoint, run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = run_dir / "checkpoints" / "best_model.pt"
            torch.save(checkpoint, best_path)
            best_path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
        if stopped_early or (config.steps and steps_done >= config.steps):
            break

    final_path = run_dir / "checkpoints" / "final_model.pt"
    metadata = checkpoint_metadata(config, model_kwargs, history)
    metadata["training_stopped_early"] = bool(stopped_early)
    metadata["steps_done"] = int(steps_done)
    torch.save({
        **metadata,
        "state_dict": model.state_dict(),
        "z_radius": config.z_context,
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
    }, final_path)
    (final_path.with_suffix(".json")).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    loss_fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in loss_fieldnames:
                loss_fieldnames.append(key)
    with (run_dir / "logs" / "losses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=loss_fieldnames or ["epoch", "step", "train_loss", "train_total_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    (run_dir / "logs" / "losses.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    bucket_history = [
        {"epoch": row.get("epoch"), **{key.removeprefix("val_").removesuffix("_loss"): value for key, value in row.items() if key.startswith("val_") and key.endswith("_loss") and key != "val_loss"}}
        for row in history
    ]
    (run_dir / "logs" / "validation_buckets.json").write_text(json.dumps(bucket_history, indent=2), encoding="utf-8")
    bucket_fields: list[str] = []
    for row in bucket_history:
        for key in row:
            if key not in bucket_fields:
                bucket_fields.append(key)
    with (run_dir / "logs" / "validation_buckets.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bucket_fields or ["epoch"])
        writer.writeheader()
        writer.writerows(bucket_history)
    save_loss_curve(run_dir / "figures" / "loss_curve.png", history)
    save_example_prediction(model, val_ds, device, run_dir / "examples" / "prediction_montage.png")
    save_bucket_prediction_examples(model, val_ds, device, run_dir / "examples" / "buckets")
    if progress:
        status = "Stopped training and saved artifacts" if stopped_early else "Finished training"
        progress(f"{status}: {final_path}")
    return run_dir


def build_config(args: argparse.Namespace) -> TrainConfig:
    if args.full_experiment:
        return TrainConfig(
            num_volumes=args.num_volumes if args.num_volumes != 24 else 1000,
            volume_shape=args.volume_shape if args.volume_shape != (16, 96, 96) else (32, 256, 256),
            patch_size=args.patch_size if args.patch_size != 64 else 128,
            z_context=args.z_context,
            batch_size=args.batch_size,
            epochs=args.epochs if args.epochs != 2 else 50,
            steps=args.steps,
            learning_rate=args.learning_rate,
            output_dir=Path(args.output_dir),
            device=args.device,
            mixed_precision=args.mixed_precision,
            seed=args.seed,
            base_channels=args.base_channels if args.base_channels != 16 else 48,
            residual_scale=args.residual_scale,
            rl_iterations=args.rl_iterations if args.rl_iterations != 8 else 50,
            rl_iteration_pool=parse_int_pool(args.rl_iteration_pool) or (50, 80, 100),
            rl_iteration_weights=parse_float_pool(args.rl_iteration_weights) or (0.35, 0.40, 0.25),
            reconvolution_weight=args.reconvolution_weight if args.reconvolution_weight != 0.0 else 0.02,
            gradient_weight=args.gradient_weight if args.gradient_weight != 0.05 else 0.08,
            train_samples_per_epoch=args.train_samples_per_epoch,
            val_samples=args.val_samples,
            synthetic_complexity="full",
            synthetic_artifact_level=args.synthetic_artifact_level if args.synthetic_artifact_level != "standard" else "strong",
            synthetic_morphology=args.synthetic_morphology,
            microscope_type=args.microscope_type,
            psf_mismatch=args.psf_mismatch if args.psf_mismatch != "none" else "mild",
            psf_mismatch_moderate_fraction=args.psf_mismatch_moderate_fraction if args.psf_mismatch_moderate_fraction != 0.0 else 0.5,
            model_type=args.model_type,
            use_conditioning=args.use_conditioning,
            residual_bound_fraction=args.residual_bound_fraction if args.residual_bound_fraction != 0.35 else 0.75,
            residual_bound_scale=args.residual_bound_scale if args.residual_bound_scale != 0.05 else 0.10,
            num_workers=args.num_workers,
            data_loader_workers=args.data_loader_workers,
            volume_cache_size=args.volume_cache_size,
            negative_residual_weight=args.negative_residual_weight if args.negative_residual_weight != 0.05 else 0.01,
            max_negative_residual_fraction=args.max_negative_residual_fraction if args.max_negative_residual_fraction != 0.25 else 0.50,
            intensity_retention_weight=args.intensity_retention_weight if args.intensity_retention_weight != 0.0 else 0.05,
            intensity_retention_min=args.intensity_retention_min,
            intensity_retention_max=args.intensity_retention_max,
            global_intensity_weight=args.global_intensity_weight,
            global_intensity_min=args.global_intensity_min,
            global_intensity_max=args.global_intensity_max,
            background_offset_weight=args.background_offset_weight,
            training_xy_padding=args.training_xy_padding,
            super_sample_xy=args.super_sample_xy,
            super_sample_z=args.super_sample_z,
        )
    if args.quick_test:
        return TrainConfig(
            num_volumes=4,
            volume_shape=(5, 32, 32),
            patch_size=24,
            z_context=1,
            batch_size=2,
            epochs=1,
            steps=2,
            learning_rate=args.learning_rate,
            output_dir=Path(args.output_dir),
            device=args.device,
            mixed_precision=args.mixed_precision,
            seed=args.seed,
            base_channels=8,
            residual_scale=args.residual_scale,
            rl_iterations=2,
            rl_iteration_pool=(2,),
            rl_iteration_weights=(),
            quick_test=True,
            train_samples_per_epoch=4,
            val_samples=2,
            reconvolution_weight=args.reconvolution_weight,
            gradient_weight=args.gradient_weight,
            synthetic_complexity=args.synthetic_complexity,
            synthetic_artifact_level="standard",
            microscope_type=args.microscope_type,
            synthetic_morphology=args.synthetic_morphology,
            psf_mismatch="none",
            model_type=args.model_type,
            use_conditioning=args.use_conditioning,
            residual_bound_fraction=args.residual_bound_fraction,
            residual_bound_scale=args.residual_bound_scale,
            num_workers=1,
            data_loader_workers=0,
            volume_cache_size=2,
            negative_residual_weight=args.negative_residual_weight,
            max_negative_residual_fraction=args.max_negative_residual_fraction,
            intensity_retention_weight=args.intensity_retention_weight,
            intensity_retention_min=args.intensity_retention_min,
            intensity_retention_max=args.intensity_retention_max,
            global_intensity_weight=args.global_intensity_weight,
            global_intensity_min=args.global_intensity_min,
            global_intensity_max=args.global_intensity_max,
            background_offset_weight=args.background_offset_weight,
            training_xy_padding=args.training_xy_padding,
            super_sample_xy=1,
            super_sample_z=1,
        )
    return TrainConfig(
        num_volumes=args.num_volumes,
        volume_shape=args.volume_shape,
        patch_size=args.patch_size,
        z_context=args.z_context,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps=args.steps,
        learning_rate=args.learning_rate,
        output_dir=Path(args.output_dir),
        device=args.device,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        base_channels=args.base_channels,
        residual_scale=args.residual_scale,
        rl_iterations=args.rl_iterations,
        rl_iteration_pool=parse_int_pool(args.rl_iteration_pool),
        rl_iteration_weights=parse_float_pool(args.rl_iteration_weights),
        reconvolution_weight=args.reconvolution_weight,
        gradient_weight=args.gradient_weight,
        train_samples_per_epoch=args.train_samples_per_epoch,
        val_samples=args.val_samples,
        synthetic_complexity=args.synthetic_complexity,
        synthetic_artifact_level=args.synthetic_artifact_level,
        synthetic_morphology=args.synthetic_morphology,
        microscope_type=args.microscope_type,
        psf_mismatch=args.psf_mismatch,
        psf_mismatch_moderate_fraction=args.psf_mismatch_moderate_fraction,
        model_type=args.model_type,
        use_conditioning=args.use_conditioning,
        residual_bound_fraction=args.residual_bound_fraction,
        residual_bound_scale=args.residual_bound_scale,
        num_workers=args.num_workers,
        data_loader_workers=args.data_loader_workers,
        volume_cache_size=args.volume_cache_size,
        negative_residual_weight=args.negative_residual_weight,
        max_negative_residual_fraction=args.max_negative_residual_fraction,
        intensity_retention_weight=args.intensity_retention_weight,
        intensity_retention_min=args.intensity_retention_min,
        intensity_retention_max=args.intensity_retention_max,
        global_intensity_weight=args.global_intensity_weight,
        global_intensity_min=args.global_intensity_min,
        global_intensity_max=args.global_intensity_max,
        background_offset_weight=args.background_offset_weight,
        training_xy_padding=args.training_xy_padding,
        super_sample_xy=args.super_sample_xy,
        super_sample_z=args.super_sample_z,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--full-experiment", action="store_true")
    parser.add_argument("--num-volumes", type=int, default=24)
    parser.add_argument("--volume-shape", type=parse_volume_shape, default=(16, 96, 96))
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--z-context", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="training_runs")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--residual-scale", type=float, default=1.0, help="Fixed multiplier applied to the U-Net residual output; 1.0 preserves the current behavior")
    parser.add_argument("--rl-iterations", type=int, default=8)
    parser.add_argument("--rl-iteration-pool", type=str, default="", help="Comma-separated RL iterations sampled per synthetic volume, e.g. 15,25,35,50,80")
    parser.add_argument("--rl-iteration-weights", type=str, default="", help="Optional comma-separated sampling weights matching --rl-iteration-pool, e.g. 0.35,0.40,0.25")
    parser.add_argument("--reconvolution-weight", type=float, default=0.0)
    parser.add_argument("--gradient-weight", type=float, default=0.05)
    parser.add_argument("--negative-residual-weight", type=float, default=0.05, help="Penalty weight for excessive negative DL residuals in signal regions")
    parser.add_argument("--max-negative-residual-fraction", type=float, default=0.25, help="Allowed negative residual as a fraction of normalized ci_rl signal before penalty")
    parser.add_argument("--intensity-retention-weight", type=float, default=0.0, help="Penalty weight for foreground intensity changes outside the configured ratio range")
    parser.add_argument("--intensity-retention-min", type=float, default=0.90)
    parser.add_argument("--intensity-retention-max", type=float, default=1.15)
    parser.add_argument("--global-intensity-weight", type=float, default=0.0, help="Penalty weight for total image energy changes outside the configured ratio range")
    parser.add_argument("--global-intensity-min", type=float, default=0.98)
    parser.add_argument("--global-intensity-max", type=float, default=1.02)
    parser.add_argument("--background-offset-weight", type=float, default=0.0, help="Penalty weight for mean residual offset in background-like pixels")
    parser.add_argument("--training-xy-padding", type=int, default=0, help="Reflect-padding halo used for DL training patches; loss is cropped to the central patch")
    parser.add_argument("--train-samples-per-epoch", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--synthetic-complexity", choices=["standard", "full"], default="standard")
    parser.add_argument("--synthetic-artifact-level", choices=["standard", "strong"], default="standard")
    parser.add_argument("--super-sample-xy", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--super-sample-z", type=int, choices=[1], default=1)
    parser.add_argument("--synthetic-morphology", choices=list(SYNTHETIC_MORPHOLOGIES), default="mixed")
    parser.add_argument("--microscope-type", choices=["widefield", "confocal", "mixed"], default="widefield")
    parser.add_argument("--psf-mismatch", choices=["none", "mild", "moderate"], default="none")
    parser.add_argument("--psf-mismatch-moderate-fraction", type=float, default=0.0)
    parser.add_argument("--model-type", choices=["ResidualUNet25D", "GatedResidualUNet25D"], default="GatedResidualUNet25D")
    parser.add_argument("--use-conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual-bound-fraction", type=float, default=0.35)
    parser.add_argument("--residual-bound-scale", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0, help="CPU workers for synthetic raw volume generation; 0 uses CPU count minus one")
    parser.add_argument("--data-loader-workers", type=int, default=0, help="PyTorch DataLoader workers during training; 0 auto-selects for CUDA and disables workers for CPU")
    parser.add_argument("--volume-cache-size", type=int, default=8, help="Number of loaded volumes cached per DataLoader worker")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_dir = train(build_config(args))
    print(run_dir)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
