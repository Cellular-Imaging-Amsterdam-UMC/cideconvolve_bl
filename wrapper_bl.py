"""
wrapper.py - Bilayers-compatible entrypoint for CIDeconvolve.

Parses Bilayers job parameters (--infolder, --outfolder, --gtfolder, etc.)
via bilayers_local, then processes each input image through the CI
deconvolution pipeline in deconvolve.py and writes results to the output
folder.

Usage (inside Docker):
    python wrapper.py --infolder /data/in --outfolder /data/out --gtfolder /data/gt --local

Usage (local):
    python wrapper.py --infolder ./infolder --outfolder ./outfolder --gtfolder ./gtfolder --local --iterations "40" --method ci_rl
"""
import csv
import logging
import os
import platform
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

# Configure logging so deconvolve.py INFO messages are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from bilayers_local import (
    CLASS_SPTCNT,
    BilayersJob,
    get_discipline,
    prepare_data,
)

# Import core first (handles torch-before-numpy DLL load order)
from core.deconvolve import (
    _DEFAULT_PINHOLE_AIRY_UNITS,
    _apply_pinhole_airy_units,
    _format_float_list,
    deconvolve,
    deconvolve_image,
    generate_psf,
    load_image,
    save_mip_png,
    save_result,
)
from core.streaming import (
    ZarrPyramidSink,
    deconvolve_streaming,
    open_region_source,
    save_streaming_provenance,
    should_stream_source,
    suggest_streaming_tile_size,
)

import numpy as np

# ---------------------------------------------------------------------------
# Output zarr format: 2 for OMERO compatibility, 3 for latest zarr spec
# ---------------------------------------------------------------------------
OUTPUT_ZARR_FORMAT = 2

# ---------------------------------------------------------------------------
# RI lookup tables - value-choices in descriptor.json use "name (RI)" format
# ---------------------------------------------------------------------------
_IMMERSION_RI = {
    "air":   1.0003,
    "water": 1.333,
    "oil":   1.515,
}

_SAMPLE_RI = {
    "water":          1.333,
    "pbs":            1.334,
    "culture medium": 1.337,
    "vectashield":    1.45,
    "prolong gold":   1.47,
    "glycerol":       1.474,
    "oil":            1.515,
    "prolong glass":  1.52,
}

# GUI-matched metadata fallback defaults.
_DEFAULT_NA = 1.4
_DEFAULT_EMISSION_WL = "520"
_DEFAULT_PIXEL_SIZE_XY_NM = 65.0
_DEFAULT_PIXEL_SIZE_Z_NM = 200.0
_DEFAULT_MICROSCOPE_TYPE = "confocal"
_DEFAULT_EXCITATION_WL = "488"
_DEFAULT_PINHOLE_AIRY = _DEFAULT_PINHOLE_AIRY_UNITS
_DEFAULT_IMMERSION_RI_CHOICE = "oil (1.515)"
_DEFAULT_SAMPLE_RI_CHOICE = "prolong gold (1.47)"
_SAMPLE_RI_DEFAULT = 1.47
_START_MODES = (
    "auto",
    "flat",
    "percentile_flat",
    "observed",
    "observed_bgsub",
    "lowpass",
    "lowpass_bgsub",
    "hybrid",
)


def _to_bool(value) -> bool:
    """Convert a value to bool, handling string 'True'/'False' from CLI."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _parse_ri_choice(raw: str, lookup: dict[str, float]) -> float | None:
    """Parse a RI choice string like 'oil (1.515)' or a bare float.

    Returns None when the value cannot be parsed.
    """
    s = str(raw).strip().lower()
    if not s:
        return None
    # Try "name (1.234)" format - extract the name part
    name = s.split("(")[0].strip()
    if name in lookup:
        return lookup[name]
    # Try bare float
    try:
        return float(s)
    except ValueError:
        return None


def _parse_float_or_default(raw, default: float) -> float:
    """Parse a float, accepting legacy 'auto' as the supplied default."""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _parse_float_list_or_default(raw, default: str) -> list[float]:
    """Parse comma-separated floats, accepting legacy 'auto' as default."""
    text = str(raw if raw is not None else default).strip()
    if not text or text.lower() == "auto":
        text = default
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    return values or [float(default)]


def _parse_tile_limits(raw, default: tuple[int, int] = (0, 64)) -> tuple[int, int]:
    """Parse tile limits as ``max_xy,max_z``; XY <= 0 means auto tile sizing."""
    text = str(raw or "").strip()
    if not text or text.lower() == "auto":
        return default
    parts = [p.strip() for p in text.replace("x", ",").split(",") if p.strip()]
    try:
        max_xy = int(parts[0]) if parts else default[0]
        max_z = int(parts[1]) if len(parts) > 1 else default[1]
    except ValueError:
        return default
    if max_xy <= 0:
        max_xy = 0
    return (max_xy if max_xy == 0 else max(max_xy, 64)), max(max_z, 1)


def _metadata_use_value(current, value, overrule_metadata: bool) -> bool:
    return value is not None and (overrule_metadata or current is None)


def _apply_cli_metadata_to_source(
    meta: dict,
    *,
    na,
    refractive_index,
    sample_ri,
    microscope_type,
    emission_wavelengths,
    excitation_wavelengths,
    pinhole_airy_units,
    pixel_size_xy,
    pixel_size_z,
    overrule_metadata: bool,
) -> dict:
    """Mirror load_image metadata override semantics for streaming sources."""
    meta = dict(meta)
    if _metadata_use_value(meta.get("na"), na, overrule_metadata):
        meta["na"] = na
    if _metadata_use_value(meta.get("refractive_index"), refractive_index, overrule_metadata):
        meta["refractive_index"] = refractive_index
    if _metadata_use_value(meta.get("sample_refractive_index"), sample_ri, overrule_metadata):
        meta["sample_refractive_index"] = sample_ri
    if _metadata_use_value(meta.get("microscope_type"), microscope_type, overrule_metadata):
        meta["microscope_type"] = microscope_type
    if _metadata_use_value(meta.get("pixel_size_x"), pixel_size_xy, overrule_metadata):
        meta["pixel_size_x"] = pixel_size_xy
    if _metadata_use_value(meta.get("pixel_size_y"), pixel_size_xy, overrule_metadata):
        meta["pixel_size_y"] = pixel_size_xy
    if _metadata_use_value(meta.get("pixel_size_z"), pixel_size_z, overrule_metadata):
        meta["pixel_size_z"] = pixel_size_z

    n_channels = int(meta.get("size_c") or meta.get("n_channels") or 1)
    channels = [dict(ch) if isinstance(ch, dict) else {} for ch in meta.get("channels", [])]
    if len(channels) < n_channels:
        channels.extend({} for _ in range(n_channels - len(channels)))
    if emission_wavelengths is not None:
        for i, wl in enumerate(emission_wavelengths):
            if i < len(channels) and _metadata_use_value(
                channels[i].get("emission_wavelength"), wl, overrule_metadata
            ):
                channels[i]["emission_wavelength"] = wl
    if excitation_wavelengths is not None:
        for i, wl in enumerate(excitation_wavelengths):
            if i < len(channels) and _metadata_use_value(
                channels[i].get("excitation_wavelength"), wl, overrule_metadata
            ):
                channels[i]["excitation_wavelength"] = wl
    meta["channels"] = channels[:n_channels]
    _apply_pinhole_airy_units(
        meta,
        _DEFAULT_PINHOLE_AIRY if pinhole_airy_units is None else pinhole_airy_units,
        overrule_metadata=overrule_metadata,
    )
    return meta


# ---------------------------------------------------------------------------
# Human-readable byte formatting
# ---------------------------------------------------------------------------
def _format_bytes(mb):
    """Format megabytes as a human-readable string."""
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _normalise_image(arr: np.ndarray) -> np.ndarray:
    """Normalise image to [0, 1] range."""
    img = np.asarray(arr, dtype=np.float32)
    lo = float(np.min(img))
    hi = float(np.max(img))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return (img - lo) / (hi - lo)


_METRIC_MAX_Z = 32
_METRIC_MAX_YX = 512
_METRIC_RADIUS_CACHE: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}


def _metric_stride_slice(size: int, limit: int) -> slice:
    """Return a deterministic stride slice capped to roughly limit samples."""
    if size <= limit:
        return slice(None)
    return slice(0, size, int(np.ceil(size / limit)))


def _metric_sample(arr: np.ndarray) -> np.ndarray:
    """Sample image data before expensive metric calculations."""
    data = np.asarray(arr)
    if data.ndim == 3:
        slices = (
            _metric_stride_slice(data.shape[0], _METRIC_MAX_Z),
            _metric_stride_slice(data.shape[1], _METRIC_MAX_YX),
            _metric_stride_slice(data.shape[2], _METRIC_MAX_YX),
        )
    elif data.ndim == 2:
        slices = (
            _metric_stride_slice(data.shape[0], _METRIC_MAX_YX),
            _metric_stride_slice(data.shape[1], _METRIC_MAX_YX),
        )
    else:
        slices = tuple(_metric_stride_slice(size, _METRIC_MAX_YX) for size in data.shape)
    return np.asarray(data[slices])


def _metric_sample_summary(arr: np.ndarray) -> str:
    """Return a compact description of metric sampling."""
    data = np.asarray(arr)
    sampled = _metric_sample(data)
    if data.shape == sampled.shape:
        return f"full shape {data.shape}"
    return f"full shape {data.shape} -> sampled {sampled.shape}"


def _metric_frequency_radius(shape: tuple[int, ...]) -> tuple[np.ndarray, float]:
    """Return cached frequency radius grid and max radius for a sampled shape."""
    cached = _METRIC_RADIUS_CACHE.get(shape)
    if cached is not None:
        return cached
    freq_axes = np.meshgrid(
        *[np.fft.fftfreq(n) for n in shape],
        indexing="ij",
    )
    radius = np.sqrt(sum(axis ** 2 for axis in freq_axes))
    max_radius = float(np.max(radius)) or 1.0
    cached = (radius, max_radius)
    _METRIC_RADIUS_CACHE[shape] = cached
    return cached


def _mean_or_zero(values: list[float]) -> float:
    """Return mean of values or 0.0 if empty."""
    return float(np.mean(values)) if values else 0.0


def _deconvolution_effect_metrics(arr: np.ndarray) -> dict[str, float]:
    """Compute no-reference metrics that better describe deconvolution effects."""
    img = _normalise_image(_metric_sample(arr))
    centered = img - float(np.mean(img))
    fft_power = np.abs(np.fft.fftn(centered)) ** 2
    total_power = float(np.sum(fft_power)) + 1e-12
    radius, max_radius = _metric_frequency_radius(img.shape)
    detail_energy = float(np.sum(fft_power[radius > 0.25 * max_radius]) / total_power)

    gradient_axes = tuple(i for i, size in enumerate(img.shape) if size > 1)
    if gradient_axes:
        grads = np.gradient(img, axis=gradient_axes)
        if isinstance(grads, np.ndarray):
            grads = [grads]
        edge_strength = float(np.mean(np.sqrt(sum(g ** 2 for g in grads))))
    else:
        edge_strength = 0.0

    p005 = float(np.percentile(img, 0.5))
    p95 = float(np.percentile(img, 95))
    p995 = float(np.percentile(img, 99.5))
    bright = img >= p95
    if np.any(bright):
        bright_power = np.abs(np.fft.fftn(centered * bright.astype(np.float32))) ** 2
        bright_total = float(np.sum(bright_power)) + 1e-12
        bright_detail_energy = float(np.sum(bright_power[radius > 0.25 * max_radius]) / bright_total)
    else:
        bright_detail_energy = 0.0
    flat = np.sort(img.ravel())
    total_intensity = float(np.sum(flat))
    if flat.size and total_intensity > 1e-12:
        index = np.arange(1, flat.size + 1, dtype=np.float64)
        signal_sparsity = float((2.0 * np.sum(index * flat)) / (flat.size * total_intensity) - (flat.size + 1.0) / flat.size)
    else:
        signal_sparsity = 0.0

    return {
        "detail_energy": detail_energy,
        "bright_detail_energy": bright_detail_energy,
        "edge_strength": edge_strength,
        "signal_sparsity": signal_sparsity,
        "robust_range": p995 - p005,
    }


def _quality_metrics(
    result_channels: list[np.ndarray],
) -> dict[str, float | int]:
    """Compute aggregate deconvolution-effect metrics from channels."""
    metric_values: dict[str, list[float]] = {
        "detail_energy": [],
        "bright_detail_energy": [],
        "edge_strength": [],
        "signal_sparsity": [],
        "robust_range": [],
    }

    for result in result_channels:
        metrics = _deconvolution_effect_metrics(result)
        for key in metric_values:
            metric_values[key].append(metrics[key])
    out: dict[str, float | int] = {"channels_compared": len(result_channels)}
    for key, values in metric_values.items():
        out[f"{key}_mean"] = _mean_or_zero(values)
    return out


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds for console output."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def _format_value(value, unit: str = "", digits: int = 4) -> str:
    """Format optional metadata values compactly."""
    if value is None:
        return "?"
    if isinstance(value, float):
        text = f"{value:.{digits}g}"
    else:
        text = str(value)
    return f"{text} {unit}".rstrip()


def _array_stats(arr: np.ndarray) -> dict[str, float | int | str | tuple[int, ...]]:
    """Return robust descriptive statistics for one image channel."""
    data = np.asarray(arr)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {
            "shape": data.shape,
            "dtype": str(data.dtype),
            "bytes_mb": data.nbytes / (1024 * 1024),
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p1": 0.0,
            "p50": 0.0,
            "p99": 0.0,
            "nonzero_percent": 0.0,
        }
    return {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "bytes_mb": data.nbytes / (1024 * 1024),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p1": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "nonzero_percent": float(np.count_nonzero(data) / data.size * 100) if data.size else 0.0,
    }


def _print_runtime_environment() -> None:
    """Print platform and dependency details relevant to performance."""
    print("\nRuntime environment")
    print(f"  Timestamp    : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Platform     : {platform.platform()}")
    print(f"  Python       : {platform.python_version()} ({sys.executable})")
    print(f"  NumPy        : {np.__version__}")
    try:
        import torch
        print(f"  PyTorch      : {torch.__version__}")
        print(f"  CUDA avail.  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version : {torch.version.cuda or '?'}")
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                print(
                    f"  GPU {idx}       : {props.name} "
                    f"({_format_bytes(props.total_memory / (1024 * 1024))})"
                )
    except Exception as exc:
        print(f"  PyTorch      : unavailable ({exc})")
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"  CPU cores    : {psutil.cpu_count(logical=False) or '?'} physical, "
              f"{psutil.cpu_count(logical=True) or '?'} logical")
        print(f"  System RAM   : {_format_bytes(vm.total / (1024 * 1024))}")
    except Exception:
        pass


def _print_image_details(filename: str, img_path: Path, meta: dict, images: list[np.ndarray]) -> None:
    """Print detailed source image metadata and channel statistics."""
    print("\n  Loaded image")
    print(f"    File       : {filename}")
    print(f"    Path       : {img_path}")
    if img_path.exists() and img_path.is_file():
        print(f"    File size  : {_format_bytes(img_path.stat().st_size / (1024 * 1024))}")
    print(f"    Channels   : {len(images)}")
    print(f"    Dimensions : X={meta.get('size_x', '?')}  Y={meta.get('size_y', '?')}  "
          f"Z={meta.get('size_z', '?')}  C={meta.get('size_c', len(images))}  "
          f"T={meta.get('size_t', '?')}")
    print(f"    Pixel size : XY={_format_value(meta.get('pixel_size_x'), 'um')}  "
          f"Z={_format_value(meta.get('pixel_size_z'), 'um')}")
    print(f"    Microscope : {meta.get('microscope_type', '?')}")
    print(f"    Objective  : NA={_format_value(meta.get('na'))}  "
          f"Mag={_format_value(meta.get('magnification'), 'x')}  "
          f"Immersion={meta.get('immersion', '?')}")
    print(f"    RI         : immersion={_format_value(meta.get('refractive_index'))}  "
          f"sample={_format_value(meta.get('sample_refractive_index'))}")

    channels = meta.get("channels", [])
    names = meta.get("channel_names") or []
    for i, img in enumerate(images):
        stats = _array_stats(img)
        ch_meta = channels[i] if i < len(channels) else {}
        name = names[i] if i < len(names) else f"Ch{i}"
        print(f"    Ch{i} {name}: shape={stats['shape']} dtype={stats['dtype']} "
              f"data={_format_bytes(stats['bytes_mb'])}")
        print(f"      wavelengths: em={_format_value(ch_meta.get('emission_wavelength'), 'nm')}  "
              f"ex={_format_value(ch_meta.get('excitation_wavelength'), 'nm')}  "
              f"mode={ch_meta.get('acquisition_mode', '?')}")
        print(f"      pinhole    : size={_format_value(ch_meta.get('pinhole_size'), ch_meta.get('pinhole_size_unit') or '')}  "
              f"effective={_format_value(ch_meta.get('pinhole_airy_units'), 'AU')}")
        print(f"      intensity  : min={stats['min']:.4g} p1={stats['p1']:.4g} "
              f"median={stats['p50']:.4g} mean={stats['mean']:.4g} "
              f"p99={stats['p99']:.4g} max={stats['max']:.4g} "
              f"nonzero={stats['nonzero_percent']:.1f}%")


def _print_psf_details(psfs: list[np.ndarray]) -> None:
    """Print PSF shape and normalization details."""
    if not psfs:
        return
    print("\n  Generated PSFs")
    for i, psf in enumerate(psfs):
        arr = np.asarray(psf)
        print(f"    Ch{i}: shape={arr.shape} dtype={arr.dtype} "
              f"sum={float(arr.sum()):.6g} peak={float(arr.max()):.6g}")


def _print_quality_comparison(source_channels: list[np.ndarray], result_channels: list[np.ndarray]) -> None:
    """Print source/result metrics that describe deconvolution effects."""
    def _change(src: float, res: float) -> str:
        if abs(src) > 1e-12:
            return f"{res / src:.2f}x"
        return f"{res - src:.4g}"

    metric_order = (
        ("detail_energy", "Detail energy"),
        ("bright_detail_energy", "Bright detail energy"),
        ("edge_strength", "Edge strength"),
        ("signal_sparsity", "Signal sparsity"),
        ("robust_range", "Robust range"),
    )
    print("\n  Image metrics")
    n_channels = min(len(source_channels), len(result_channels))
    if n_channels:
        print(f"    Metrics sample: {_metric_sample_summary(source_channels[0])} "
              f"(caps: Z<=32, Y/X<=512)")
    for ch_idx in range(n_channels):
        source_q = _deconvolution_effect_metrics(source_channels[ch_idx])
        result_q = _deconvolution_effect_metrics(result_channels[ch_idx])
        print(f"    Channel {ch_idx}")
        print("      Metric                 Source        Result      Change")
        print("      -------------------------------------------------------")
        for key, label in metric_order:
            src = float(source_q.get(key, 0.0))
            res = float(result_q.get(key, 0.0))
            print(f"      {label:<18} {src:>11.4g} {res:>13.4g} { _change(src, res):>10}")
    if n_channels > 1:
        source_mean = _quality_metrics(source_channels[:n_channels])
        result_mean = _quality_metrics(result_channels[:n_channels])
        print("    Mean across channels")
        print("      Metric                 Source        Result      Change")
        print("      -------------------------------------------------------")
        for key, label in metric_order:
            mean_key = f"{key}_mean"
            src = float(source_mean.get(mean_key, 0.0))
            res = float(result_mean.get(mean_key, 0.0))
            print(f"      {label:<18} {src:>11.4g} {res:>13.4g} { _change(src, res):>10}")


def _print_resource_metrics(metrics: dict[str, float]) -> None:
    """Print resource metrics gathered by _MetricsMonitor."""
    gpu_delta = (
        metrics.get("torch_gpu_delta_mb", 0.0)
        if metrics.get("torch_gpu_delta_mb", 0.0) > 0
        else metrics.get("gpu_mem_delta_peak_mb", 0.0)
    )
    print("\n  Resource metrics")
    print(f"    Deconv time : {_format_duration(metrics.get('time_s', 0.0))}")
    print(f"    CPU         : avg={metrics.get('cpu_percent_avg', 0.0):.0f}%  "
          f"peak={metrics.get('cpu_percent_peak', 0.0):.0f}%")
    print(f"    RAM         : peak={_format_bytes(metrics.get('ram_peak_mb', 0.0))}  "
          f"delta={_format_bytes(metrics.get('ram_delta_peak_mb', 0.0))}  "
          f"avg={_format_bytes(metrics.get('ram_avg_mb', 0.0))}")
    if metrics.get("gpu_total_mb", 0.0) > 0 or metrics.get("gpu_mem_peak_mb", 0.0) > 0:
        if metrics.get("gpu_util_available", 0.0) > 0:
            print(f"    GPU         : util avg={metrics.get('gpu_util_avg', 0.0):.0f}%  "
                  f"peak={metrics.get('gpu_util_peak', 0.0):.0f}%")
        else:
            print("    GPU         : util n/a (NVML unavailable)")
        print(f"    VRAM        : peak={_format_bytes(metrics.get('gpu_mem_peak_mb', 0.0))}  "
              f"delta={_format_bytes(gpu_delta)}  "
              f"torch peak={_format_bytes(metrics.get('torch_gpu_peak_mb', 0.0))}")


# ---------------------------------------------------------------------------
# OME-Zarr HCS plate helpers
# ---------------------------------------------------------------------------

def _is_hcs_plate(zarr_path: Path) -> bool:
    """Return True if *zarr_path* is an OME-Zarr HCS plate (has 'plate' in root attrs)."""
    import zarr
    try:
        store = zarr.open(str(zarr_path), mode="r")
        return "plate" in store.attrs
    except Exception:
        return False


def _get_zarr_plate_info(zarr_path: Path) -> dict:
    """Parse an HCS plate zarr and return plate metadata + list of (row, col, field) tuples."""
    import zarr
    store = zarr.open(str(zarr_path), mode="r")
    plate_attrs = dict(store.attrs)
    plate_meta = plate_attrs["plate"]

    wells_and_fields = []
    for well_entry in plate_meta["wells"]:
        well_path = well_entry["path"]  # e.g. "A/1"
        parts = well_path.split("/")
        row, col = parts[0], parts[1]
        well_group = store[well_path]
        well_attrs = dict(well_group.attrs)
        if "well" in well_attrs and "images" in well_attrs["well"]:
            for img_entry in well_attrs["well"]["images"]:
                field = img_entry["path"]  # e.g. "0"
                wells_and_fields.append((row, col, field))
        else:
            # Fallback: enumerate numeric subdirectories
            for key in sorted(well_group.keys()):
                if key.isdigit():
                    wells_and_fields.append((row, col, key))

    return {
        "plate_attrs": plate_attrs,
        "plate_meta": plate_meta,
        "wells_and_fields": wells_and_fields,
    }


def _ensure_channel_metadata(meta: dict, n_channels: int) -> None:
    """Keep per-channel metadata aligned with the actual image channel count."""
    channels = [
        dict(ch) if isinstance(ch, dict) else {}
        for ch in (meta.get("channels") or [])
    ]
    if len(channels) < n_channels:
        channels.extend({} for _ in range(n_channels - len(channels)))
    meta["channels"] = channels[:n_channels]

    channel_names = list(meta.get("channel_names") or [])
    if len(channel_names) < n_channels:
        channel_names.extend(
            f"Ch{i}" for i in range(len(channel_names), n_channels)
        )
    meta["channel_names"] = channel_names[:n_channels]


def _load_zarr_field(
    zarr_path: Path,
    row: str,
    col: str,
    field: str,
    *,
    na=None,
    refractive_index=None,
    sample_refractive_index=None,
    microscope_type=None,
    pixel_size_xy=None,
    pixel_size_z=None,
    emission_wavelengths=None,
    excitation_wavelengths=None,
    pinhole_airy_units=None,
    overrule_metadata: bool = True,
) -> dict:
    """Load a single field from an HCS plate zarr.

    Returns a dict compatible with load_image() output:
        {'images': [np.ndarray per channel], 'metadata': dict}
    """
    import zarr

    store = zarr.open(str(zarr_path), mode="r")
    field_group = store[f"{row}/{col}/{field}"]
    field_attrs = dict(field_group.attrs)

    # Read level 0 (highest resolution)
    data = np.asarray(field_group["0"][:])

    # Squeeze leading dimensions until we have at most 5D (T, C, Z, Y, X)
    while data.ndim > 5 and data.shape[0] == 1:
        data = data[0]

    # Expected shapes: (T, C, Z, Y, X) or (C, Z, Y, X) or (C, Y, X)
    if data.ndim == 5:
        data = data[0]  # Take T=0 -> (C, Z, Y, X)
    if data.ndim == 4:
        n_c, n_z, n_y, n_x = data.shape
    elif data.ndim == 3:
        n_c, n_y, n_x = data.shape
        n_z = 1
    else:
        raise ValueError(f"Unexpected field data shape: {data.shape}")

    # Split into per-channel arrays
    images = []
    for c in range(n_c):
        if data.ndim == 4:
            ch_data = data[c]  # (Z, Y, X)
            if ch_data.shape[0] == 1:
                ch_data = ch_data[0]  # Squeeze Z=1 -> (Y, X)
        else:
            ch_data = data[c]  # (Y, X)
        images.append(np.asarray(ch_data, dtype=np.float32))

    # Extract metadata from field attrs
    meta = {}
    meta["n_channels"] = n_c
    meta["size_x"] = n_x
    meta["size_y"] = n_y
    meta["size_z"] = n_z

    # Pixel sizes from multiscales coordinate transforms
    multiscales = field_attrs.get("multiscales", [])
    if multiscales:
        ms = multiscales[0]
        datasets = ms.get("datasets", [])
        if datasets:
            transforms = datasets[0].get("coordinateTransformations", [])
            for t in transforms:
                if t.get("type") == "scale":
                    scale = t["scale"]
                    # Scale array: for 5D TCZYX -> indices [0]=T, [1]=C, [2]=Z, [3]=Y, [4]=X
                    if len(scale) == 5:
                        meta["pixel_size_z"] = scale[2]
                        meta["pixel_size_y"] = scale[3]
                        meta["pixel_size_x"] = scale[4]
                    elif len(scale) == 4:
                        meta["pixel_size_z"] = scale[1]
                        meta["pixel_size_y"] = scale[2]
                        meta["pixel_size_x"] = scale[3]
                    elif len(scale) == 3:
                        meta["pixel_size_y"] = scale[1]
                        meta["pixel_size_x"] = scale[2]
                    break

    # Channel info from omero metadata
    omero = field_attrs.get("omero", {})
    omero_channels = omero.get("channels", [])
    ch_info = []
    channel_names = []
    for i, och in enumerate(omero_channels):
        color = och.get("color")
        if isinstance(color, str) and len(color.strip().lstrip("#")) == 6:
            text = color.strip().lstrip("#")
            try:
                color = tuple(int(text[j:j + 2], 16) for j in (0, 2, 4))
            except ValueError:
                color = None
        window = och.get("window") or {}
        info = {
            "emission_wavelength": None,
            "excitation_wavelength": None,
            "acquisition_mode": None,
            "pinhole_size": None,
            "color": color,
            "active": bool(och.get("active", True)),
            "window_start": window.get("start"),
            "window_end": window.get("end"),
        }
        channel_names.append(och.get("label", f"Ch{i}"))
        ch_info.append(info)
    meta["channels"] = ch_info
    meta["channel_names"] = channel_names
    _ensure_channel_metadata(meta, n_c)

    def _use_value(current, value) -> bool:
        return value is not None and (overrule_metadata or current is None)

    # Apply user metadata values. With overrule_metadata=False, these are
    # fallbacks only; existing image metadata is preserved.
    if _use_value(meta.get("na"), na):
        meta["na"] = na
    if _use_value(meta.get("refractive_index"), refractive_index):
        meta["refractive_index"] = refractive_index
    if _use_value(meta.get("microscope_type"), microscope_type):
        meta["microscope_type"] = microscope_type
    if _use_value(meta.get("pixel_size_x"), pixel_size_xy):
        meta["pixel_size_x"] = pixel_size_xy
    if _use_value(meta.get("pixel_size_y"), pixel_size_xy):
        meta["pixel_size_y"] = pixel_size_xy
    if _use_value(meta.get("pixel_size_z"), pixel_size_z):
        meta["pixel_size_z"] = pixel_size_z
    if emission_wavelengths is not None:
        for i, wl in enumerate(emission_wavelengths):
            if i < len(meta["channels"]):
                ch = meta["channels"][i]
                if _use_value(ch.get("emission_wavelength"), wl):
                    ch["emission_wavelength"] = wl
    if excitation_wavelengths is not None:
        for i, wl in enumerate(excitation_wavelengths):
            if i < len(meta["channels"]):
                ch = meta["channels"][i]
                if _use_value(ch.get("excitation_wavelength"), wl):
                    ch["excitation_wavelength"] = wl

    _defaulted = set()
    _defaults = {
        "na": _DEFAULT_NA,
        "refractive_index": 1.515,
        "microscope_type": "widefield",
        "pixel_size_x": _DEFAULT_PIXEL_SIZE_XY_NM / 1000.0,
        "pixel_size_y": _DEFAULT_PIXEL_SIZE_XY_NM / 1000.0,
        "pixel_size_z": _DEFAULT_PIXEL_SIZE_Z_NM / 1000.0,
    }
    for k, v in _defaults.items():
        if meta.get(k) is None:
            meta[k] = v
            _defaulted.add(k)
    meta["_defaulted_keys"] = _defaulted

    # Ensure emission wavelengths have a default
    _em_defaulted = False
    for ch in meta["channels"]:
        if ch.get("emission_wavelength") is None:
            ch["emission_wavelength"] = 520.0
            _em_defaulted = True
    if _em_defaulted:
        _defaulted.add("emission_wavelength")

    if _apply_pinhole_airy_units(
        meta,
        _DEFAULT_PINHOLE_AIRY if pinhole_airy_units is None else pinhole_airy_units,
        overrule_metadata=overrule_metadata,
    ):
        _defaulted.discard("pinhole_airy_units")
    else:
        _defaulted.add("pinhole_airy_units")

    if overrule_metadata and sample_refractive_index is not None:
        meta["sample_refractive_index"] = sample_refractive_index
    elif meta.get("sample_refractive_index") is None:
        meta["sample_refractive_index"] = (
            sample_refractive_index
            if sample_refractive_index is not None
            else _SAMPLE_RI_DEFAULT
        )

    return {"images": images, "metadata": meta}


def _init_output_plate_zarr(
    out_zarr_path: Path,
    source_plate_attrs: dict,
    wells_and_fields: list,
) -> None:
    """Create the output HCS plate zarr skeleton (root + well groups)."""
    import zarr

    store = zarr.open(str(out_zarr_path), mode="w", zarr_format=OUTPUT_ZARR_FORMAT)

    # Copy and update plate metadata for OME-Zarr 0.5
    plate_meta = dict(source_plate_attrs.get("plate", {}))
    stem = out_zarr_path.stem
    if stem.endswith(".ome"):
        stem = stem[:-4]
    plate_meta["name"] = stem
    plate_meta["version"] = "0.5"
    store.attrs["plate"] = plate_meta

    # Copy creator if present
    if "_creator" in source_plate_attrs:
        store.attrs["_creator"] = source_plate_attrs["_creator"]

    # Create well groups
    wells_seen = set()
    for row, col, field in wells_and_fields:
        well_path = f"{row}/{col}"
        if well_path not in wells_seen:
            wells_seen.add(well_path)
            well_group = store.require_group(well_path)

            # Collect all fields for this well
            well_fields = [
                f for r, c, f in wells_and_fields
                if r == row and c == col
            ]
            well_group.attrs["well"] = {
                "images": [{"path": f} for f in sorted(well_fields)],
                "version": "0.5",
            }


def _downsample_2x_xy(data: np.ndarray) -> np.ndarray:
    """Downsample by 2x in XY using block mean. Handles odd dimensions."""
    if data.ndim == 5:
        # (T, C, Z, Y, X)
        t, c, z, y, x = data.shape
        y2 = (y // 2) * 2
        x2 = (x // 2) * 2
        cropped = data[:, :, :, :y2, :x2]
        return cropped.reshape(t, c, z, y2 // 2, 2, x2 // 2, 2).mean(axis=(4, 6))
    elif data.ndim == 4:
        # (C, Z, Y, X)
        c, z, y, x = data.shape
        y2 = (y // 2) * 2
        x2 = (x // 2) * 2
        cropped = data[:, :, :y2, :x2]
        return cropped.reshape(c, z, y2 // 2, 2, x2 // 2, 2).mean(axis=(3, 5))
    elif data.ndim == 3:
        # (C, Y, X)
        c, y, x = data.shape
        y2 = (y // 2) * 2
        x2 = (x // 2) * 2
        cropped = data[:, :y2, :x2]
        return cropped.reshape(c, y2 // 2, 2, x2 // 2, 2).mean(axis=(2, 4))
    else:
        raise ValueError(f"Cannot downsample array with {data.ndim} dimensions")


def _write_zarr_field(
    result_channels: list,
    metadata: dict,
    out_zarr_path: Path,
    row: str,
    col: str,
    field: str,
    orig_field_attrs: dict,
) -> None:
    """Write deconvolved channels as a field in the output plate zarr."""
    import zarr

    store = zarr.open(str(out_zarr_path), mode="a", zarr_format=OUTPUT_ZARR_FORMAT)
    field_group = store.require_group(f"{row}/{col}/{field}")

    # Stack channels: determine if 3D or 2D
    is_3d = result_channels[0].ndim == 3
    if is_3d:
        stack = np.stack(result_channels, axis=0)  # (C, Z, Y, X)
        stack = stack[np.newaxis, ...]  # (1, C, Z, Y, X) = TCZYX
    else:
        stack = np.stack(result_channels, axis=0)  # (C, Y, X)
        stack = stack[np.newaxis, :, np.newaxis, :, :]  # (1, C, 1, Y, X) = TCZYX

    stack = stack.astype(np.float32)
    _, n_c, n_z, n_y, n_x = stack.shape

    # Compressor and array kwargs differ between zarr v2 and v3
    if OUTPUT_ZARR_FORMAT == 2:
        from numcodecs import Blosc
        _compressor_kw = {
            "compressor": Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
            "chunk_key_encoding": {"name": "v2", "configuration": {"separator": "/"}},
        }
    else:
        from zarr.codecs import BloscCodec
        _compressor_kw = {"compressors": [BloscCodec(cname="zstd", clevel=5)]}

    # Determine number of pyramid levels from source multiscales
    src_ms = orig_field_attrs.get("multiscales", [{}])[0]
    src_datasets = src_ms.get("datasets", [{"path": "0"}])
    n_levels = len(src_datasets)

    # Pixel sizes
    px_x = metadata.get("pixel_size_x", 0.1)
    px_y = metadata.get("pixel_size_y", 0.1)
    px_z = metadata.get("pixel_size_z", 0.3)

    # Write all pyramid levels
    datasets = []
    current = stack
    for lvl in range(n_levels):
        if lvl > 0:
            current = _downsample_2x_xy(current).astype(np.float32)
        cy = min(512, current.shape[-2])
        cx = min(512, current.shape[-1])
        field_group.create_array(
            str(lvl),
            data=current,
            chunks=(1, 1, n_z, cy, cx),
            overwrite=True,
            **_compressor_kw,
        )
        scale_factor = 2 ** lvl
        datasets.append({
            "path": str(lvl),
            "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, px_z, px_y * scale_factor, px_x * scale_factor]},
            ],
        })

    # Build OME-Zarr 0.5 multiscales metadata
    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    multiscales = [{
        "version": "0.5",
        "axes": axes,
        "datasets": datasets,
        "name": f"{row}/{col}/{field}",
    }]

    field_group.attrs["multiscales"] = multiscales

    # Copy omero metadata from source
    omero = orig_field_attrs.get("omero")
    if omero is not None:
        field_group.attrs["omero"] = omero


def _projection_output_suffix(projection: str) -> str:
    """Return the filename suffix for an actual Z-projected output."""
    projection = str(projection).strip().lower()
    if projection == "mip":
        return "_mip-proj"
    if projection == "sum":
        return "_sum-proj"
    return ""


def _run_plate_zarr(
    zarr_path: Path,
    out_path: str,
    *,
    niter_list: list,
    method: str,
    device,
    na,
    refractive_index,
    sample_ri: float,
    microscope_type,
    emission_wavelengths,
    excitation_wavelengths,
    pinhole_airy_units,
    pixel_size_xy,
    pixel_size_z,
    overrule_metadata: bool,
    tv_lambda: float,
    damping,
    sparse_hessian_weight: float,
    sparse_hessian_reg: float,
    background,
    offset,
    prefilter_sigma: float,
    start: str,
    convergence: str,
    rel_threshold: float,
    check_every: int,
    ri_coverslip,
    ri_coverslip_design,
    ri_immersion_design,
    t_g: float,
    t_g0: float,
    t_i0: float,
    z_p: float,
    two_d_mode: str = "auto",
    two_d_wf_aggressiveness: str = "balanced",
    two_d_wf_bg_radius_um: float = 0.5,
    two_d_wf_bg_scale: float = 1.0,
    projection: str = "none",
) -> None:
    """Process an HCS plate zarr: deconvolve every field and write output plate zarr."""
    import zarr

    plate_info = _get_zarr_plate_info(zarr_path)
    plate_attrs = plate_info["plate_attrs"]
    wells_and_fields = plate_info["wells_and_fields"]

    plate_stem = zarr_path.stem
    if plate_stem.endswith(".ome"):
        plate_stem = plate_stem[:-4]
    out_zarr_name = f"{plate_stem}_decon.ome.zarr"
    out_zarr_path = Path(out_path) / out_zarr_name

    print(f"\n  HCS Plate detected: {zarr_path.name}")
    print(f"  Wells/fields: {len(wells_and_fields)}")
    print(f"  Output: {out_zarr_name}")

    _init_output_plate_zarr(out_zarr_path, plate_attrs, wells_and_fields)

    # Read source zarr for field attrs
    source_store = zarr.open(str(zarr_path), mode="r")

    total = len(wells_and_fields)
    t_plate_start = time.time()

    for idx, (row, col, field) in enumerate(wells_and_fields):
        field_id = f"{row}/{col}/{field}"
        print(f"\n  [{idx + 1}/{total}] Processing field {field_id}")

        t0 = time.time()

        # Load field data
        data = _load_zarr_field(
            zarr_path, row, col, field,
            na=na,
            refractive_index=refractive_index,
            sample_refractive_index=sample_ri,
            microscope_type=microscope_type,
            pixel_size_xy=pixel_size_xy,
            pixel_size_z=pixel_size_z,
            emission_wavelengths=emission_wavelengths,
            excitation_wavelengths=excitation_wavelengths,
            pinhole_airy_units=pinhole_airy_units,
            overrule_metadata=overrule_metadata,
        )
        images = data["images"]
        meta = data["metadata"]

        print(f"    Channels: {len(images)}, shape: {images[0].shape}")

        # Deconvolve each channel
        result_channels = []
        psfs = []
        for ch_idx in range(len(images)):
            psf = generate_psf(
                meta, channel_idx=ch_idx,
                ri_coverslip=ri_coverslip,
                ri_coverslip_design=ri_coverslip_design,
                ri_immersion_design=ri_immersion_design,
                t_g=t_g, t_g0=t_g0, t_i0=t_i0, z_p=z_p,
            )
            psfs.append(psf)

            img = images[ch_idx]
            # Match PSF dimensionality
            keep_hidden_2d_psf = (
                img.ndim == 2
                and psf.ndim == 3
                and meta.get("microscope_type", "widefield") == "widefield"
                and method in ("ci_rl", "ci_rl_tv")
                and str(two_d_mode).strip().lower() == "auto"
            )
            if img.ndim == 2 and psf.ndim == 3 and not keep_hidden_2d_psf:
                psf = psf[psf.shape[0] // 2]
            elif img.ndim == 3 and psf.ndim == 2:
                psf = psf[np.newaxis, :, :]

            # Per-channel iteration count
            if isinstance(niter_list, list) and len(niter_list) > 1:
                ch_niter = niter_list[ch_idx] if ch_idx < len(niter_list) else niter_list[-1]
            else:
                ch_niter = niter_list[0] if isinstance(niter_list, list) else niter_list

            result = deconvolve(
                img, psf, method=method,
                niter=ch_niter, background=background, damping=damping, offset=offset,
                prefilter_sigma=prefilter_sigma, start=start,
                convergence=convergence, rel_threshold=rel_threshold,
                check_every=check_every, device=device,
                tv_lambda=tv_lambda if method == "ci_rl_tv" else 0.0,
                sparse_hessian_weight=sparse_hessian_weight,
                sparse_hessian_reg=sparse_hessian_reg,
                pixel_size_xy=meta.get("pixel_size_x"),
                pixel_size_z=meta.get("pixel_size_z"),
                microscope_type=meta.get("microscope_type", "widefield"),
                two_d_mode=two_d_mode,
                two_d_wf_aggressiveness=two_d_wf_aggressiveness,
                two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
                two_d_wf_bg_scale=two_d_wf_bg_scale,
            )
            result_channels.append(result)

        # Get original field attrs for metadata copying
        field_group = source_store[f"{row}/{col}/{field}"]
        orig_field_attrs = dict(field_group.attrs)

        # Write to output zarr
        _write_zarr_field(
            result_channels, meta, out_zarr_path,
            row, col, field, orig_field_attrs,
        )

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    total_time = time.time() - t_plate_start
    print(f"\n  Plate processing complete: {total} fields in {total_time:.1f}s")
    print(f"  Output: {out_zarr_path}")


def _run_streaming_regular_image(
    img_path: Path,
    out_path: str,
    *,
    stem: str,
    niter_list: list,
    method: str,
    device,
    na,
    refractive_index,
    sample_ri: float,
    microscope_type,
    emission_wavelengths,
    excitation_wavelengths,
    pinhole_airy_units,
    pixel_size_xy,
    pixel_size_z,
    overrule_metadata: bool,
    tv_lambda: float,
    damping,
    sparse_hessian_weight: float,
    sparse_hessian_reg: float,
    background,
    offset,
    prefilter_sigma: float,
    start: str,
    convergence: str,
    rel_threshold: float,
    check_every: int,
    ri_coverslip,
    ri_coverslip_design,
    ri_immersion_design,
    t_g: float,
    t_g0: float,
    t_i0: float,
    z_p: float,
    two_d_mode: str,
    two_d_wf_aggressiveness: str,
    two_d_wf_bg_radius_um: float,
    two_d_wf_bg_scale: float,
    tile_limits: tuple[int, int],
    scene=None,
    hcs_field=None,
) -> Path:
    """Stream a regular image to OME-Zarr, reading only halo-extended tiles."""
    if method == "ci_rl_dl":
        raise ValueError("Streaming ci_rl_dl is not enabled yet; use ci_rl/ci_rl_tv or eager ci_rl_dl.")

    source = open_region_source(img_path, scene=scene, hcs_field=hcs_field)
    source.metadata = _apply_cli_metadata_to_source(
        source.metadata,
        na=na,
        refractive_index=refractive_index,
        sample_ri=sample_ri,
        microscope_type=microscope_type,
        emission_wavelengths=emission_wavelengths,
        excitation_wavelengths=excitation_wavelengths,
        pinhole_airy_units=pinhole_airy_units,
        pixel_size_xy=pixel_size_xy,
        pixel_size_z=pixel_size_z,
        overrule_metadata=overrule_metadata,
    )
    tile_xy = int(tile_limits[0])
    if tile_xy <= 0:
        tile_xy = suggest_streaming_tile_size(
            source.shape,
            psf_xy_est=65,
            method=method,
            device=device,
        )
    tile_limits = (tile_xy, int(tile_limits[1]))
    out_zarr = Path(out_path) / f"{stem}_decon.ome.zarr"
    sink = ZarrPyramidSink(
        out_zarr,
        shape=source.shape,
        metadata=source.metadata,
        zarr_format=OUTPUT_ZARR_FORMAT,
        resume=True,
    )

    psf_cache: dict[int, np.ndarray] = {}

    def _psf_for_channel(ch_idx: int) -> np.ndarray:
        cached = psf_cache.get(ch_idx)
        if cached is not None:
            return cached
        psf = generate_psf(
            source.metadata,
            channel_idx=ch_idx,
            ri_coverslip=ri_coverslip,
            ri_coverslip_design=ri_coverslip_design,
            ri_immersion_design=ri_immersion_design,
            t_g=t_g,
            t_g0=t_g0,
            t_i0=t_i0,
            z_p=z_p,
            two_d_mode=two_d_mode if method in ("ci_rl", "ci_rl_tv") else "legacy_2d",
        )
        psf_cache[ch_idx] = psf
        return psf

    def _deconvolve_tile(tile_img: np.ndarray, psf: np.ndarray, ch_idx: int) -> np.ndarray:
        effective_psf = psf
        keep_hidden_2d_psf = (
            tile_img.ndim == 2
            and psf.ndim == 3
            and source.metadata.get("microscope_type", "widefield") == "widefield"
            and method in ("ci_rl", "ci_rl_tv")
            and str(two_d_mode).strip().lower() == "auto"
        )
        if tile_img.ndim == 2 and effective_psf.ndim == 3 and not keep_hidden_2d_psf:
            effective_psf = effective_psf[effective_psf.shape[0] // 2]
        elif tile_img.ndim == 3 and effective_psf.ndim == 2:
            effective_psf = effective_psf[np.newaxis, :, :]
        if isinstance(niter_list, list) and len(niter_list) > 1:
            ch_niter = niter_list[ch_idx] if ch_idx < len(niter_list) else niter_list[-1]
        else:
            ch_niter = niter_list[0] if isinstance(niter_list, list) else niter_list
        return deconvolve(
            tile_img,
            effective_psf,
            method=method,
            niter=ch_niter,
            background=background,
            damping=damping,
            offset=offset,
            prefilter_sigma=prefilter_sigma,
            start=start,
            convergence=convergence,
            rel_threshold=rel_threshold,
            check_every=check_every,
            device=device,
            tv_lambda=tv_lambda if method == "ci_rl_tv" else 0.0,
            sparse_hessian_weight=sparse_hessian_weight,
            sparse_hessian_reg=sparse_hessian_reg,
            pixel_size_xy=source.metadata.get("pixel_size_x"),
            pixel_size_z=source.metadata.get("pixel_size_z"),
            microscope_type=source.metadata.get("microscope_type", "widefield"),
            two_d_mode=two_d_mode,
            two_d_wf_aggressiveness=two_d_wf_aggressiveness,
            two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
            two_d_wf_bg_scale=two_d_wf_bg_scale,
        )

    def _progress(payload: dict) -> None:
        event = payload.get("event")
        if event == "tile_start":
            done = int(payload.get("done", 0))
            total = int(payload.get("total", 0))
            print(
                f"    Tile {done + 1}/{total}: "
                f"T={payload.get('timepoint')} C={payload.get('channel')} "
                f"core={payload.get('core')}"
            )
        elif event == "pyramid_start":
            print("    Building OME-Zarr pyramid levels...")

    print("\n  Streaming deconvolution")
    print(f"    Source     : {source.source_id}")
    print(f"    Shape      : T={source.shape[0]} C={source.shape[1]} Z={source.shape[2]} Y={source.shape[3]} X={source.shape[4]}")
    print(f"    Tile limits: XY={tile_limits[0]} px  Z={tile_limits[1]} slices (Z streaming reserved)")
    print(f"    Output     : {out_zarr.name}")
    t0 = time.time()
    summary = deconvolve_streaming(
        source,
        sink,
        psf_for_channel=_psf_for_channel,
        deconvolve_tile=_deconvolve_tile,
        tile_yx=(tile_limits[0], tile_limits[0]),
        progress=_progress,
        resume=True,
        build_pyramids=True,
    )
    provenance = save_streaming_provenance(
        out_zarr.with_suffix(out_zarr.suffix + ".provenance.json"),
        source=source,
        sink=sink,
        params={
            "method": method,
            "iterations": niter_list,
            "tile_limits": tile_limits,
            "convergence": convergence,
            "rel_threshold": rel_threshold,
            "background": background,
            "offset": offset,
            "prefilter_sigma": prefilter_sigma,
            "start": start,
        },
        summary=summary,
    )
    print(f"    Completed  : {summary['tiles_completed']}/{summary['tiles_total']} tile writes")
    print(f"    Time       : {_format_duration(time.time() - t0)}")
    print(f"    Provenance : {provenance.name}")
    return out_zarr


def main(argv):
    with BilayersJob.from_cli(argv) as bj:
        parameters = getattr(bj, "parameters", SimpleNamespace())

        # Extract parameters with defaults from descriptor.json
        iter_raw = str(getattr(parameters, "iterations", "40")).strip()
        niter_list = [max(1, int(s.strip())) for s in iter_raw.split(",") if s.strip()]
        if not niter_list:
            niter_list = [40]
        method = getattr(parameters, "method", "ci_rl")
        device_param = getattr(parameters, "device", "auto")
        device = None if device_param in (None, "auto") else device_param

        # PSF metadata parameters. By default image metadata wins; these values
        # are fallbacks for missing metadata. With overrule enabled they replace
        # any image metadata.
        overrule_metadata = _to_bool(getattr(parameters, "overrule_image_metadata", False))
        na_value = _parse_float_or_default(getattr(parameters, "na", _DEFAULT_NA), _DEFAULT_NA)
        ri_raw = str(getattr(parameters, "refractive_index", _DEFAULT_IMMERSION_RI_CHOICE))
        ri_value = _parse_ri_choice(ri_raw, _IMMERSION_RI) or 1.515
        sample_ri_raw = str(getattr(parameters, "sample_ri", _DEFAULT_SAMPLE_RI_CHOICE))
        sample_ri_value = _parse_ri_choice(sample_ri_raw, _SAMPLE_RI) or _SAMPLE_RI_DEFAULT
        micro_value = str(getattr(parameters, "microscope_type", _DEFAULT_MICROSCOPE_TYPE)).strip().lower()
        if micro_value == "auto":
            micro_value = _DEFAULT_MICROSCOPE_TYPE
        em_raw = str(getattr(parameters, "emission_wl", _DEFAULT_EMISSION_WL)).strip()
        em_value = _parse_float_list_or_default(em_raw, _DEFAULT_EMISSION_WL)
        ex_raw = str(getattr(parameters, "excitation_wl", _DEFAULT_EXCITATION_WL)).strip()
        ex_value = _parse_float_list_or_default(ex_raw, _DEFAULT_EXCITATION_WL)
        pinhole_airy = _parse_float_list_or_default(
            getattr(parameters, "pinhole_airy", str(_DEFAULT_PINHOLE_AIRY)),
            str(_DEFAULT_PINHOLE_AIRY),
        )

        # Deconvolution parameters
        tv_lambda = float(getattr(parameters, "tv_lambda", 0.0001))
        damping_raw = str(getattr(parameters, "damping", "none")).strip().lower()
        if damping_raw in ("none", "0", "0.0"):
            damping = 0.0
        elif damping_raw == "auto":
            damping = "auto"
        else:
            damping = float(damping_raw)
        bg_raw = str(getattr(parameters, "background", "auto")).strip()
        background = bg_raw if bg_raw.lower() == "auto" else float(bg_raw)
        offset_raw = str(getattr(parameters, "offset", "auto")).strip().lower()
        if offset_raw in ("none", "0", "0.0"):
            offset = 0.0
        elif offset_raw == "auto":
            offset = "auto"
        else:
            offset = float(offset_raw)
        prefilter_sigma = float(getattr(parameters, "prefilter_sigma", 0.0))
        start = str(getattr(parameters, "start", "auto")).strip().lower()
        if start not in _START_MODES:
            start = "flat"
        sparse_hessian_weight = float(getattr(parameters, "sparse_hessian_weight", 0.6))
        sparse_hessian_reg = float(getattr(parameters, "sparse_hessian_reg", 0.98))
        convergence = str(getattr(parameters, "convergence", "auto")).strip().lower()
        if convergence in ("none", "fixed"):
            convergence = "fixed"
        rel_threshold = float(getattr(parameters, "rel_threshold", 0.005))
        check_every = 5          # convergence check interval

        # Hardcoded defaults (removed from descriptor to reduce parameter count)
        t_g = 170000.0           # coverslip thickness (nm) — standard #1.5
        t_g0 = 170000.0          # design coverslip thickness (nm)
        t_i0 = 100000.0          # design immersion thickness (nm)
        z_p = 0.0                # particle depth (nm)

        # Pixel size parameters from descriptor are in nm; loader metadata uses um.
        px_xy_raw = str(getattr(parameters, "pixel_size_xy", _DEFAULT_PIXEL_SIZE_XY_NM)).strip()
        px_xy_nm = _parse_float_or_default(px_xy_raw, _DEFAULT_PIXEL_SIZE_XY_NM)
        px_xy_value = px_xy_nm / 1000.0
        px_z_raw = str(getattr(parameters, "pixel_size_z", _DEFAULT_PIXEL_SIZE_Z_NM)).strip()
        px_z_nm = _parse_float_or_default(px_z_raw, _DEFAULT_PIXEL_SIZE_Z_NM)
        px_z_value = px_z_nm / 1000.0

        na_override = na_value
        ri_override = ri_value
        sample_ri = sample_ri_value
        micro_override = micro_value
        em_override = em_value
        ex_override = ex_value
        pinhole_airy_override = pinhole_airy
        px_xy_override = px_xy_value
        px_z_override = px_z_value

        projection = str(getattr(parameters, "projection", "none")).lower()
        benchmark = _to_bool(getattr(parameters, "benchmark", False))
        bench_crop = _to_bool(getattr(parameters, "bench_crop", False))
        compute_metrics = _to_bool(getattr(parameters, "compute_metrics", False))
        output_format = str(getattr(parameters, "output_format", "ome-tiff")).strip().lower()
        if output_format in ("ome_zarr", "zarr"):
            output_format = "ome-zarr"
        streaming_mode = str(getattr(parameters, "streaming", "auto")).strip().lower()
        tile_limits = _parse_tile_limits(getattr(parameters, "tile_limits", "auto"))
        streaming_threshold_gb = float(getattr(parameters, "streaming_threshold_gb", 2.0))
        scene = getattr(parameters, "scene", None)
        scene = None if scene in (None, "", "auto") else scene
        hcs_field = getattr(parameters, "hcs_field", None)
        hcs_field = None if hcs_field in (None, "", "auto") else str(hcs_field)

        # 2D widefield parameters
        two_d_mode = str(getattr(parameters, "two_d_mode", "auto")).strip().lower()
        two_d_wf_aggressiveness = str(getattr(parameters, "two_d_wf_aggressiveness", "Balanced")).strip()
        two_d_wf_bg_radius_um = float(getattr(parameters, "two_d_wf_bg_radius_um", 0.5))
        two_d_wf_bg_scale = float(getattr(parameters, "two_d_wf_bg_scale", 1.0))

        print("=" * 70)
        print("CIDeconvolve - Bilayers Workflow")
        print("=" * 70)
        _print_runtime_environment()
        print("\nRun configuration")
        print(f"  Input dir    : {bj.input_dir}")
        print(f"  Output dir   : {bj.output_dir}")
        print(f"  Method       : {method}")
        print(f"  Iterations   : {', '.join(str(n) for n in niter_list)}")
        print(f"  Device       : {device_param}")
        print(f"  Projection   : {projection}")
        print(f"  Output format: {output_format}")
        tile_text = "auto" if int(tile_limits[0]) <= 0 else f"{tile_limits[0]} px"
        print(f"  Streaming    : {streaming_mode} (threshold={streaming_threshold_gb:g} GB, tile={tile_text})")
        print(f"  Metadata     : {'overrule image metadata' if overrule_metadata else 'use image metadata'}")
        if method == "ci_rl_tv":
            print(f"  TV lambda    : {tv_lambda}")
        if method in ("ci_rl", "ci_rl_tv"):
            print(f"  Damping      : {damping}")
        if method == "ci_sparse_hessian":
            print(f"  Sparse weight: {sparse_hessian_weight}")
            print(f"  Sparse reg   : {sparse_hessian_reg}")
        print(f"  Background   : {background}")
        print(f"  Offset       : {offset}")
        if prefilter_sigma > 0.0:
            print(f"  Prefilter    : sigma={prefilter_sigma}")
        print(f"  Start        : {start}")
        print(f"  Convergence  : {convergence} (threshold={rel_threshold}, every={check_every})")
        if overrule_metadata:
            print(f"  NA           : {na_value}")
            print(f"  Immersion    : {ri_raw} -> RI {ri_value}")
            print(f"  Sample medium: {sample_ri_raw} -> RI {sample_ri_value}")
            print(f"  Microscope   : {micro_value}")
            print(f"  Emission WL  : {em_value}")
            print(f"  Excitation WL: {ex_value or 'none'}")
            print(f"  Pinhole      : {_format_float_list(pinhole_airy)} AU")
            print(f"  Pixel size   : XY={px_xy_nm:g} nm  Z={px_z_nm:g} nm")
        else:
            print("  Metadata params: descriptor values used only where image metadata is missing")
        if benchmark:
            print(f"  Benchmark    : ON (crop={bench_crop})")
        print(f"  Image metrics: {'ON' if compute_metrics else 'OFF'}")

        # Prepare data directories and collect input images
        in_imgs, _, in_path, _, out_path, tmp_path = prepare_data(
            get_discipline(bj, default=CLASS_SPTCNT), bj, is_2d=False, **bj.flags
        )

        if not in_imgs:
            print("CIDeconvolve workflow failed: no input images found.")
            return 1

        print(f"\nFound {len(in_imgs)} input image(s).")

        # ---- Benchmark mode ----
        if benchmark:
            try:
                # Always use only the first image/plate field for benchmark
                ok = _run_benchmark(
                    in_imgs[0], in_path, out_path,
                    niter_list=niter_list,
                    device=device,
                    bench_crop=bench_crop,
                    compute_metrics=compute_metrics,
                    na=na_override,
                    refractive_index=ri_override,
                    sample_ri=sample_ri,
                    microscope_type=micro_override,
                    emission_wavelengths=em_override,
                    excitation_wavelengths=ex_override,
                    pinhole_airy_units=pinhole_airy_override,
                    overrule_metadata=overrule_metadata,
                    pixel_size_xy=px_xy_override,
                    pixel_size_z=px_z_override,
                    tv_lambda=tv_lambda,
                    damping=damping,
                    sparse_hessian_weight=sparse_hessian_weight,
                    sparse_hessian_reg=sparse_hessian_reg,
                    background=background,
                    offset=offset,
                    prefilter_sigma=prefilter_sigma,
                    start=start,
                    convergence=convergence,
                    rel_threshold=rel_threshold,
                    check_every=check_every,
                    ri_coverslip=ri_override if overrule_metadata else None,
                    ri_coverslip_design=ri_override if overrule_metadata else None,
                    ri_immersion_design=ri_override if overrule_metadata else None,
                    t_g=t_g, t_g0=t_g0, t_i0=t_i0, z_p=z_p,
                )
            except Exception as exc:
                print(f"Benchmark failed: {exc}")
                import traceback
                traceback.print_exc()
                ok = False
            if tmp_path and Path(tmp_path).exists():
                shutil.rmtree(tmp_path, ignore_errors=True)
            print(f"\n{'=' * 70}")
            if ok:
                print("CIDeconvolve benchmark complete.")
            else:
                print("CIDeconvolve benchmark failed.")
            return 0 if ok else 1

        # ---- Separate plate zarrs from regular images ----
        plate_imgs = []
        regular_imgs = []
        failed_items: list[str] = []
        for img_resource in in_imgs:
            img_path = Path(in_path) / img_resource.filename
            if img_path.is_dir() and img_path.suffix.lower() == ".zarr" and _is_hcs_plate(img_path):
                plate_imgs.append(img_resource)
            else:
                regular_imgs.append(img_resource)

        # ---- Process HCS plate zarrs ----
        for img_resource in plate_imgs:
            zarr_path = Path(in_path) / img_resource.filename
            print(f"\n{'=' * 60}")
            print(f"Processing HCS Plate: {img_resource.filename}")
            print(f"{'=' * 60}")

            try:
                _run_plate_zarr(
                    zarr_path, out_path,
                    niter_list=niter_list,
                    method=method,
                    device=device,
                    na=na_override,
                    refractive_index=ri_override,
                    sample_ri=sample_ri,
                    microscope_type=micro_override,
                    emission_wavelengths=em_override,
                    excitation_wavelengths=ex_override,
                    pinhole_airy_units=pinhole_airy_override,
                    overrule_metadata=overrule_metadata,
                    pixel_size_xy=px_xy_override,
                    pixel_size_z=px_z_override,
                    tv_lambda=tv_lambda,
                    damping=damping,
                    sparse_hessian_weight=sparse_hessian_weight,
                    sparse_hessian_reg=sparse_hessian_reg,
                    background=background,
                    offset=offset,
                    prefilter_sigma=prefilter_sigma,
                    start=start,
                    convergence=convergence,
                    rel_threshold=rel_threshold,
                    check_every=check_every,
                    ri_coverslip=ri_override if overrule_metadata else None,
                    ri_coverslip_design=ri_override if overrule_metadata else None,
                    ri_immersion_design=ri_override if overrule_metadata else None,
                    t_g=t_g, t_g0=t_g0, t_i0=t_i0, z_p=z_p,
                    two_d_mode=two_d_mode,
                    two_d_wf_aggressiveness=two_d_wf_aggressiveness,
                    two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
                    two_d_wf_bg_scale=two_d_wf_bg_scale,
                    projection=projection,
                )
            except Exception as exc:
                failed_items.append(img_resource.filename)
                print(f"  ERROR processing plate {img_resource.filename}: {exc}")
                import traceback
                traceback.print_exc()

        # ---- Process regular (non-plate) images ----
        for img_resource in regular_imgs:
            img_path = Path(in_path) / img_resource.filename
            print(f"\n{'=' * 60}")
            print(f"Processing: {img_resource.filename}")
            print(f"{'=' * 60}")

            t0 = time.time()
            load_time = 0.0
            deconv_metrics: dict[str, float] = {}
            save_time = 0.0
            monitor: _MetricsMonitor | None = None

            try:
                use_streaming = False
                if projection != "none" and output_format == "ome-zarr":
                    raise ValueError("OME-Zarr streaming output currently writes full Z data; set projection=none.")
                if streaming_mode not in ("auto", "always", "never"):
                    raise ValueError("--streaming must be auto, always, or never")
                if output_format == "ome-zarr" or streaming_mode == "always":
                    use_streaming = True
                elif streaming_mode == "auto":
                    try:
                        probe_source = open_region_source(img_path, scene=scene, hcs_field=hcs_field)
                        use_streaming = should_stream_source(
                            probe_source.shape,
                            threshold_gb=streaming_threshold_gb,
                        )
                        if use_streaming:
                            print(
                                "  Streaming auto-enabled: source shape "
                                f"{probe_source.shape} exceeds {streaming_threshold_gb:g} GB threshold."
                            )
                    except Exception as probe_exc:
                        print(f"  Streaming probe unavailable; using eager load ({probe_exc})")

                if use_streaming:
                    if projection != "none":
                        raise ValueError("Streaming output currently writes full Z data; set projection=none.")
                    if output_format not in ("ome-zarr", "zarr"):
                        print("  Large image detected; switching output format to OME-Zarr for streamed float output.")
                    out_zarr = _run_streaming_regular_image(
                        img_path,
                        out_path,
                        stem=_stem(img_resource.filename),
                        niter_list=niter_list,
                        method=method,
                        device=device,
                        na=na_override,
                        refractive_index=ri_override,
                        sample_ri=sample_ri,
                        microscope_type=micro_override,
                        emission_wavelengths=em_override,
                        excitation_wavelengths=ex_override,
                        pinhole_airy_units=pinhole_airy_override,
                        pixel_size_xy=px_xy_override,
                        pixel_size_z=px_z_override,
                        overrule_metadata=overrule_metadata,
                        tv_lambda=tv_lambda,
                        damping=damping,
                        sparse_hessian_weight=sparse_hessian_weight,
                        sparse_hessian_reg=sparse_hessian_reg,
                        background=background,
                        offset=offset,
                        prefilter_sigma=prefilter_sigma,
                        start=start,
                        convergence=convergence,
                        rel_threshold=rel_threshold,
                        check_every=check_every,
                        ri_coverslip=ri_override if overrule_metadata else None,
                        ri_coverslip_design=ri_override if overrule_metadata else None,
                        ri_immersion_design=ri_override if overrule_metadata else None,
                        t_g=t_g,
                        t_g0=t_g0,
                        t_i0=t_i0,
                        z_p=z_p,
                        two_d_mode=two_d_mode,
                        two_d_wf_aggressiveness=two_d_wf_aggressiveness,
                        two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
                        two_d_wf_bg_scale=two_d_wf_bg_scale,
                        tile_limits=tile_limits,
                        scene=scene,
                        hcs_field=hcs_field,
                    )
                    print(f"  Output path : {out_zarr}")
                    elapsed = time.time() - t0
                    print("\n  Timing summary")
                    print(f"    Total      : {_format_duration(elapsed)}")
                    continue

                # Load image and extract metadata
                t_load = time.time()
                data = load_image(
                    img_path,
                    na=na_override,
                    refractive_index=ri_override,
                    sample_refractive_index=sample_ri,
                    microscope_type=micro_override,
                    emission_wavelengths=em_override,
                    excitation_wavelengths=ex_override,
                    pinhole_airy_units=pinhole_airy_override,
                    overrule_metadata=overrule_metadata,
                    pixel_size_xy=px_xy_override,
                    pixel_size_z=px_z_override,
                )
                meta = data["metadata"]
                images = data["images"]
                load_time = time.time() - t_load
                _print_image_details(img_resource.filename, img_path, meta, images)
                print(f"    Load time  : {_format_duration(load_time)}")

                # Create a temp dir alongside outfolder for intermediate files
                tmp_work = Path(out_path) / "tmp"
                tmp_work.mkdir(parents=True, exist_ok=True)

                # ----- Deconvolve -----
                print("\n  Deconvolution")
                print(f"    Method     : {method}")
                print(f"    Iterations : {', '.join(str(n) for n in niter_list)}")
                print(f"    Device     : {device_param}")
                print(f"    Background : {background}")
                print(f"    Offset     : {offset}")
                print(f"    Start      : {start}")
                print(f"    Convergence: {convergence} "
                      f"(threshold={rel_threshold}, every={check_every})")
                if method == "ci_rl_tv":
                    print(f"    TV lambda  : {tv_lambda}")
                if method in ("ci_rl", "ci_rl_tv"):
                    print(f"    Damping    : {damping}")
                    print(f"    2D WF mode : {two_d_mode} "
                          f"(aggr={two_d_wf_aggressiveness}, "
                          f"bg radius={two_d_wf_bg_radius_um} um, "
                          f"bg scale={two_d_wf_bg_scale})")
                if method == "ci_sparse_hessian":
                    print(f"    Sparse     : weight={sparse_hessian_weight}, "
                          f"reg={sparse_hessian_reg}")
                if prefilter_sigma > 0.0:
                    print(f"    Prefilter  : sigma={prefilter_sigma}")

                monitor = _MetricsMonitor()
                monitor.start()
                result_channels = []
                psfs = []
                for ch_idx, img in enumerate(images):
                    ch_niter = niter_list[ch_idx] if ch_idx < len(niter_list) else niter_list[-1]
                    t_ch = time.time()
                    print(f"    Channel {ch_idx}: PSF generation...")
                    psf = generate_psf(
                        meta,
                        channel_idx=ch_idx,
                        ri_coverslip=ri_override if overrule_metadata else None,
                        ri_coverslip_design=ri_override if overrule_metadata else None,
                        ri_immersion_design=ri_override if overrule_metadata else None,
                        t_g=t_g,
                        t_g0=t_g0,
                        t_i0=t_i0,
                        z_p=z_p,
                        two_d_mode=two_d_mode if method in ("ci_rl", "ci_rl_tv") else "legacy_2d",
                    )
                    psfs.append(psf)

                    keep_hidden_2d_psf = (
                        img.ndim == 2
                        and psf.ndim == 3
                        and meta.get("microscope_type", "widefield") == "widefield"
                        and method in ("ci_rl", "ci_rl_tv")
                        and str(two_d_mode).strip().lower() == "auto"
                    )
                    effective_psf = psf
                    if img.ndim == 2 and effective_psf.ndim == 3 and not keep_hidden_2d_psf:
                        effective_psf = effective_psf[effective_psf.shape[0] // 2]
                    elif img.ndim == 3 and effective_psf.ndim == 2:
                        effective_psf = effective_psf[np.newaxis, :, :]

                    print(f"    Channel {ch_idx}: deconvolving shape={img.shape} "
                          f"psf={effective_psf.shape} iterations={ch_niter}")
                    result_channels.append(
                        deconvolve(
                            img,
                            effective_psf,
                            method=method,
                            niter=ch_niter,
                            background=background,
                            damping=damping,
                            offset=offset,
                            prefilter_sigma=prefilter_sigma,
                            start=start,
                            convergence=convergence,
                            rel_threshold=rel_threshold,
                            check_every=check_every,
                            device=device,
                            tv_lambda=tv_lambda,
                            sparse_hessian_weight=sparse_hessian_weight,
                            sparse_hessian_reg=sparse_hessian_reg,
                            pixel_size_xy=meta.get("pixel_size_x"),
                            pixel_size_z=meta.get("pixel_size_z"),
                            microscope_type=meta.get("microscope_type", "widefield"),
                            two_d_mode=two_d_mode,
                            two_d_wf_aggressiveness=two_d_wf_aggressiveness,
                            two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
                            two_d_wf_bg_scale=two_d_wf_bg_scale,
                        )
                    )
                    print(f"    Channel {ch_idx}: done in {_format_duration(time.time() - t_ch)}")
                result = {
                    "channels": result_channels,
                    "psfs": psfs,
                    "metadata": meta,
                    "source_channels": images,
                }
                deconv_metrics = monitor.stop()

                if result is None:
                    failed_items.append(img_resource.filename)
                    print(f"  ERROR: deconvolution returned no result for {img_resource.filename}")
                    shutil.rmtree(tmp_work, ignore_errors=True)
                    continue

                _print_psf_details(result.get("psfs", []))
                _print_resource_metrics(deconv_metrics)
                if compute_metrics and result.get("source_channels"):
                    t_metrics = time.time()
                    _print_quality_comparison(result["source_channels"], result["channels"])
                    print(f"  Image metrics computed in {_format_duration(time.time() - t_metrics)}")
                else:
                    print("\n  Image metrics skipped (disabled).")

                stem = _stem(img_resource.filename)
                is_3d = result["channels"][0].ndim == 3

                t_save = time.time()
                if projection in ("mip", "sum") and is_3d:
                    out_name = f"{stem}_decon{_projection_output_suffix(projection)}.ome.tiff"
                    tmp_file = tmp_work / out_name
                    proj_result = dict(result)
                    if projection == "mip":
                        proj_result["channels"] = [
                            ch.max(axis=0) for ch in result["channels"]
                        ]
                        if result.get("source_channels"):
                            proj_result["source_channels"] = [
                                ch.max(axis=0) for ch in result["source_channels"]
                            ]
                    else:  # sum
                        proj_result["channels"] = [
                            ch.astype(np.float32).sum(axis=0) for ch in result["channels"]
                        ]
                        if result.get("source_channels"):
                            proj_result["source_channels"] = [
                                ch.astype(np.float32).sum(axis=0) for ch in result["source_channels"]
                            ]
                    save_result(proj_result, str(tmp_file))
                    print(f"  Saved {projection.upper()}: {out_name}")
                else:
                    out_name = f"{stem}_decon.ome.tiff"
                    tmp_file = tmp_work / out_name
                    save_result(result, str(tmp_file))
                    print(f"  Saved: {out_name}")
                save_time = time.time() - t_save

                # Move only the deconvolved TIFF to the output folder
                dest = Path(out_path) / out_name
                shutil.move(str(tmp_file), str(dest))
                if dest.exists():
                    print(f"  Output file : {dest}")
                    print(f"  Output size : {_format_bytes(dest.stat().st_size / (1024 * 1024))}")

                # Clean up the temp working directory
                shutil.rmtree(tmp_work, ignore_errors=True)

            except Exception as exc:
                failed_items.append(img_resource.filename)
                if monitor is not None and not deconv_metrics:
                    try:
                        deconv_metrics = monitor.stop()
                    except Exception:
                        pass
                print(f"  ERROR processing {img_resource.filename}: {exc}")
                import traceback
                traceback.print_exc()
                # Clean up temp dir on failure so no partial files remain
                tmp_work = Path(out_path) / "tmp"
                shutil.rmtree(tmp_work, ignore_errors=True)
                continue

            elapsed = time.time() - t0
            print("\n  Timing summary")
            print(f"    Load       : {_format_duration(load_time)}")
            print(f"    Deconvolve : {_format_duration(deconv_metrics.get('time_s', 0.0))}")
            print(f"    Save       : {_format_duration(save_time)}")
            print(f"    Total      : {_format_duration(elapsed)}")

        # Clean up tmp folder before the final status so the last log line is the outcome.
        if tmp_path and Path(tmp_path).exists():
            shutil.rmtree(tmp_path, ignore_errors=True)

        print(f"\n{'=' * 70}")
        if failed_items:
            print(f"CIDeconvolve workflow completed with errors: {len(failed_items)} failed item(s).")
            print(f"Failed items: {', '.join(failed_items)}")
            print("CIDeconvolve workflow failed.")
        else:
            print("CIDeconvolve workflow complete.")
        return 1 if failed_items else 0



# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
_BENCH_METHODS = ["ci_rl", "ci_rl_tv", "ci_sparse_hessian"]


def _method_device(method: str) -> str:
    """Return the compute device label for a benchmark method."""
    if method.startswith("ci_"):
        import torch
        return "CUDA" if torch.cuda.is_available() else "CPU"
    return "?"


# ---------------------------------------------------------------------------
# Background metrics monitor
# ---------------------------------------------------------------------------
class _MetricsMonitor:
    """Daemon thread that samples CPU/RAM and GPU metrics during a run."""

    def __init__(self, interval=0.1):
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = None

        # Sampled data
        self._cpu_percent: list[float] = []
        self._ram_bytes: list[int] = []
        self._gpu_util: list[float] = []
        self._gpu_mem_bytes: list[int] = []

        # Baselines
        self._ram_baseline = 0
        self._gpu_mem_baseline = 0
        self._torch_baseline = 0

        # Timing
        self._t0 = 0.0
        self._t1 = 0.0

        # Detect capabilities
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
        except ImportError:
            pass

        self._nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def start(self):
        """Record baselines and begin sampling."""
        self._cpu_percent.clear()
        self._ram_bytes.clear()
        self._gpu_util.clear()
        self._gpu_mem_bytes.clear()
        self._stop_event.clear()

        if self._proc:
            self._proc.cpu_percent()          # prime
            self._ram_baseline = self._proc.memory_info().rss
        else:
            self._ram_baseline = 0

        if self._nvml_handle:
            import pynvml
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            self._gpu_mem_baseline = mem_info.used
        else:
            self._gpu_mem_baseline = 0

        self._torch_baseline = 0
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._torch_baseline = torch.cuda.memory_allocated()
        except Exception:
            pass

        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        """Sampling loop running in background thread."""
        while not self._stop_event.is_set():
            if self._proc:
                try:
                    self._cpu_percent.append(self._proc.cpu_percent())
                    self._ram_bytes.append(self._proc.memory_info().rss)
                except Exception:
                    pass
            if self._nvml_handle:
                try:
                    import pynvml
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    self._gpu_util.append(util.gpu)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    self._gpu_mem_bytes.append(mem_info.used)
                except Exception:
                    pass
            self._stop_event.wait(self._interval)

    def stop(self):
        """Stop sampling and return metrics dict."""
        self._t1 = time.perf_counter()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        elapsed = self._t1 - self._t0
        MB = 1024 * 1024

        m: dict[str, float] = {
            "time_s": elapsed,
            "cpu_percent_avg": 0.0,
            "cpu_percent_peak": 0.0,
            "ram_peak_mb": 0.0,
            "ram_avg_mb": 0.0,
            "ram_delta_peak_mb": 0.0,
            "gpu_util_avg": 0.0,
            "gpu_util_peak": 0.0,
            "gpu_mem_peak_mb": 0.0,
            "gpu_mem_avg_mb": 0.0,
            "gpu_mem_delta_peak_mb": 0.0,
            "torch_gpu_peak_mb": 0.0,
            "torch_gpu_delta_mb": 0.0,
            "gpu_total_mb": 0.0,
            "gpu_spill_mb": 0.0,
            "ram_total_mb": 0.0,
            "ram_percent": 0.0,
            "gpu_mem_percent": 0.0,
            "gpu_util_available": 0.0,
        }

        if self._cpu_percent:
            m["cpu_percent_avg"] = sum(self._cpu_percent) / len(self._cpu_percent)
            m["cpu_percent_peak"] = max(self._cpu_percent)

        if self._proc:
            import psutil
            m["ram_total_mb"] = psutil.virtual_memory().total / MB

        if self._ram_bytes:
            m["ram_peak_mb"] = max(self._ram_bytes) / MB
            m["ram_avg_mb"] = sum(self._ram_bytes) / len(self._ram_bytes) / MB
            m["ram_delta_peak_mb"] = (max(self._ram_bytes) - self._ram_baseline) / MB
            if m["ram_total_mb"] > 0:
                m["ram_percent"] = m["ram_peak_mb"] / m["ram_total_mb"] * 100

        if self._gpu_util:
            m["gpu_util_avg"] = sum(self._gpu_util) / len(self._gpu_util)
            m["gpu_util_peak"] = max(self._gpu_util)
            m["gpu_util_available"] = 1.0

        if self._gpu_mem_bytes:
            m["gpu_mem_peak_mb"] = max(self._gpu_mem_bytes) / MB
            m["gpu_mem_avg_mb"] = (
                sum(self._gpu_mem_bytes) / len(self._gpu_mem_bytes) / MB
            )
            m["gpu_mem_delta_peak_mb"] = (
                max(self._gpu_mem_bytes) - self._gpu_mem_baseline
            ) / MB

        if self._nvml_handle:
            try:
                import pynvml
                total_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                m["gpu_total_mb"] = total_info.total / MB
            except Exception:
                pass

        try:
            import torch
            if torch.cuda.is_available():
                m["torch_gpu_peak_mb"] = torch.cuda.max_memory_allocated() / MB
                m["torch_gpu_delta_mb"] = (
                    torch.cuda.max_memory_allocated() - self._torch_baseline
                ) / MB
                if m["gpu_total_mb"] <= 0:
                    props = torch.cuda.get_device_properties(torch.cuda.current_device())
                    m["gpu_total_mb"] = props.total_memory / MB
        except Exception:
            pass

        # NVML is not always available in Docker/Singularity even when CUDA works.
        # In that case, expose the PyTorch allocator peak in the generic VRAM
        # fields so CSV summaries do not look like GPU usage was zero.
        if m["gpu_mem_peak_mb"] <= 0 and m["torch_gpu_peak_mb"] > 0:
            m["gpu_mem_peak_mb"] = m["torch_gpu_peak_mb"]
            m["gpu_mem_avg_mb"] = m["torch_gpu_peak_mb"]
        if m["gpu_mem_delta_peak_mb"] <= 0 and m["torch_gpu_delta_mb"] > 0:
            m["gpu_mem_delta_peak_mb"] = m["torch_gpu_delta_mb"]

        if m["gpu_total_mb"] > 0 and m["gpu_mem_peak_mb"] > 0:
            m["gpu_mem_percent"] = (
                m["gpu_mem_peak_mb"] / m["gpu_total_mb"] * 100
            )

        if m["gpu_total_mb"] > 0 and m["torch_gpu_delta_mb"] > m["gpu_total_mb"]:
            m["gpu_spill_mb"] = m["torch_gpu_delta_mb"] - m["gpu_total_mb"]

        return m


# ---------------------------------------------------------------------------
# CSV & montage helpers
# ---------------------------------------------------------------------------

def _write_metrics_csv(csv_path: Path, all_metrics: dict[str, dict]):
    """Write benchmark metrics to a CSV file."""
    fieldnames = [
        "label", "device", "time_s",
        "cpu_percent_avg", "cpu_percent_peak",
        "ram_total_mb", "ram_peak_mb", "ram_percent", "ram_avg_mb",
        "ram_delta_peak_mb",
        "gpu_util_avg", "gpu_util_peak",
        "gpu_util_available",
        "gpu_total_mb", "gpu_mem_peak_mb", "gpu_mem_percent", "gpu_mem_avg_mb",
        "gpu_mem_delta_peak_mb",
        "torch_gpu_peak_mb", "torch_gpu_delta_mb",
        "gpu_spill_mb",
    ]
    quality_fieldnames = [
        "channels_compared", "detail_energy_mean", "bright_detail_energy_mean",
        "edge_strength_mean", "signal_sparsity_mean", "robust_range_mean",
    ]
    include_quality = any(
        any(key in metrics for key in quality_fieldnames)
        for metrics in all_metrics.values()
    )
    if include_quality:
        fieldnames += quality_fieldnames
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, m in sorted(all_metrics.items()):
            row = {"label": label, "device": m.get("device", "")}
            for k in fieldnames[2:]:
                row[k] = f"{m.get(k, 0.0):.2f}"
            writer.writerow(row)
    print(f"\n  Metrics CSV saved -> {csv_path}")


def _scaled_font(max_dim):
    """Return a Pillow font scaled to the image size and a matching label height."""
    from PIL import ImageFont

    # Scale font: ~3.5% of the largest image dimension (min 18px)
    font_size = max(18, int(max_dim * 0.035))
    label_height = int(font_size * 2.5) + 10  # room for 2 lines of text

    font = None
    for name in (
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(name, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    return font, label_height


def _make_metadata_panel(meta, width, height, font):
    """Create a metadata text panel for the montage."""
    from PIL import Image, ImageDraw

    panel = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(panel)

    lines = [
        f"NA: {meta.get('na', '?')}",
        f"RI immersion: {meta.get('refractive_index', '?')}",
        f"RI sample: {meta.get('sample_refractive_index', '?')}",
        f"Pixel XY: {meta.get('pixel_size_x', '?')} um",
        f"Pixel Z:  {meta.get('pixel_size_z', '?')} um",
        f"Size: {meta.get('size_x', '?')}x{meta.get('size_y', '?')}"
        f"x{meta.get('size_z', '?')}",
        f"Microscope: {meta.get('microscope_type', '?')}",
    ]
    for i, ch in enumerate(meta.get("channels", [])):
        em = ch.get("emission_wavelength") or "?"
        lines.append(f"Ch{i}: Em {em} nm")

    margin = max(8, int(height * 0.02))
    draw.text((margin, margin), "\n".join(lines), fill=(255, 255, 255), font=font)
    return panel


def _make_benchmark_montage(
    out_path,
    tmp_path,
    stem,
    available_methods,
    bench_iterations,
    all_metrics,
    metadata,
):
    """Create an RGB montage of all benchmark MIP PNGs.

    Layout: Row 0 = Source + metadata panel spanning remaining columns.
    Each subsequent row = one iteration config, one column per method.
    """
    from PIL import Image, ImageDraw

    out_dir = Path(out_path)
    tmp_dir = Path(tmp_path)

    # Column order
    col_order = [(m, None) for m in available_methods]

    # Row 0: source MIP
    rows = [[(tmp_dir / "mip_source.ome.png", "Source")]]

    # One row per iteration config
    for nit_tag, nit_label in bench_iterations:
        row = []
        for method, _variant in col_order:
            fname = f"mip_{stem}_{method}_{nit_tag}i.ome.png"
            metrics_key = f"{method}_{nit_tag}i"
            met = all_metrics.get(metrics_key)
            if met is not None:
                label = f"{method}\n{nit_label} iter  {met['time_s']:.1f}s"
            else:
                label = f"{method}\n{nit_label} iter"
            row.append((tmp_dir / fname, label))
        rows.append(row)

    # Load existing images, skip missing
    loaded_rows = []
    total = 0
    for row_entries in rows:
        row_images = []
        for path, label in row_entries:
            if path.exists():
                img = Image.open(path).convert("RGB")
                row_images.append((img, label))
                total += 1
        if row_images:
            loaded_rows.append(row_images)

    if total == 0:
        print("  No MIP PNG files found -- skipping montage.")
        return None

    padding = 4

    all_imgs = [img for row in loaded_rows for img, _ in row]
    max_w = max(img.size[0] for img in all_imgs)
    max_h = max(img.size[1] for img in all_imgs)

    # Scale font to image size
    font, label_height = _scaled_font(max(max_w, max_h))

    n_cols = max(len(row) for row in loaded_rows)
    n_rows = len(loaded_rows)
    cell_w = max_w + 2 * padding
    cell_h = max_h + label_height + 2 * padding

    # Metadata panel spans all columns to the right of Source
    span_cols = max(n_cols - 1, 1)
    meta_w = span_cols * cell_w - 2 * padding
    meta_panel = _make_metadata_panel(metadata, meta_w, max_h, font)

    montage_w = n_cols * cell_w
    montage_h = n_rows * cell_h

    montage = Image.new("RGB", (montage_w, montage_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(montage)

    for row_idx, row_images in enumerate(loaded_rows):
        for col_idx, (img, label) in enumerate(row_images):
            x0 = col_idx * cell_w + padding
            y0 = row_idx * cell_h + padding
            x_off = (max_w - img.size[0]) // 2
            y_off = (max_h - img.size[1]) // 2
            montage.paste(img, (x0 + x_off, y0 + y_off))

            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            tx = x0 + (max_w - tw) // 2
            ty = y0 + max_h + padding
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    # Paste metadata panel at row 0, starting at column 1
    meta_x = cell_w + padding
    meta_y = padding
    montage.paste(meta_panel, (meta_x, meta_y))

    montage_path = out_dir / f"decon_benchmark_{stem}.png"
    montage.save(str(montage_path))
    print(f"  Saved montage: {montage_path}  ({montage_w}x{montage_h})")
    return montage_path


def _make_per_channel_montages(
    out_path,
    tmp_path,
    stem,
    available_methods,
    bench_iterations,
    all_metrics,
    metadata,
):
    """Create one greyscale montage per channel from benchmark MIP TIFFs."""
    from PIL import Image, ImageDraw
    import tifffile

    out_dir = Path(out_path)
    tmp_dir = Path(tmp_path)
    n_ch = metadata.get("n_channels", 1)
    if n_ch < 1:
        n_ch = 1

    col_order = [(m, None) for m in available_methods]

    # Row 0 = source, rows 1-N = per iteration config
    mip_rows = [
        [("Source", tmp_dir / "mip_source.ome.tiff")],
    ]
    for nit_tag, nit_label in bench_iterations:
        row = []
        for method, _variant in col_order:
            fname = f"mip_{stem}_{method}_{nit_tag}i.ome.tiff"
            metrics_key = f"{method}_{nit_tag}i"
            met = all_metrics.get(metrics_key)
            if met is not None:
                label = f"{method}\n{nit_label} iter  {met['time_s']:.1f}s"
            else:
                label = f"{method}\n{nit_label} iter"
            row.append((label, tmp_dir / fname))
        mip_rows.append(row)

    # Load TIFF arrays per row
    loaded_rows = []
    for row_entries in mip_rows:
        row_data = []
        for label, path in row_entries:
            if path.exists():
                arr = tifffile.imread(str(path))
                if arr.ndim == 2:
                    arr = arr[np.newaxis]
                row_data.append((label, arr))
        if row_data:
            loaded_rows.append(row_data)

    if not loaded_rows:
        return

    print(f"\n  Creating per-channel montages ({n_ch} channels)...")

    padding = 4

    # Pre-compute scaled font from the first loaded image dimensions
    first_arr = loaded_rows[0][0][1]  # (label, arr)
    arr_max_dim = max(first_arr.shape[-2], first_arr.shape[-1])
    font, label_height = _scaled_font(arr_max_dim)

    for ch_idx in range(n_ch):
        panel_rows = []
        for row_data in loaded_rows:
            row_panels = []
            for label, arr in row_data:
                if ch_idx >= arr.shape[0]:
                    continue
                ch_data = arr[ch_idx].astype(np.float64)
                lo, hi = ch_data.min(), ch_data.max()
                if hi > lo:
                    ch_data = (ch_data - lo) / (hi - lo)
                else:
                    ch_data = np.zeros_like(ch_data)
                ch_uint8 = (ch_data * 255).astype(np.uint8)
                img = Image.fromarray(ch_uint8, mode="L").convert("RGB")
                img_draw = ImageDraw.Draw(img)
                img_draw.text(
                    (4, 2), f"Ch{ch_idx}",
                    fill=(255, 255, 255), font=font,
                )
                row_panels.append((img, label))
            if row_panels:
                panel_rows.append(row_panels)

        if not panel_rows:
            continue

        all_imgs = [img for row in panel_rows for img, _ in row]
        max_w = max(img.size[0] for img in all_imgs)
        max_h = max(img.size[1] for img in all_imgs)

        n_cols = max(len(row) for row in panel_rows)
        n_grid_rows = len(panel_rows)
        cell_w = max_w + 2 * padding
        cell_h = max_h + label_height + 2 * padding

        span_cols = max(n_cols - 1, 1)
        meta_w = span_cols * cell_w - 2 * padding
        meta_panel = _make_metadata_panel(metadata, meta_w, max_h, font)

        montage_w = n_cols * cell_w
        montage_h = n_grid_rows * cell_h

        montage = Image.new("RGB", (montage_w, montage_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(montage)

        for row_idx, row_panels_r in enumerate(panel_rows):
            for col_idx, (img, label) in enumerate(row_panels_r):
                x0 = col_idx * cell_w + padding
                y0 = row_idx * cell_h + padding
                x_off = (max_w - img.size[0]) // 2
                y_off = (max_h - img.size[1]) // 2
                montage.paste(img, (x0 + x_off, y0 + y_off))

                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                tx = x0 + (max_w - tw) // 2
                ty = y0 + max_h + padding
                draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

        meta_x = cell_w + padding
        meta_y = padding
        montage.paste(meta_panel, (meta_x, meta_y))

        ch_path = out_dir / f"decon_benchmark_{stem}_ch{ch_idx}.png"
        montage.save(str(ch_path))
        print(f"    Ch{ch_idx}: {ch_path}  ({montage_w}x{montage_h})")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _write_benchmark_ome_tiff(path: Path, images):
    """Write channel-first benchmark data with explicit OME axes."""
    import tifffile

    stack = np.stack(images, axis=0)
    if stack.ndim == 4:
        axes = "CZYX"
    elif stack.ndim == 3:
        axes = "CYX"
    else:
        raise ValueError(f"Unsupported benchmark stack shape: {stack.shape}")
    tifffile.imwrite(
        str(path),
        stack,
        ome=True,
        photometric="minisblack",
        metadata={"axes": axes},
    )


def _run_benchmark(
    img_resource,
    in_path,
    out_path,
    *,
    niter_list,
    device,
    bench_crop,
    compute_metrics,
    na,
    refractive_index,
    sample_ri,
    microscope_type,
    emission_wavelengths,
    excitation_wavelengths,
    pinhole_airy_units,
    pixel_size_xy,
    pixel_size_z,
    overrule_metadata,
    tv_lambda,
    damping,
    sparse_hessian_weight,
    sparse_hessian_reg,
    background,
    offset,
    prefilter_sigma,
    start,
    convergence,
    rel_threshold,
    check_every,
    ri_coverslip,
    ri_coverslip_design,
    ri_immersion_design,
    t_g,
    t_g0,
    t_i0,
    z_p,
):
    """Run benchmark on the first input image with both methods."""
    import gc
    import torch

    img_path = Path(in_path) / img_resource.filename
    stem = _stem(img_resource.filename)
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Derive iteration tag and label from descriptor iterations
    if len(set(niter_list)) == 1:
        nit_tag = str(niter_list[0])
        nit_label = str(niter_list[0])
    else:
        nit_tag = "-".join(str(n) for n in niter_list)
        nit_label = "/".join(str(n) for n in niter_list)
    bench_iterations = [(nit_tag, nit_label)]

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: {img_resource.filename}")
    print(f"  Methods   : {', '.join(_BENCH_METHODS)}")
    print(f"  Iterations: {nit_label}")
    print(f"  Crop      : {bench_crop}")
    print(f"  Metrics   : {compute_metrics}")
    print(f"{'=' * 70}")

    # Load image once — for HCS plates, use only the first well/field
    _plate_path = img_path
    if _plate_path.is_dir() and _plate_path.suffix.lower() == ".zarr" and _is_hcs_plate(_plate_path):
        plate_info = _get_zarr_plate_info(_plate_path)
        wf = plate_info["wells_and_fields"]
        if not wf:
            print("  No fields found in plate. Exiting benchmark.")
            return False
        row, col, field = wf[0]
        print(f"  Plate detected — using first field: {row}/{col}/{field}")
        data = _load_zarr_field(
            _plate_path, row, col, field,
            na=na,
            refractive_index=refractive_index,
            sample_refractive_index=sample_ri,
            microscope_type=microscope_type,
            pixel_size_xy=pixel_size_xy,
            pixel_size_z=pixel_size_z,
            emission_wavelengths=emission_wavelengths,
            excitation_wavelengths=excitation_wavelengths,
            pinhole_airy_units=pinhole_airy_units,
            overrule_metadata=overrule_metadata,
        )
        meta = data["metadata"]
        images = data["images"]
        # Save as temp OME-TIFF so benchmark methods can reload via img_path
        tmp_tiff = tmp_dir / f"{stem}_plate_field0.ome.tiff"
        _write_benchmark_ome_tiff(tmp_tiff, images)
        img_path = tmp_tiff
        stem = _stem(tmp_tiff.name)
    else:
        data = load_image(
            img_path,
            na=na,
            refractive_index=refractive_index,
            sample_refractive_index=sample_ri,
            microscope_type=microscope_type,
            emission_wavelengths=emission_wavelengths,
            excitation_wavelengths=excitation_wavelengths,
            pinhole_airy_units=pinhole_airy_units,
            pixel_size_xy=pixel_size_xy,
            pixel_size_z=pixel_size_z,
            overrule_metadata=overrule_metadata,
        )
        meta = data["metadata"]
        images = data["images"]

    # If bench_crop, centre-crop each channel to a default tile size
    _DEFAULT_CROP = 512
    if bench_crop:
        cropped = []
        for ch in images:
            if ch.ndim == 3:
                nz, ny, nx = ch.shape
                cz = min(nz, 64)
                cy = min(ny, _DEFAULT_CROP)
                cx = min(nx, _DEFAULT_CROP)
                z0 = (nz - cz) // 2
                y0 = (ny - cy) // 2
                x0 = (nx - cx) // 2
                cropped.append(ch[z0:z0+cz, y0:y0+cy, x0:x0+cx])
            else:
                ny, nx = ch.shape
                cy = min(ny, _DEFAULT_CROP)
                cx = min(nx, _DEFAULT_CROP)
                y0 = (ny - cy) // 2
                x0 = (nx - cx) // 2
                cropped.append(ch[y0:y0+cy, x0:x0+cx])
        images = cropped
        if images[0].ndim == 3:
            meta["size_z"], meta["size_y"], meta["size_x"] = images[0].shape
        else:
            meta["size_y"], meta["size_x"] = images[0].shape
        crop_path = tmp_dir / f"{stem}_bench_crop.ome.tiff"
        _write_benchmark_ome_tiff(crop_path, images)
        img_path = crop_path
        print(f"  Cropped to: {images[0].shape}")

    all_metrics: dict[str, dict] = {}
    available_methods = list(_BENCH_METHODS)

    print(f"\n  Benchmarking {len(available_methods)} method(s)")

    common_kw = dict(
        device=device,
        na=na,
        refractive_index=refractive_index,
        sample_refractive_index=sample_ri,
        microscope_type=microscope_type,
        emission_wavelengths=emission_wavelengths,
        excitation_wavelengths=excitation_wavelengths,
        pinhole_airy_units=pinhole_airy_units,
        pixel_size_xy=pixel_size_xy,
        pixel_size_z=pixel_size_z,
        overrule_metadata=overrule_metadata,
        background=background,
        offset=offset,
        prefilter_sigma=prefilter_sigma,
        start=start,
        convergence=convergence,
        rel_threshold=rel_threshold,
        check_every=check_every,
        ri_coverslip=ri_coverslip,
        ri_coverslip_design=ri_coverslip_design,
        ri_immersion_design=ri_immersion_design,
        t_g=t_g, t_g0=t_g0, t_i0=t_i0, z_p=z_p,
    )

    for m in available_methods:
        label = f"{m}_{nit_tag}i"
        print(f"\n  -- {m}, {nit_label} iterations --")
        try:
            monitor = _MetricsMonitor()
            monitor.start()
            result = deconvolve_image(
                img_path,
                method=m,
                niter=niter_list,
                tv_lambda=tv_lambda if m == "ci_rl_tv" else 0.0,
                damping=damping if m in ("ci_rl", "ci_rl_tv") else 0.0,
                sparse_hessian_weight=sparse_hessian_weight if m == "ci_sparse_hessian" else 0.6,
                sparse_hessian_reg=sparse_hessian_reg if m == "ci_sparse_hessian" else 0.98,
                **common_kw,
            )
            out_name = f"{stem}_{m}_{nit_tag}i.ome.tiff"
            out_file = tmp_dir / out_name
            save_result(result, str(out_file), mip_only=True)
            metrics = monitor.stop()
            metrics["device"] = _method_device(m)
            
            if compute_metrics:
                # Compute image metrics only when explicitly requested; this can be slow.
                quality = _quality_metrics(result["channels"])
                metrics.update(quality)
            
            all_metrics[label] = metrics
            print(f"    {metrics['time_s']:.1f}s"
                  f"  RAM d{_format_bytes(metrics['ram_delta_peak_mb'])}"
                  f"  GPU d{_format_bytes(metrics['gpu_mem_delta_peak_mb'])}"
                  f" -> {out_name}")
            del result

        except ValueError as exc:
            print(f"    SKIPPED: {exc}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            import traceback
            traceback.print_exc()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Wait for GPU memory to settle
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                prev_used = None
                for _ in range(10):
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if prev_used is not None and mem.used == prev_used:
                        break
                    prev_used = mem.used
                    time.sleep(0.5)
            except Exception:
                time.sleep(3)
        else:
            time.sleep(2)

    # --- Metrics summary ---
    if all_metrics:
        _print_metrics_summary(all_metrics)

    # --- CSV export ---
    csv_path = out_dir / f"benchmark_metrics_{stem}.csv"
    _write_metrics_csv(csv_path, all_metrics)

    # --- Montages ---
    _make_benchmark_montage(
        str(out_dir), str(tmp_dir), stem, available_methods,
        bench_iterations, all_metrics, meta,
    )
    _make_per_channel_montages(
        str(out_dir), str(tmp_dir), stem, available_methods,
        bench_iterations, all_metrics, meta,
    )

    # Clean up tmp folder
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Cleaned up benchmark tmp folder: {tmp_dir}")
    return bool(all_metrics)


def _print_metrics_summary(all_metrics):
    """Print a formatted table of benchmark metrics to stdout."""
    hdr = (f"  {'Method':<25} {'Device':>6} {'Time':>7} {'CPU%':>6}"
           f" {'RAM pk':>8} {'RAM d':>8}"
           f" {'GPU%':>6} {'VRAM pk':>8} {'GPU d':>8}")
    sep = "  " + "-" * len(hdr.strip())
    print(f"\n{sep}")
    print(f"  Benchmark metrics summary:")
    print(f"{sep}")
    print(hdr)
    print(f"{sep}")
    for lbl, m in sorted(all_metrics.items()):
        gpu_delta = (m['torch_gpu_delta_mb']
                     if m.get('torch_gpu_delta_mb', 0) > 0
                     else m['gpu_mem_delta_peak_mb'])
        gpu_util = (
            f"{m['gpu_util_avg']:>5.0f}%"
            if m.get("gpu_util_available", 0.0) > 0
            else f"{'n/a':>6}"
        )
        print(f"  {lbl:<25} {m.get('device', '?'):>6}"
              f" {m['time_s']:>6.1f}s"
              f" {m['cpu_percent_avg']:>5.0f}%"
              f" {_format_bytes(m['ram_peak_mb']):>8}"
              f" {_format_bytes(m['ram_delta_peak_mb']):>8}"
              f" {gpu_util}"
              f" {_format_bytes(m['gpu_mem_peak_mb']):>8}"
              f" {_format_bytes(gpu_delta):>8}")
    print(f"{sep}")


def _stem(filename: str) -> str:
    """Derive a clean output stem from an image filename."""
    stem = Path(filename).stem
    for ext in (".tiff", ".tif", ".ome"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    return stem


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception:
        import traceback
        traceback.print_exc()
        print("CIDeconvolve workflow failed.")
        sys.exit(1)
