"""Streaming tile I/O for large CIDeconvolve datasets.

The existing solvers already know how to deconvolve a NumPy tile.  This module
keeps large images out of RAM by reading halo-extended regions and writing core
tiles directly to an output pyramid sink.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import numpy as np

from ._meta_helpers import (
    _DEFAULT_PINHOLE_AIRY_UNITS,
    _apply_pinhole_airy_units,
    apply_dye_wavelength_fallbacks,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileRegion:
    """A core output tile and the halo-extended input region that feeds it."""

    tile_index: int
    y_core: slice
    x_core: slice
    y_ext: slice
    x_ext: slice


class ImageRegionSource(Protocol):
    """Region-readable image source in canonical ``T, C, Z, Y, X`` layout."""

    shape: tuple[int, int, int, int, int]
    metadata: dict[str, Any]
    source_id: str

    def read_region(
        self,
        *,
        t: int,
        c: int,
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        """Return a ``Z, Y, X`` float32 region."""

    def read_pyramid_level(
        self,
        level: int,
        *,
        t: int,
        c: int,
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        """Return a lower-resolution region when the source exposes pyramids."""
        raise NotImplementedError


class PyramidSink(Protocol):
    """Incremental output sink for streamed deconvolution results."""

    shape: tuple[int, int, int, int, int]
    path: Optional[Path]

    def write_tile(
        self,
        *,
        t: int,
        c: int,
        z: slice,
        y: slice,
        x: slice,
        data: np.ndarray,
    ) -> None:
        """Write a ``Z, Y, X`` tile into level 0."""

    def mark_tile_complete(self, tile_key: str) -> None:
        """Persist tile completion state for resumable jobs."""

    def is_tile_complete(self, tile_key: str) -> bool:
        """Return True if this tile is already complete in a resumable job."""

    def build_pyramids(self) -> None:
        """Build lower-resolution pyramid levels."""

    def validate(self) -> None:
        """Raise if required output arrays or metadata are missing."""

    def close(self) -> None:
        """Flush any pending data."""


def _slice_len(slc: slice, size: int) -> int:
    start, stop, step = slc.indices(size)
    return max(0, math.ceil((stop - start) / step))


def _normalise_to_zyx(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    while data.ndim > 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim == 2:
        return data[np.newaxis, :, :]
    if data.ndim == 3:
        return data
    raise ValueError(f"Expected a YX or ZYX region, got shape {data.shape}")


def _as_solver_input(tile_zyx: np.ndarray) -> np.ndarray:
    tile = _normalise_to_zyx(tile_zyx)
    return tile[0] if tile.shape[0] == 1 else tile


def _as_zyx_result(result: np.ndarray) -> np.ndarray:
    data = np.asarray(result, dtype=np.float32)
    if data.ndim == 2:
        return data[np.newaxis, :, :]
    if data.ndim == 3:
        return data
    raise ValueError(f"Expected a YX or ZYX solver result, got shape {data.shape}")


def _copy_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    return dict(metadata or {})


def _coerce_rgb(color: Any) -> tuple[int, int, int] | None:
    if isinstance(color, str):
        text = color.strip().lstrip("#")
        if len(text) == 6:
            try:
                return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
            except ValueError:
                return None
        return None
    if isinstance(color, Sequence) and not isinstance(color, (bytes, bytearray)):
        vals = list(color)
        if len(vals) >= 3:
            try:
                return tuple(max(0, min(255, int(v))) for v in vals[:3])  # type: ignore[return-value]
            except (TypeError, ValueError):
                return None
    return None


def _rgb_to_ome_hex(color: Any) -> str | None:
    rgb = _coerce_rgb(color)
    if rgb is None:
        return None
    return "".join(f"{v:02X}" for v in rgb)


def _rgb_to_ome_int(color: Any) -> int | None:
    rgb = _coerce_rgb(color)
    if rgb is None:
        return None
    r, g, b = rgb
    return int((r << 24) | (g << 16) | (b << 8) | 255)


def _positive_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out) or out <= 0:
        return float(default)
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return value


def _apply_basic_metadata_defaults(meta: dict[str, Any], shape: tuple[int, int, int, int, int]) -> dict[str, Any]:
    t, c, z, y, x = shape
    meta.setdefault("size_t", t)
    meta.setdefault("size_c", c)
    meta.setdefault("size_z", z)
    meta.setdefault("size_y", y)
    meta.setdefault("size_x", x)
    meta.setdefault("n_channels", c)
    defaults = {
        "na": 1.4,
        "refractive_index": 1.515,
        "sample_refractive_index": 1.47,
        "microscope_type": "widefield",
        "pixel_size_x": 0.065,
        "pixel_size_y": 0.065,
        "pixel_size_z": 0.2,
    }
    defaulted = set(meta.get("_defaulted_keys", set()))
    for key, value in defaults.items():
        if meta.get(key) is None:
            meta[key] = value
            defaulted.add(key)
    channels = [dict(ch) if isinstance(ch, dict) else {} for ch in meta.get("channels", [])]
    if len(channels) < c:
        channels.extend({} for _ in range(c - len(channels)))
    names = list(meta.get("channel_names") or [])
    if len(names) < c:
        names.extend(f"Ch{i}" for i in range(len(names), c))
    meta["channel_names"] = names[:c]
    meta["channels"] = channels[:c]
    apply_dye_wavelength_fallbacks(meta, c)
    channels = [dict(ch) if isinstance(ch, dict) else {} for ch in meta.get("channels", [])]
    if len(channels) < c:
        channels.extend({} for _ in range(c - len(channels)))
    for ch in channels:
        if ch.get("emission_wavelength") is None:
            ch["emission_wavelength"] = 520.0
            defaulted.add("emission_wavelength")
    meta["channels"] = channels[:c]
    if not _apply_pinhole_airy_units(meta, _DEFAULT_PINHOLE_AIRY_UNITS, overrule_metadata=False):
        defaulted.add("pinhole_airy_units")
    meta["_defaulted_keys"] = defaulted
    return meta


def _cideconvolve_metadata_payload(metadata: dict[str, Any], shape: tuple[int, int, int, int, int]) -> dict[str, Any]:
    return {
        "creator": "CIDeconvolve",
        "shape_tczyx": list(shape),
        "metadata": _jsonable(metadata),
        "physical_pixel_sizes_um": {
            "x": _positive_float(metadata.get("pixel_size_x"), 1.0),
            "y": _positive_float(metadata.get("pixel_size_y"), _positive_float(metadata.get("pixel_size_x"), 1.0)),
            "z": _positive_float(metadata.get("pixel_size_z"), 1.0),
        },
        "channels": _jsonable(metadata.get("channels") or []),
        "processing": _jsonable(metadata.get("cideconvolve_processing") or {}),
        "source": {
            "id": metadata.get("id"),
            "name": metadata.get("name"),
            "source_id": metadata.get("source_id"),
        },
    }


class InMemoryRegionSource:
    """Region source backed by a list of channel arrays."""

    def __init__(self, channels: Sequence[np.ndarray], metadata: Optional[dict[str, Any]] = None, *, source_id: str = "memory"):
        if not channels:
            raise ValueError("InMemoryRegionSource requires at least one channel")
        arrays = [_normalise_to_zyx(np.asarray(ch, dtype=np.float32)) for ch in channels]
        z, y, x = arrays[0].shape
        for idx, arr in enumerate(arrays):
            if arr.shape != (z, y, x):
                raise ValueError(f"Channel {idx} has shape {arr.shape}, expected {(z, y, x)}")
        self._channels = arrays
        self.shape = (1, len(arrays), z, y, x)
        self.metadata = _apply_basic_metadata_defaults(_copy_metadata(metadata), self.shape)
        self.source_id = source_id

    def read_region(self, *, t: int, c: int, z: slice, y: slice, x: slice) -> np.ndarray:
        if int(t) != 0:
            raise IndexError("In-memory source has only one timepoint")
        return np.asarray(self._channels[int(c)][z, y, x], dtype=np.float32)


class BioImageRegionSource:
    """BioIO source that reads only the requested dask-backed region when possible."""

    def __init__(self, path: str | Path, *, scene: str | int | None = None):
        from bioio import BioImage

        self.path = Path(path)
        self._img = BioImage(str(path))
        if scene is not None:
            scene_value = self._img.scenes[int(scene)] if isinstance(scene, int) else scene
            self._img.set_scene(scene_value)
        self.source_id = f"bioio:{self.path}"
        size_t = int(getattr(self._img.dims, "T", 1) or 1)
        size_c = int(getattr(self._img.dims, "C", 1) or 1)
        size_z = int(getattr(self._img.dims, "Z", 1) or 1)
        size_y = int(getattr(self._img.dims, "Y", 1) or 1)
        size_x = int(getattr(self._img.dims, "X", 1) or 1)
        self.shape = (size_t, size_c, size_z, size_y, size_x)
        meta: dict[str, Any] = {
            "size_t": size_t,
            "size_c": size_c,
            "size_z": size_z,
            "size_y": size_y,
            "size_x": size_x,
            "n_channels": size_c,
            "channel_names": list(getattr(self._img, "channel_names", []) or []),
        }
        try:
            pps = self._img.physical_pixel_sizes
            meta["pixel_size_x"] = getattr(pps, "X", None)
            meta["pixel_size_y"] = getattr(pps, "Y", None)
            meta["pixel_size_z"] = getattr(pps, "Z", None)
        except Exception:
            pass
        self.metadata = _apply_basic_metadata_defaults(meta, self.shape)

    def read_region(self, *, t: int, c: int, z: slice, y: slice, x: slice) -> np.ndarray:
        selector: dict[str, Any] = {"T": int(t), "C": int(c)}
        if hasattr(self._img, "get_image_dask_data"):
            arr = self._img.get_image_dask_data("ZYX", **selector)
            return np.asarray(arr[z, y, x].compute(), dtype=np.float32)
        arr = self._img.get_image_data("ZYX", **selector)
        return np.asarray(arr[z, y, x], dtype=np.float32)


class ZarrRegionSource:
    """Region source for OME-Zarr multiscales or HCS plate fields."""

    def __init__(
        self,
        path: str | Path,
        *,
        array_path: str = "0",
        hcs_field: str | None = None,
    ):
        import zarr

        self.path = Path(path)
        self._root = zarr.open(str(self.path), mode="r")
        self._group = self._resolve_group(hcs_field)
        self._array_path = str(array_path)
        self._array = self._group[self._array_path]
        self.source_id = f"zarr:{self.path}:{self._group.path}/{self._array_path}"
        self._layout = self._infer_layout(tuple(int(v) for v in self._array.shape))
        self.shape = self._layout["shape_tczyx"]
        self.metadata = _apply_basic_metadata_defaults(self._metadata_from_attrs(), self.shape)

    def _resolve_group(self, hcs_field: str | None):
        root_attrs = dict(self._root.attrs)
        if "plate" not in root_attrs:
            return self._root
        if hcs_field is None:
            plate = root_attrs.get("plate", {})
            for well in plate.get("wells", []):
                well_path = str(well.get("path", ""))
                if not well_path:
                    continue
                well_group = self._root[well_path]
                well_attrs = dict(well_group.attrs)
                for image in well_attrs.get("well", {}).get("images", []) or []:
                    field = str(image.get("path", ""))
                    if field:
                        return self._root[f"{well_path}/{field}"]
                for key in sorted(well_group.keys()):
                    if str(key).isdigit():
                        return self._root[f"{well_path}/{key}"]
            raise ValueError(f"No HCS fields found in {self.path}")
        return self._root[hcs_field.strip("/")]

    @staticmethod
    def _infer_layout(shape: tuple[int, ...]) -> dict[str, Any]:
        if len(shape) == 5:
            return {"kind": "TCZYX", "shape_tczyx": shape}
        if len(shape) == 4:
            c, z, y, x = shape
            return {"kind": "CZYX", "shape_tczyx": (1, c, z, y, x)}
        if len(shape) == 3:
            c, y, x = shape
            return {"kind": "CYX", "shape_tczyx": (1, c, 1, y, x)}
        if len(shape) == 2:
            y, x = shape
            return {"kind": "YX", "shape_tczyx": (1, 1, 1, y, x)}
        raise ValueError(f"Unsupported Zarr array shape {shape}; expected TCZYX, CZYX, CYX, or YX")

    def _metadata_from_attrs(self) -> dict[str, Any]:
        attrs = dict(self._group.attrs)
        t, c, z, y, x = self.shape
        meta: dict[str, Any] = {
            "size_t": t,
            "size_c": c,
            "size_z": z,
            "size_y": y,
            "size_x": x,
            "n_channels": c,
        }
        multiscales = attrs.get("multiscales", [])
        if multiscales:
            datasets = multiscales[0].get("datasets", [])
            if datasets:
                for transform in datasets[0].get("coordinateTransformations", []):
                    if transform.get("type") != "scale":
                        continue
                    scale = transform.get("scale", [])
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
        omero_channels = attrs.get("omero", {}).get("channels", [])
        if omero_channels:
            meta["channel_names"] = [
                str(ch.get("label") or f"Ch{i}") for i, ch in enumerate(omero_channels)
            ]
            channels = []
            for i, ch in enumerate(omero_channels):
                window = ch.get("window") or {}
                info = {
                    "name": str(ch.get("label") or f"Ch{i}"),
                    "color": _coerce_rgb(ch.get("color")),
                    "active": bool(ch.get("active", True)),
                }
                if window.get("start") is not None:
                    info["window_start"] = window.get("start")
                if window.get("end") is not None:
                    info["window_end"] = window.get("end")
                channels.append(info)
            meta["channels"] = channels
        return meta

    def read_region(self, *, t: int, c: int, z: slice, y: slice, x: slice) -> np.ndarray:
        kind = self._layout["kind"]
        if kind == "TCZYX":
            data = self._array[int(t), int(c), z, y, x]
        elif kind == "CZYX":
            data = self._array[int(c), z, y, x]
        elif kind == "CYX":
            data = self._array[int(c), y, x][np.newaxis, :, :]
        elif kind == "YX":
            data = self._array[y, x][np.newaxis, :, :]
        else:  # pragma: no cover - protected by _infer_layout
            raise ValueError(f"Unsupported layout {kind}")
        return np.asarray(data, dtype=np.float32)

    def read_pyramid_level(self, level: int, *, t: int, c: int, z: slice, y: slice, x: slice) -> np.ndarray:
        array = self._group[str(int(level))]
        old_array, self._array = self._array, array
        old_layout, self._layout = self._layout, self._infer_layout(tuple(int(v) for v in array.shape))
        try:
            return self.read_region(t=t, c=c, z=z, y=y, x=x)
        finally:
            self._array = old_array
            self._layout = old_layout


class OmeZarrRegionSource:
    """Region source backed by the official ome-zarr-py reader."""

    def __init__(self, path: str | Path, *, array_path: str = "0"):
        from core.ome_zarr_io import open_ome_zarr_image_node

        self.path, self._node = open_ome_zarr_image_node(path)
        self._level = int(array_path)
        self._levels = self._node.data
        self._array = self._levels[self._level]
        self.source_id = f"ome-zarr:{self.path}:{self._level}"
        self._layout = ZarrRegionSource._infer_layout(tuple(int(v) for v in self._array.shape))
        self.shape = self._layout["shape_tczyx"]
        self.metadata = _apply_basic_metadata_defaults(self._metadata_from_node(), self.shape)

    def _metadata_from_node(self) -> dict[str, Any]:
        t, c, z, y, x = self.shape
        meta: dict[str, Any] = {
            "size_t": t,
            "size_c": c,
            "size_z": z,
            "size_y": y,
            "size_x": x,
            "n_channels": c,
        }
        node_meta = dict(getattr(self._node, "metadata", {}) or {})
        if node_meta.get("name"):
            meta["name"] = node_meta.get("name")
        transforms = node_meta.get("coordinateTransformations") or []
        if transforms and isinstance(transforms[0], list):
            for transform in transforms[0]:
                if transform.get("type") != "scale":
                    continue
                scale = transform.get("scale", [])
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
        return meta

    def read_region(self, *, t: int, c: int, z: slice, y: slice, x: slice) -> np.ndarray:
        kind = self._layout["kind"]
        if kind == "TCZYX":
            data = self._array[int(t), int(c), z, y, x]
        elif kind == "CZYX":
            data = self._array[int(c), z, y, x]
        elif kind == "CYX":
            data = self._array[int(c), y, x][np.newaxis, :, :]
        elif kind == "YX":
            data = self._array[y, x][np.newaxis, :, :]
        else:
            raise ValueError(f"Unsupported OME-Zarr layout {kind}")
        return np.asarray(data.compute(), dtype=np.float32)

    def read_pyramid_level(self, level: int, *, t: int, c: int, z: slice, y: slice, x: slice) -> np.ndarray:
        old_array = self._array
        old_layout = self._layout
        self._array = self._levels[int(level)]
        self._layout = ZarrRegionSource._infer_layout(tuple(int(v) for v in self._array.shape))
        try:
            return self.read_region(t=t, c=c, z=z, y=y, x=x)
        finally:
            self._array = old_array
            self._layout = old_layout


def open_region_source(
    path: str | Path,
    *,
    scene: str | int | None = None,
    hcs_field: str | None = None,
) -> ImageRegionSource:
    """Open a source with region reads, preferring native Zarr access."""
    path = Path(path)
    if path.is_dir() and path.suffix.lower() == ".zarr":
        if hcs_field is not None:
            return ZarrRegionSource(path, hcs_field=hcs_field)
        try:
            import zarr
            root = zarr.open(str(path), mode="r")
            if "plate" in dict(root.attrs):
                return ZarrRegionSource(path, hcs_field=hcs_field)
        except Exception:
            pass
        return OmeZarrRegionSource(path)
    return BioImageRegionSource(path, scene=scene)


class InMemoryPyramidSink:
    """Small test sink that stores level 0 in memory."""

    path: Optional[Path] = None

    def __init__(self, shape: tuple[int, int, int, int, int], metadata: Optional[dict[str, Any]] = None):
        self.shape = tuple(int(v) for v in shape)
        self.metadata = _apply_basic_metadata_defaults(_copy_metadata(metadata), self.shape)
        self.data = np.zeros(self.shape, dtype=np.float32)
        self.completed_tiles: set[str] = set()
        self.pyramids: list[np.ndarray] = []

    def write_tile(self, *, t: int, c: int, z: slice, y: slice, x: slice, data: np.ndarray) -> None:
        self.data[int(t), int(c), z, y, x] = _normalise_to_zyx(data)

    def mark_tile_complete(self, tile_key: str) -> None:
        self.completed_tiles.add(tile_key)

    def is_tile_complete(self, tile_key: str) -> bool:
        return tile_key in self.completed_tiles

    def build_pyramids(self) -> None:
        current = self.data
        self.pyramids = []
        while min(current.shape[-2:]) > 1 and len(self.pyramids) < 8:
            current = _downsample_2x_xy(current)
            self.pyramids.append(current)

    def validate(self) -> None:
        if self.data.shape != self.shape:
            raise ValueError(f"Sink shape {self.data.shape} does not match expected {self.shape}")

    def close(self) -> None:
        pass


class ZarrPyramidSink:
    """OME-Zarr 0.5 multiscale sink with resumable tile manifest."""

    def __init__(
        self,
        path: str | Path,
        *,
        shape: tuple[int, int, int, int, int],
        metadata: Optional[dict[str, Any]] = None,
        chunks: tuple[int, int, int, int, int] | None = None,
        levels: int | None = None,
        zarr_format: int = 2,
        resume: bool = True,
    ):
        import zarr

        self.path = Path(path)
        self.shape = tuple(int(v) for v in shape)
        self.metadata = _apply_basic_metadata_defaults(_copy_metadata(metadata), self.shape)
        self.levels = int(levels) if levels is not None else _default_pyramid_levels(self.shape[-2:])
        self.zarr_format = int(zarr_format)
        self.resume = bool(resume)
        self.path.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.path / ".cideconvolve_stream_manifest.json"
        self._completed = self._load_manifest()
        try:
            self._root = zarr.open(str(self.path), mode="a", zarr_format=self.zarr_format)
        except TypeError:
            self._root = zarr.open(str(self.path), mode="a")
        self._chunks = chunks or (
            1,
            1,
            max(1, min(self.shape[2], 16)),
            min(512, self.shape[3]),
            min(512, self.shape[4]),
        )
        self._compressor_kwargs = self._compressor_args()
        self._level0 = self._require_array("0", self.shape, self._chunks)
        self._write_metadata_stub()

    def _compressor_args(self) -> dict[str, Any]:
        if self.zarr_format == 2:
            try:
                from numcodecs import Blosc
                return {"compressor": Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)}
            except Exception:
                return {}
        try:
            from zarr.codecs import BloscCodec
            return {"compressors": [BloscCodec(cname="zstd", clevel=5)]}
        except Exception:
            return {}

    def _require_array(self, name: str, shape: tuple[int, ...], chunks: tuple[int, ...]):
        if name in self._root:
            arr = self._root[name]
            if tuple(arr.shape) != tuple(shape):
                raise ValueError(f"Existing Zarr level {name} has shape {arr.shape}, expected {shape}")
            return arr
        if hasattr(self._root, "create_array"):
            return self._root.create_array(
                name,
                shape=shape,
                chunks=chunks,
                dtype="float32",
                overwrite=False,
                **self._compressor_kwargs,
            )
        return self._root.create_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype="float32",
            overwrite=False,
            **self._compressor_kwargs,
        )

    def _load_manifest(self) -> set[str]:
        if not self.resume or not self._manifest_path.exists():
            return set()
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return set(str(v) for v in data.get("completed_tiles", []))
        except Exception:
            return set()

    def _save_manifest(self) -> None:
        payload = {
            "shape": list(self.shape),
            "completed_tiles": sorted(self._completed),
        }
        self._manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_tile(self, *, t: int, c: int, z: slice, y: slice, x: slice, data: np.ndarray) -> None:
        self._level0[int(t), int(c), z, y, x] = _normalise_to_zyx(data)

    def mark_tile_complete(self, tile_key: str) -> None:
        self._completed.add(str(tile_key))
        self._save_manifest()

    def is_tile_complete(self, tile_key: str) -> bool:
        return str(tile_key) in self._completed

    def _write_metadata_stub(self) -> None:
        px_x = _positive_float(self.metadata.get("pixel_size_x"), 1.0)
        px_y = _positive_float(self.metadata.get("pixel_size_y"), px_x)
        px_z = _positive_float(self.metadata.get("pixel_size_z"), 1.0)
        datasets = []
        for level in range(max(self.levels, 1)):
            scale_factor = 2 ** level
            datasets.append({
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1, 1, px_z, px_y * scale_factor, px_x * scale_factor],
                    }
                ],
            })
        self._root.attrs["multiscales"] = [{
            "version": "0.4" if self.zarr_format == 2 else "0.5",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
            "name": self.path.name if self.path is not None else "CIDeconvolve",
            "type": "mean",
            "metadata": {"method": "chunked 2x2 XY mean"},
        }]
        channel_names = list(self.metadata.get("channel_names") or [])
        source_channels = [dict(ch) if isinstance(ch, dict) else {} for ch in self.metadata.get("channels", [])]
        channels = []
        for i in range(self.shape[1]):
            src = source_channels[i] if i < len(source_channels) else {}
            label = src.get("name") or src.get("label") or (channel_names[i] if i < len(channel_names) else f"Ch{i}")
            entry: dict[str, Any] = {"label": str(label)}
            color_hex = _rgb_to_ome_hex(src.get("color"))
            if color_hex is not None:
                entry["color"] = color_hex
            else:
                entry["color"] = "FFFFFF"
            entry["active"] = bool(src.get("active", True))
            entry["coefficient"] = 1
            entry["family"] = "linear"
            entry["inverted"] = False
            window_start = src.get("window_start")
            window_end = src.get("window_end")
            start = float(window_start if window_start is not None else 0.0)
            end = float(window_end if window_end is not None else max(start, 1.0))
            entry["window"] = {
                "start": start,
                "end": end,
                "min": min(start, 0.0),
                "max": max(end, 1.0),
            }
            channels.append(entry)
        omero: dict[str, Any] = {
            "channels": channels,
            "name": str(self.metadata.get("name") or (self.path.name if self.path is not None else "CIDeconvolve")),
            "rdefs": {
                "defaultT": int(self.metadata.get("default_t", 0) or 0),
                "defaultZ": int(self.metadata.get("default_z", 0) or 0),
                "model": "color",
            },
        }
        image_id = self.metadata.get("id")
        try:
            if image_id is not None:
                omero["id"] = int(image_id)
        except (TypeError, ValueError):
            pass
        self._root.attrs["omero"] = omero
        payload = _cideconvolve_metadata_payload(self.metadata, self.shape)
        payload["streaming"] = True
        self._root.attrs["_creator"] = payload
        self._root.attrs["cideconvolve"] = payload

    def build_pyramids(self) -> None:
        current = self._level0
        for level in range(1, max(self.levels, 1)):
            shape = (
                self.shape[0],
                self.shape[1],
                self.shape[2],
                max(1, math.ceil(current.shape[-2] / 2)),
                max(1, math.ceil(current.shape[-1] / 2)),
            )
            chunks = (
                1,
                1,
                max(1, min(shape[2], self._chunks[2])),
                min(512, shape[3]),
                min(512, shape[4]),
            )
            out = self._require_array(str(level), shape, chunks)
            _downsample_array_2x_xy(current, out)
            current = out
        self._write_metadata_stub()

    def validate(self) -> None:
        if "0" not in self._root:
            raise ValueError("OME-Zarr output is missing level 0")
        if tuple(self._root["0"].shape) != self.shape:
            raise ValueError(f"OME-Zarr level 0 has shape {self._root['0'].shape}, expected {self.shape}")
        if "multiscales" not in self._root.attrs:
            raise ValueError("OME-Zarr output is missing multiscales metadata")

    def close(self) -> None:
        self._save_manifest()


class TiledOmeTiffSink:
    """Tiled BigTIFF sink backed by temporary chunked level-0 staging.

    TIFF needs planes written in a stable order, while streamed deconvolution
    writes arbitrary XY tiles.  This sink stages level 0 in an on-disk memmap,
    optionally builds XY pyramid levels in additional memmaps, then writes a
    tiled BigTIFF with SubIFD pyramid levels on close.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        shape: tuple[int, int, int, int, int],
        metadata: Optional[dict[str, Any]] = None,
        tile_yx: tuple[int, int] = (512, 512),
        levels: int | None = None,
        compression: str | None = "lzw",
        temp_dir: str | Path | None = None,
    ):
        self.path = Path(path)
        self.shape = tuple(int(v) for v in shape)
        self.metadata = _apply_basic_metadata_defaults(_copy_metadata(metadata), self.shape)
        self.tile_yx = (max(16, int(tile_yx[0])), max(16, int(tile_yx[1])))
        self.levels = int(levels) if levels is not None else _default_pyramid_levels(self.shape[-2:])
        self.compression = compression
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_root = Path(temp_dir) if temp_dir is not None else None
        self._tmpdir = Path(tempfile.mkdtemp(prefix="cideconv_tiff_", dir=str(self._tmp_root) if self._tmp_root else None))
        self._level0_path = self._tmpdir / "level0.dat"
        self._level0 = np.memmap(self._level0_path, dtype="float32", mode="w+", shape=self.shape)
        self._completed: set[str] = set()
        self._pyramids: list[np.memmap] = []
        self._closed = False

    @staticmethod
    def _tile_dim(requested: int, extent: int) -> int:
        """Return a TIFF tile side length accepted by tifffile.

        TIFF tile dimensions must be multiples of 16.  For small images we use
        the smallest 16-aligned tile that covers the image side; for larger
        images we keep the requested tile size.
        """
        requested = max(16, int(requested))
        extent = max(1, int(extent))
        rounded_extent = max(16, int(math.ceil(extent / 16.0) * 16))
        return max(16, min(requested, rounded_extent))

    def write_tile(self, *, t: int, c: int, z: slice, y: slice, x: slice, data: np.ndarray) -> None:
        self._level0[int(t), int(c), z, y, x] = _normalise_to_zyx(data)

    def mark_tile_complete(self, tile_key: str) -> None:
        self._completed.add(str(tile_key))

    def is_tile_complete(self, tile_key: str) -> bool:
        return str(tile_key) in self._completed

    def build_pyramids(self) -> None:
        current = self._level0
        self._pyramids = []
        for level in range(1, max(self.levels, 1)):
            shape = (
                self.shape[0],
                self.shape[1],
                self.shape[2],
                max(1, math.ceil(int(current.shape[-2]) / 2)),
                max(1, math.ceil(int(current.shape[-1]) / 2)),
            )
            level_path = self._tmpdir / f"level{level}.dat"
            out = np.memmap(level_path, dtype="float32", mode="w+", shape=shape)
            _downsample_array_2x_xy(current, out)
            out.flush()
            self._pyramids.append(out)
            current = out

    def validate(self) -> None:
        if tuple(self._level0.shape) != self.shape:
            raise ValueError(f"BigTIFF staging shape {self._level0.shape} does not match expected {self.shape}")

    def _ome_metadata(self) -> dict[str, Any]:
        px_x = _positive_float(self.metadata.get("pixel_size_x"), 1.0)
        px_y = _positive_float(self.metadata.get("pixel_size_y"), px_x)
        px_z = _positive_float(self.metadata.get("pixel_size_z"), 1.0)
        raw_names = self.metadata.get("channel_names") or []
        names = list(raw_names) if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes, bytearray)) else []
        raw_channels = self.metadata.get("channels", [])
        if isinstance(raw_channels, dict):
            raw_channels = [raw_channels]
        elif not isinstance(raw_channels, Sequence) or isinstance(raw_channels, (str, bytes, bytearray)):
            raw_channels = []
        channels = [dict(ch) if isinstance(ch, dict) else {} for ch in raw_channels]
        channel_names = []
        channel_colors = []
        emission_wavelengths = []
        emission_units = []
        excitation_wavelengths = []
        excitation_units = []
        pinhole_sizes = []
        pinhole_units = []
        for i in range(self.shape[1]):
            ch = channels[i] if i < len(channels) else {}
            channel_names.append(str(ch.get("name") or ch.get("label") or (names[i] if i < len(names) else f"Ch{i}")))
            channel_colors.append(_rgb_to_ome_int(ch.get("color")))
            emission_wavelengths.append(ch.get("emission_wavelength"))
            emission_units.append("nm")
            excitation_wavelengths.append(ch.get("excitation_wavelength"))
            excitation_units.append("nm")
            pinhole_sizes.append(ch.get("pinhole_size_um") or ch.get("pinhole_size"))
            pinhole_units.append("µm")
        channel_meta: dict[str, Any] = {"Name": channel_names}
        if any(value is not None for value in channel_colors):
            channel_meta["Color"] = [value if value is not None else 0xFFFFFFFF for value in channel_colors]
        if emission_wavelengths and all(value is not None for value in emission_wavelengths):
            channel_meta["EmissionWavelength"] = [
                float(value) for value in emission_wavelengths
            ]
            channel_meta["EmissionWavelengthUnit"] = emission_units
        if excitation_wavelengths and all(value is not None for value in excitation_wavelengths):
            channel_meta["ExcitationWavelength"] = [
                float(value) for value in excitation_wavelengths
            ]
            channel_meta["ExcitationWavelengthUnit"] = excitation_units
        if pinhole_sizes and all(value is not None for value in pinhole_sizes):
            channel_meta["PinholeSize"] = [
                float(value) for value in pinhole_sizes
            ]
            channel_meta["PinholeSizeUnit"] = pinhole_units
        return {
            "Name": str(self.metadata.get("name") or self.path.stem),
            "axes": "TCZYX",
            "PhysicalSizeX": px_x,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": px_y,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": px_z,
            "PhysicalSizeZUnit": "µm",
            "Channel": channel_meta,
        }

    def _description_metadata(self) -> str:
        return json.dumps(_cideconvolve_metadata_payload(self.metadata, self.shape), default=str)

    def _write_tiff_once(self, compression: str | None) -> None:
        try:
            import tifffile
        except Exception as exc:  # pragma: no cover - depends on optional GUI dependency
            raise RuntimeError("Writing OME-TIFF/BigTIFF output requires tifffile") from exc

        if self.path.exists():
            self.path.unlink()
        self._level0.flush()
        for pyramid in self._pyramids:
            pyramid.flush()

        tile = (
            self._tile_dim(self.tile_yx[0], int(self.shape[-2])),
            self._tile_dim(self.tile_yx[1], int(self.shape[-1])),
        )
        write_kwargs = {
            "photometric": "minisblack",
            "tile": tile,
            "metadata": self._ome_metadata(),
            "extratags": [(65000, "s", 0, self._description_metadata(), True)],
        }
        if self._pyramids:
            write_kwargs["subifds"] = len(self._pyramids)
        if compression:
            write_kwargs["compression"] = compression
            write_kwargs["predictor"] = True

        with tifffile.TiffWriter(str(self.path), bigtiff=True, ome=True) as tif:
            tif.write(self._level0, **write_kwargs)
            for pyramid in self._pyramids:
                level_tile = (
                    self._tile_dim(self.tile_yx[0], int(pyramid.shape[-2])),
                    self._tile_dim(self.tile_yx[1], int(pyramid.shape[-1])),
                )
                level_kwargs = {
                    "photometric": "minisblack",
                    "tile": level_tile,
                    "subfiletype": 1,
                }
                if compression:
                    level_kwargs["compression"] = compression
                    level_kwargs["predictor"] = True
                tif.write(pyramid, **level_kwargs)

    def _write_tiff(self) -> None:
        try:
            self._write_tiff_once(self.compression)
        except Exception as exc:
            if not self.compression:
                raise
            log.warning(
                "Could not write compressed OME-TIFF with %s compression; writing uncompressed fallback: %s",
                self.compression,
                exc,
            )
            self._write_tiff_once(None)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._write_tiff()
        finally:
            self._closed = True
            try:
                self._level0._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            for pyramid in self._pyramids:
                try:
                    pyramid._mmap.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def abort(self) -> None:
        """Drop staged data without writing a partial TIFF."""
        if self._closed:
            return
        self._closed = True
        try:
            self._level0._mmap.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        for pyramid in self._pyramids:
            try:
                pyramid._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class ProjectionPyramidSink:
    """Project ZYX tiles over Z before writing to an inner Z=1 pyramid sink."""

    def __init__(
        self,
        inner: PyramidSink,
        *,
        source_shape: tuple[int, int, int, int, int],
        mode: str = "mip",
    ):
        self.inner = inner
        self.shape = tuple(int(v) for v in source_shape)
        self.path = getattr(inner, "path", None)
        self.mode = str(mode or "mip").strip().lower()
        if self.mode not in {"mip", "max", "sum", "mean"}:
            raise ValueError(f"Unsupported projection mode {mode!r}; expected mip, sum, or mean")
        inner_shape = tuple(int(v) for v in getattr(inner, "shape"))
        expected_inner = (self.shape[0], self.shape[1], 1, self.shape[3], self.shape[4])
        if inner_shape != expected_inner:
            raise ValueError(f"Projection inner sink shape {inner_shape} does not match expected {expected_inner}")

    def _project(self, data: np.ndarray) -> np.ndarray:
        arr = _normalise_to_zyx(data)
        if arr.shape[0] == 1:
            return arr
        if self.mode in {"mip", "max"}:
            out = np.max(arr, axis=0)
        elif self.mode == "sum":
            out = np.sum(arr, axis=0)
        else:
            out = np.mean(arr, axis=0)
        return out.astype(np.float32, copy=False)[np.newaxis, :, :]

    def write_tile(self, *, t: int, c: int, z: slice, y: slice, x: slice, data: np.ndarray) -> None:
        self.inner.write_tile(
            t=int(t),
            c=int(c),
            z=slice(0, 1),
            y=y,
            x=x,
            data=self._project(data),
        )

    def mark_tile_complete(self, tile_key: str) -> None:
        self.inner.mark_tile_complete(tile_key)

    def is_tile_complete(self, tile_key: str) -> bool:
        return self.inner.is_tile_complete(tile_key)

    def build_pyramids(self) -> None:
        self.inner.build_pyramids()

    def validate(self) -> None:
        self.inner.validate()

    def close(self) -> None:
        self.inner.close()

    def abort(self) -> None:
        abort = getattr(self.inner, "abort", None)
        if callable(abort):
            abort()


def _default_pyramid_levels(shape_yx: tuple[int, int]) -> int:
    y, x = (int(shape_yx[0]), int(shape_yx[1]))
    levels = 1
    while max(y, x) > 512 and levels < 8:
        y = max(1, math.ceil(y / 2))
        x = max(1, math.ceil(x / 2))
        levels += 1
    return levels


def _downsample_2x_xy(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    y, x = arr.shape[-2:]
    if y == 1 and x == 1:
        return arr.copy()
    pad_y = y % 2
    pad_x = x % 2
    if pad_y or pad_x:
        arr = np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(0, pad_y), (0, pad_x)], mode="edge")
        y, x = arr.shape[-2:]
    return arr.reshape(*arr.shape[:-2], y // 2, 2, x // 2, 2).mean(axis=(-3, -1)).astype(np.float32)


def _downsample_array_2x_xy(src, dst, *, y_chunk: int = 512, x_chunk: int = 512) -> None:
    out_y, out_x = int(dst.shape[-2]), int(dst.shape[-1])
    for y0 in range(0, out_y, y_chunk):
        y1 = min(out_y, y0 + y_chunk)
        src_y = slice(y0 * 2, min(int(src.shape[-2]), y1 * 2))
        for x0 in range(0, out_x, x_chunk):
            x1 = min(out_x, x0 + x_chunk)
            src_x = slice(x0 * 2, min(int(src.shape[-1]), x1 * 2))
            block = np.asarray(src[..., src_y, src_x], dtype=np.float32)
            down = _downsample_2x_xy(block)
            dst[..., y0:y1, x0:x1] = down[..., : y1 - y0, : x1 - x0]


def compute_tile_regions(shape_zyx: tuple[int, int, int], tile_yx: tuple[int, int], halo_yx: tuple[int, int]) -> list[TileRegion]:
    """Return halo-extended XY tile regions for a ``Z, Y, X`` image shape."""
    _, size_y, size_x = (int(v) for v in shape_zyx)
    tile_y = max(int(tile_yx[0]), 1)
    tile_x = max(int(tile_yx[1]), 1)
    halo_y = max(int(halo_yx[0]), 0)
    halo_x = max(int(halo_yx[1]), 0)
    regions: list[TileRegion] = []
    idx = 0
    for y0 in range(0, size_y, tile_y):
        y1 = min(size_y, y0 + tile_y)
        for x0 in range(0, size_x, tile_x):
            x1 = min(size_x, x0 + tile_x)
            regions.append(TileRegion(
                tile_index=idx,
                y_core=slice(y0, y1),
                x_core=slice(x0, x1),
                y_ext=slice(max(0, y0 - halo_y), min(size_y, y1 + halo_y)),
                x_ext=slice(max(0, x0 - halo_x), min(size_x, x1 + halo_x)),
            ))
            idx += 1
    return regions


def should_stream_source(
    shape_tczyx: tuple[int, int, int, int, int],
    *,
    threshold_gb: float = 2.0,
    bytes_per_voxel: int = 4,
) -> bool:
    """Return True when eager loading the source would exceed the threshold."""
    total = int(np.prod(shape_tczyx)) * int(bytes_per_voxel)
    return total >= float(threshold_gb) * (1024 ** 3)


def streaming_memory_budget_bytes(
    *,
    device: str | None = None,
    vram_fraction: float = 0.50,
) -> int:
    """Return a practical memory budget for one streamed solver tile."""
    try:
        import torch
        dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if dev.type == "cuda":
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            try:
                free, total = torch.cuda.mem_get_info(idx)
                return int(max(512 * 1024 ** 2, min(total * vram_fraction, free * 0.80)))
            except Exception:
                total = torch.cuda.get_device_properties(idx).total_memory
                return int(total * vram_fraction)
    except Exception:
        pass
    try:
        import psutil
        available = psutil.virtual_memory().available
        return int(min(available * 0.50, 16 * 1024 ** 3))
    except Exception:
        return int(4 * 1024 ** 3)


def suggest_streaming_tile_size(
    shape_tczyx: tuple[int, int, int, int, int],
    *,
    psf_xy_est: int = 65,
    method: str = "ci_rl",
    device: str | None = None,
    memory_budget_bytes: int | None = None,
    max_tile_xy: int = 4096,
    multiple: int = 64,
) -> int:
    """Suggest an XY core tile size for streamed deconvolution.

    The estimate is intentionally based on the solver work volume, not only the
    input tile size.  Streaming crops oversized 3D PSFs to the tile Z depth, so
    the dominant work volume is roughly ``(2*Z - 1) * work_y * work_x``.
    """
    _, _, size_z, size_y, size_x = (int(v) for v in shape_tczyx)
    max_image_xy = max(size_y, size_x, 1)
    if size_z <= 1:
        return min(max_image_xy, max_tile_xy)

    budget = int(memory_budget_bytes) if memory_budget_bytes is not None else streaming_memory_budget_bytes(device=device)
    method_key = str(method or "ci_rl").strip().lower()
    bytes_per_work_voxel = {
        "ci_rl": 96.0,
        "ci_rl_tv": 112.0,
        "ci_sparse_hessian": 128.0,
    }.get(method_key, 48.0)
    work_z = max(1, 2 * size_z - 1)
    psf_xy = max(16, int(psf_xy_est))
    halo = max(psf_xy // 2, 16)
    allowed_work_xy = math.sqrt(max(float(budget), 1.0) / max(bytes_per_work_voxel * work_z, 1.0))
    tile_xy = int(allowed_work_xy - (2 * halo) - psf_xy + 1)
    tile_xy = max(256, min(tile_xy, max_image_xy, int(max_tile_xy)))
    multiple = max(1, int(multiple))
    if tile_xy < max_image_xy:
        tile_xy = max(256, (tile_xy // multiple) * multiple)
    return max(1, int(tile_xy))


def _tile_key(t: int, c: int, tile: TileRegion) -> str:
    return (
        f"t{int(t):04d}_c{int(c):04d}_"
        f"y{tile.y_core.start}-{tile.y_core.stop}_x{tile.x_core.start}-{tile.x_core.stop}"
    )


def deconvolve_streaming(
    source: ImageRegionSource,
    sink: PyramidSink,
    *,
    psf_for_channel: Callable[[int], np.ndarray],
    deconvolve_tile: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    channels: Optional[Sequence[int]] = None,
    timepoints: Optional[Sequence[int]] = None,
    tile_yx: tuple[int, int] = (1024, 1024),
    halo_yx: Optional[tuple[int, int]] = None,
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
    resume: bool = True,
    build_pyramids: bool = True,
) -> dict[str, Any]:
    """Stream deconvolution over source XY tiles and write directly to *sink*.

    ``deconvolve_tile`` receives ``(tile_image, effective_psf, channel_index)``
    and must return a YX or ZYX float array.
    """
    shape = tuple(int(v) for v in source.shape)
    if tuple(sink.shape) != shape:
        raise ValueError(f"Sink shape {sink.shape} does not match source shape {shape}")
    size_t, size_c, size_z, size_y, size_x = shape
    selected_t = list(timepoints) if timepoints is not None else list(range(size_t))
    selected_c = list(channels) if channels is not None else list(range(size_c))
    z_slice = slice(0, size_z)

    psfs: dict[int, np.ndarray] = {}
    max_psf_yx = 0
    for c in selected_c:
        psf = np.asarray(psf_for_channel(int(c)), dtype=np.float32)
        psfs[int(c)] = psf
        if psf.ndim >= 2:
            max_psf_yx = max(max_psf_yx, int(psf.shape[-2]), int(psf.shape[-1]))
    if halo_yx is None:
        halo = max(max_psf_yx // 2, 16)
        halo_yx = (halo, halo)

    regions = compute_tile_regions((size_z, size_y, size_x), tile_yx, halo_yx)
    total_tiles = len(selected_t) * len(selected_c) * len(regions)
    done = 0
    started = 0
    for t in selected_t:
        for c in selected_c:
            psf = psfs[int(c)]
            for tile in regions:
                key = _tile_key(int(t), int(c), tile)
                if resume and sink.is_tile_complete(key):
                    done += 1
                    continue
                started += 1
                if progress is not None:
                    progress({
                        "event": "tile_start",
                        "done": done,
                        "total": total_tiles,
                        "timepoint": int(t),
                        "channel": int(c),
                        "tile_index": tile.tile_index,
                        "tile_count": len(regions),
                        "core": (tile.y_core.start, tile.y_core.stop, tile.x_core.start, tile.x_core.stop),
                    })
                tile_data = source.read_region(
                    t=int(t),
                    c=int(c),
                    z=z_slice,
                    y=tile.y_ext,
                    x=tile.x_ext,
                )
                result = _as_zyx_result(deconvolve_tile(_as_solver_input(tile_data), psf, int(c)))
                crop_y0 = tile.y_core.start - tile.y_ext.start
                crop_y1 = crop_y0 + _slice_len(tile.y_core, size_y)
                crop_x0 = tile.x_core.start - tile.x_ext.start
                crop_x1 = crop_x0 + _slice_len(tile.x_core, size_x)
                core = result[:, crop_y0:crop_y1, crop_x0:crop_x1]
                sink.write_tile(
                    t=int(t),
                    c=int(c),
                    z=z_slice,
                    y=tile.y_core,
                    x=tile.x_core,
                    data=core,
                )
                sink.mark_tile_complete(key)
                done += 1
                if progress is not None:
                    progress({
                        "event": "tile_done",
                        "done": done,
                        "total": total_tiles,
                        "timepoint": int(t),
                        "channel": int(c),
                        "tile_index": tile.tile_index,
                        "tile_count": len(regions),
                    })

    if build_pyramids:
        if progress is not None:
            progress({"event": "pyramid_start", "done": done, "total": total_tiles})
        sink.build_pyramids()
    sink.validate()
    sink.close()
    return {
        "tiles_total": total_tiles,
        "tiles_processed": started,
        "tiles_completed": done,
        "tile_regions": len(regions),
        "tile_yx": [int(tile_yx[0]), int(tile_yx[1])],
        "shape": shape,
    }


def save_streaming_provenance(
    path: str | Path,
    *,
    source: ImageRegionSource,
    sink: PyramidSink,
    params: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    """Write a JSON sidecar with source, output, and processing settings."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_id": getattr(source, "source_id", ""),
        "source_shape_tczyx": list(source.shape),
        "output": str(getattr(sink, "path", "") or ""),
        "metadata": source.metadata,
        "params": params,
        "summary": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path
