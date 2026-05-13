"""gui_deconvolve_ci.py — Standalone PyQt6 GUI for CI deconvolution.

Provides a graphical interface to the ``ci_rl_deconvolve``,
``ci_sparse_hessian_deconvolve``, and ``ci_generate_psf`` functions from
``deconvolve_ci.py``. Supports
multi-channel 3-D OME-TIFF input with per-channel PSF generation,
side-by-side input/output viewing with a shared Z-slider, and
MIP / SUM projection toggle.

Usage:
    python gui_deconvolve_ci.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# Ensure repo root is on sys.path so core.* is importable when this
# script is run directly (python gui/gui_deconvolve_ci.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

# Windows taskbar: set AppUserModelID so the taskbar shows our icon
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "ci.gui_deconvolve_ci"
    )

from PyQt6.QtCore import QObject, QEvent, Qt, QRectF, QSize, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QImage, QPainter, QPixmap, QTextCursor, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QAbstractItemView,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)
try:
    from PyQt6.QtSvg import QSvgRenderer
except ImportError:
    QSvgRenderer = None

from ci_dual_viewer import DualViewerWidget

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ICON_PATH = SCRIPT_DIR / "icon.svg"
LAST_SETTINGS_PATH = SCRIPT_DIR / ".last_settings.json"
OMERO_SESSION_REUSE_S = 10 * 60


class _GuiLogEmitter(QObject):
    line = pyqtSignal(str)


class _QtLogHandler(logging.Handler):
    """Forward Python logging records into the GUI log dialog."""

    def __init__(self, emitter: _GuiLogEmitter):
        super().__init__(level=logging.INFO)
        self._emitter = emitter
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emitter.line.emit(self.format(record))
        except Exception:
            pass


def _load_app_icon() -> QIcon:
    """Build a Windows-friendly multi-size icon from the bundled SVG."""
    if not ICON_PATH.exists():
        return QIcon()
    if QSvgRenderer is None:
        return QIcon(str(ICON_PATH))

    renderer = QSvgRenderer(str(ICON_PATH))
    if not renderer.isValid():
        return QIcon(str(ICON_PATH))

    icon = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        pixmap = QPixmap(QSize(size, size))
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        icon.addPixmap(pixmap)
    return icon

def _default_settings_dir() -> str:
    """Return Documents folder (Windows) or HOME (Linux) as default dir."""
    if sys.platform == "win32":
        docs = Path.home() / "Documents"
        if docs.is_dir():
            return str(docs)
    return str(Path.home())


def _default_downloads_dir() -> Path:
    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        return downloads
    return Path(_default_settings_dir())


def _is_accessible_directory(path: Path) -> bool:
    try:
        return path.is_dir() and os.access(path, os.R_OK | os.X_OK)
    except OSError:
        return False


def _accessible_directory_or_home(path_like: Any) -> str:
    try:
        path = Path(str(path_like)).expanduser()
    except (TypeError, ValueError):
        return str(Path.home())
    if path.is_file():
        path = path.parent
    if _is_accessible_directory(path):
        return str(path)
    return str(Path.home())


def _safe_filename_stem(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "deconvolution"


from core._meta_helpers import (
    _DEFAULT_PINHOLE_AIRY_UNITS,
    _apply_map_metadata,
    _apply_pinhole_airy_units,
    _calculate_pinhole_airy_units,
    _format_float_list,
    _metadata_float,
    _metadata_float_list,
    _ome_enum_name,
    _pinhole_size_to_um,
)


def _display_enum_name(value: Any) -> str:
    """Return a readable label for OME enum-like metadata values."""
    if value is None:
        return ""
    name = getattr(value, "name", None)
    text = str(name if name is not None else value).strip().split(".")[-1]
    return text.replace("_", " ").title() if text.isupper() else text


def _format_pinhole_values(values: list[float]) -> str:
    return ", ".join(f"{value:.2f}" for value in values)


def _default_channel_name(index: int) -> str:
    return f"CH{index + 1}"


# ---------------------------------------------------------------------------
# Channel colour helpers (same scheme as deconvolve.save_mip_png)
# ---------------------------------------------------------------------------

_FALLBACK_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (255, 0, 0),       # Red
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
]

_BGRCYM = [
    (0, 0, 255),      # Blue
    (0, 255, 0),      # Green
    (255, 0, 0),      # Red
    (0, 255, 255),    # Cyan
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
]


def _emission_to_rgb(wavelength_nm: Optional[float]) -> tuple[int, int, int]:
    """Map an emission wavelength (nm) to an approximate RGB colour."""
    if wavelength_nm is None:
        return (255, 255, 255)
    wl = wavelength_nm
    r = g = b = 0.0
    if 380 <= wl < 440:
        r = -(wl - 440) / (440 - 380)
        b = 1.0
    elif 440 <= wl < 490:
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510:
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
    elif 645 <= wl <= 780:
        r = 1.0
    else:
        return (255, 255, 255)
    return (int(r * 255), int(g * 255), int(b * 255))


def _channel_color(metadata: dict, ch_idx: int) -> tuple[int, int, int]:
    """Determine display colour for a channel from metadata."""
    channels = metadata.get("channels", [])
    ch = channels[ch_idx] if ch_idx < len(channels) and isinstance(channels[ch_idx], dict) else {}
    color = ch.get("color")
    if isinstance(color, str):
        text = color.strip().lstrip("#")
        if len(text) == 6:
            try:
                return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
            except ValueError:
                pass
    if isinstance(color, (list, tuple)) and len(color) >= 3 and tuple(color[:3]) != (255, 255, 255):
        try:
            return tuple(max(0, min(255, int(v))) for v in color[:3])  # type: ignore[return-value]
        except (TypeError, ValueError):
            pass
    em = ch.get("emission_wavelength")
    rgb = _emission_to_rgb(em)
    if rgb == (255, 255, 255):
        rgb = _FALLBACK_COLORS[ch_idx % len(_FALLBACK_COLORS)]
    return rgb


def _resolve_channel_colors(
    metadata: dict, n_ch: int
) -> list[tuple[int, int, int]]:
    """Return a list of display colours, de-duplicating if needed."""
    colors = [_channel_color(metadata, i) for i in range(n_ch)]
    if n_ch > 1 and len(set(colors)) == 1:
        colors = [_BGRCYM[i % len(_BGRCYM)] for i in range(n_ch)]
    return colors


def _release_cuda_memory(*, synchronize: bool = False) -> None:
    """Return cached CUDA memory to the driver after large tile jobs."""
    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        if synchronize:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Zoomable image view (QGraphicsView with wheel-zoom and pan)
# ---------------------------------------------------------------------------

class ZoomableImageView(QGraphicsView):
    """QGraphicsView that supports mouse-wheel zoom and middle-button pan.

    Multiple views can be linked via ``link_to()`` so that zoom level
    and scroll position stay synchronised.
    """

    _ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._linked: list[ZoomableImageView] = []
        self._syncing = False

        self.setRenderHints(
            self.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMinimumSize(256, 256)
        self.setStyleSheet("background-color: #1a1a1a; border: none;")

    # --- linking ---
    def link_to(self, other: "ZoomableImageView"):
        """Bidirectionally link two views so zoom/pan stay in sync."""
        if other not in self._linked:
            self._linked.append(other)
        if self not in other._linked:
            other._linked.append(self)

    # --- public API ---
    def set_pixmap(self, pix: QPixmap | None):
        self._scene.clear()
        self._pixmap_item = None
        if pix is not None and not pix.isNull():
            self._pixmap_item = self._scene.addPixmap(pix)
            self._scene.setSceneRect(QRectF(pix.rect()))

    def clear(self):
        self._scene.clear()
        self._pixmap_item = None

    def fit_in_view(self):
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    # --- zoom ---
    def wheelEvent(self, event: QWheelEvent):
        degrees = event.angleDelta().y() / 8
        steps = degrees / 15
        factor = self._ZOOM_FACTOR ** steps
        self.scale(factor, factor)
        self._sync_transform()

    def _sync_transform(self):
        if self._syncing:
            return
        self._syncing = True
        t = self.transform()
        cx = self.horizontalScrollBar().value()
        cy = self.verticalScrollBar().value()
        for v in self._linked:
            v._syncing = True
            v.setTransform(t)
            v.horizontalScrollBar().setValue(cx)
            v.verticalScrollBar().setValue(cy)
            v._syncing = False
        self._syncing = False

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        if not self._syncing:
            self._sync_transform()


# ---------------------------------------------------------------------------
# Helpers: numpy 2-D arrays → QPixmap
# ---------------------------------------------------------------------------

def _composite_to_pixmap(
    slices: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float]]],
    width: int = 0,
) -> QPixmap:
    """Build an RGB composite from (2-D array, colour, (lo, hi)) triples → QPixmap.

    The (lo, hi) pair should be the global min/max across the full 3-D
    volume so that contrast is consistent across Z-slices.
    """
    if not slices:
        return QPixmap()
    h, w_ = slices[0][0].shape
    canvas = np.zeros((h, w_, 3), dtype=np.float64)
    for arr, rgb, (lo, hi) in slices:
        ch_img = arr.astype(np.float64)
        if hi - lo > 0:
            ch_img = (ch_img - lo) / (hi - lo)
        else:
            ch_img = np.zeros_like(ch_img)
        for c in range(3):
            canvas[:, :, c] += ch_img * (rgb[c] / 255.0)
    canvas = np.clip(canvas, 0, 1)
    canvas = (canvas * 255).astype(np.uint8)
    canvas = np.ascontiguousarray(canvas)
    qimg = QImage(canvas.data, w_, h, w_ * 3, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())
    if width > 0:
        pix = pix.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
    return pix


# ---------------------------------------------------------------------------
# Lightweight image loader (avoids importing deconvolve.py / torch)
# ---------------------------------------------------------------------------


def _apply_metadata_defaults(images: list, meta: dict) -> dict:
    """Track metadata source and apply defaults for missing fields."""
    meta_keys_from_file = set(meta.keys())
    ch_list = meta.get("channels", [])
    if ch_list:
        meta_keys_from_file.add("channels")
        if all("emission_wavelength" in c for c in ch_list):
            meta_keys_from_file.add("emission_wavelength")
        if all("excitation_wavelength" in c for c in ch_list):
            meta_keys_from_file.add("excitation_wavelength")
        if all("acquisition_mode" in c for c in ch_list):
            meta_keys_from_file.add("acquisition_mode")

    meta.setdefault("na", 1.4)
    meta.setdefault("pixel_size_x", 0.065)
    meta.setdefault("pixel_size_z", 0.2)
    meta.setdefault("refractive_index", 1.515)
    meta.setdefault("microscope_type", "widefield")
    if "channels" not in meta:
        meta["channels"] = [{"emission_wavelength": 520.0} for _ in images]
    if _apply_pinhole_airy_units(meta):
        meta_keys_from_file.add("pinhole_airy_units")
    channel_names = [str(name) for name in meta.get("channel_names", []) if str(name).strip()]
    if len(channel_names) < len(images):
        channel_names.extend(_default_channel_name(i) for i in range(len(channel_names), len(images)))
    meta["channel_names"] = channel_names
    meta["n_channels"] = len(images)
    meta["_from_file"] = meta_keys_from_file
    return meta


def _normalize_channel_to_tzyx(channel: np.ndarray) -> np.ndarray:
    """Normalize supported image arrays to TZYX."""
    arr = np.asarray(channel, dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis, np.newaxis, :, :]
    if arr.ndim == 3:
        return arr[np.newaxis, :, :, :]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Unsupported channel shape {arr.shape}; expected 2D, 3D, or 4D.")


def _normalize_image_bundle(images: list[np.ndarray], meta: dict) -> dict:
    """Normalize per-channel image volumes to TZYX and enrich metadata."""
    channels = [_normalize_channel_to_tzyx(img) for img in images]
    if not channels:
        raise ValueError("No image channels were loaded.")
    size_t, size_z, size_y, size_x = channels[0].shape
    for idx, channel in enumerate(channels[1:], start=1):
        if channel.shape != channels[0].shape:
            raise ValueError(
                f"Channel {idx} shape {channel.shape} does not match channel 0 shape {channels[0].shape}."
            )
    meta = _apply_metadata_defaults(channels, meta)
    meta["size_t"] = size_t
    meta["size_z"] = size_z
    meta["size_y"] = size_y
    meta["size_x"] = size_x
    meta["size_c"] = len(channels)
    meta["default_t"] = 0
    meta["default_z"] = size_z // 2 if size_z > 1 else 0
    if "channel_names" not in meta or len(meta["channel_names"]) < len(channels):
        existing = list(meta.get("channel_names", []))
        meta["channel_names"] = existing + [
            _default_channel_name(i) for i in range(len(existing), len(channels))
        ]
    for idx, channel_meta in enumerate(meta.get("channels", [])):
        if "name" not in channel_meta and idx < len(meta["channel_names"]):
            channel_meta["name"] = meta["channel_names"][idx]
    return {"images": channels, "metadata": meta}


def _channel_stack_to_solver_input(channel_zyx: np.ndarray) -> np.ndarray:
    """Convert a normalized ZYX stack into 2D or 3D solver input."""
    volume = np.asarray(channel_zyx, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected ZYX input, got {volume.shape}.")
    if volume.shape[0] == 1:
        return volume[0]
    return volume


def _solver_output_to_zyx(result: np.ndarray) -> np.ndarray:
    """Normalize solver output back to ZYX for preview/export storage."""
    arr = np.asarray(result, dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unexpected solver result shape {arr.shape}.")


def _current_channel_names(meta: dict, n_channels: int) -> list[str]:
    names = list(meta.get("channel_names") or [])
    if names:
        return names[:n_channels]
    channels = meta.get("channels", [])
    if channels:
        return [
            str(ch.get("name") or f"{_default_channel_name(i)} em={ch.get('emission_wavelength', '?')}")
            for i, ch in enumerate(channels[:n_channels])
        ]
    return [_default_channel_name(i) for i in range(n_channels)]


def _extract_bioio_metadata(img, path_str: str) -> dict:
    _ACQ_MODE_MAP = {
        "LASER_SCANNING_CONFOCAL_MICROSCOPY": "confocal",
        "SPINNING_DISK_CONFOCAL": "confocal",
        "SLIT_SCAN_CONFOCAL": "confocal",
        "MULTI_PHOTON_MICROSCOPY": "confocal",
        "WIDE_FIELD": "widefield",
        "OTHER": "widefield",
    }
    _IMM_RI = {
        "OIL": 1.515,
        "WATER": 1.333,
        "GLYCEROL": 1.47,
        "AIR": 1.0,
        "MULTI": 1.515,
    }
    meta: dict = {}

    # Physical pixel sizes (µm)
    try:
        pps = img.physical_pixel_sizes
    except Exception:
        pps = None
    if pps is not None:
        px_x = getattr(pps, "X", None)
        px_z = getattr(pps, "Z", None)
        if px_x:
            meta["pixel_size_x"] = px_x
        if px_z:
            meta["pixel_size_z"] = px_z

    # OME metadata (unified across formats via ome-types)
    try:
        ome = img.ome_metadata
        if ome and hasattr(ome, "images") and ome.images:
            im0 = ome.images[0]

            # Per-channel metadata
            ch_list = []
            for c in (im0.pixels.channels or []):
                ch_d: dict = {}
                ch_d["id"] = getattr(c, "id", None)
                if c.emission_wavelength is not None:
                    ch_d["emission_wavelength"] = float(c.emission_wavelength)
                if c.excitation_wavelength is not None:
                    ch_d["excitation_wavelength"] = float(c.excitation_wavelength)
                if getattr(c, "pinhole_size", None) is not None:
                    ch_d["pinhole_size"] = float(c.pinhole_size)
                    ch_d["pinhole_size_unit"] = _ome_enum_name(getattr(c, "pinhole_size_unit", None))
                # Acquisition mode (use first channel's for top-level microscope type)
                if c.acquisition_mode:
                    name = getattr(c.acquisition_mode, "name",
                                   str(c.acquisition_mode))
                    ch_d["acquisition_mode"] = _display_enum_name(c.acquisition_mode)
                    if "microscope_type" not in meta:
                        meta["microscope_type"] = _ACQ_MODE_MAP.get(
                            name, "widefield")
                ch_list.append(ch_d)
            if ch_list:
                meta["channels"] = ch_list

            structured = getattr(ome, "structured_annotations", None)
            map_values: dict[str, str] = {}
            if structured is not None:
                for annotation in getattr(structured, "map_annotations", []) or []:
                    value = getattr(annotation, "value", None)
                    for item in getattr(value, "ms", []) or []:
                        key = getattr(item, "k", None)
                        val = getattr(item, "value", None)
                        if key and val is not None:
                            map_values[str(key)] = str(val)
            _apply_map_metadata(meta, map_values)

            # Objective (NA, immersion → RI)
            if ome.instruments:
                for inst in ome.instruments:
                    for obj in (inst.objectives or []):
                        if obj.lens_na and "na" not in meta:
                            meta["na"] = float(obj.lens_na)
                        if obj.nominal_magnification and "magnification" not in meta:
                            meta["magnification"] = float(obj.nominal_magnification)
                        if obj.immersion and "refractive_index" not in meta:
                            imm_name = getattr(
                                obj.immersion, "name",
                                str(obj.immersion)).upper()
                            if imm_name in _IMM_RI:
                                meta["refractive_index"] = _IMM_RI[imm_name]

            # ObjectiveSettings — may contain explicit RI (overrides above)
            if hasattr(im0, "objective_settings") and im0.objective_settings:
                os_ = im0.objective_settings
                if os_.refractive_index:
                    meta["refractive_index"] = float(os_.refractive_index)
    except Exception:
        pass

    try:
        channel_names = list(img.channel_names or [])
        if channel_names:
            meta["channel_names"] = [str(name) for name in channel_names]
    except Exception:
        pass

    # Fallback: try to parse channel names as emission wavelengths
    # (e.g. OME-Zarr stores channel names like '520.0', '600.0')
    if "channels" not in meta:
        try:
            ch_names = img.channel_names or []
            ch_list = []
            for nm in ch_names:
                val = float(str(nm))
                if 300 < val < 900:
                    ch_list.append({"emission_wavelength": val})
            if ch_list:
                meta["channels"] = ch_list
        except (ValueError, TypeError):
            pass

    # ND2 fallback: native nd2 metadata for RI when ome_metadata lacks it
    ext = Path(path_str).suffix.lower()
    if ext == ".nd2" and "refractive_index" not in meta:
        try:
            import nd2
            with nd2.ND2File(str(path_str)) as f:
                chs = f.metadata.channels if f.metadata else []
                if chs and chs[0].microscope:
                    ri = chs[0].microscope.immersionRefractiveIndex
                    if ri is not None and ri > 0:
                        meta["refractive_index"] = ri
        except Exception:
            pass

    return meta


def _bioio_dim_size(img, dim_name: str, default: int = 1) -> int:
    try:
        value = getattr(img.dims, dim_name)
        if value is not None:
            return max(int(value), 1)
    except Exception:
        pass
    try:
        order = str(getattr(img.dims, "order", ""))
        shape = tuple(int(v) for v in getattr(img, "shape", ()))
        if dim_name in order and len(order) == len(shape):
            return max(int(shape[order.index(dim_name)]), 1)
    except Exception:
        pass
    return max(int(default), 1)


def _is_hcs_zarr_plate(path: Path) -> bool:
    try:
        import zarr
        store = zarr.open(str(path), mode="r")
        return "plate" in store.attrs
    except Exception:
        return False


def _first_hcs_zarr_field(path: Path) -> tuple[str, str, str]:
    import zarr

    store = zarr.open(str(path), mode="r")
    plate = dict(store.attrs).get("plate", {})
    for well_entry in plate.get("wells", []):
        well_path = str(well_entry.get("path", ""))
        parts = well_path.split("/")
        if len(parts) < 2:
            continue
        well = store[well_path]
        well_attrs = dict(well.attrs)
        for image in well_attrs.get("well", {}).get("images", []):
            field = str(image.get("path", ""))
            if field:
                return parts[0], parts[1], field
        for key in sorted(well.keys()):
            if str(key).isdigit():
                return parts[0], parts[1], str(key)
    raise ValueError(f"No image fields found in HCS plate {path}")


def _hcs_zarr_metadata(field_group, row: str, col: str, field: str) -> dict:
    field_attrs = dict(field_group.attrs)
    shape = tuple(int(v) for v in field_group["0"].shape)
    while len(shape) > 5 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) == 5:
        size_t, size_c, size_z, size_y, size_x = shape
    elif len(shape) == 4:
        size_t = 1
        size_c, size_z, size_y, size_x = shape
    elif len(shape) == 3:
        size_t = 1
        size_c, size_y, size_x = shape
        size_z = 1
    else:
        raise ValueError(f"Unexpected HCS field data shape: {shape}")

    meta: dict = {
        "size_t": size_t,
        "size_c": size_c,
        "size_z": size_z,
        "size_y": size_y,
        "size_x": size_x,
        "n_channels": size_c,
        "default_t": 0,
        "default_z": size_z // 2 if size_z > 1 else 0,
        "hcs_plate_field": f"{row}/{col}/{field}",
    }

    multiscales = field_attrs.get("multiscales", [])
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

    omero_channels = field_attrs.get("omero", {}).get("channels", [])
    if omero_channels:
        meta["channel_names"] = [
            str(ch.get("label") or _default_channel_name(i)) for i, ch in enumerate(omero_channels)
        ]
        channels = []
        for i, ch in enumerate(omero_channels):
            window = ch.get("window") or {}
            channels.append({
                "name": str(ch.get("label") or _default_channel_name(i)),
                "color": ch.get("color"),
                "active": bool(ch.get("active", True)),
                "window_start": window.get("start"),
                "window_end": window.get("end"),
            })
        meta["channels"] = channels
    return meta


def _normalize_stack_to_zyx(arr: np.ndarray) -> np.ndarray:
    stack = np.asarray(arr, dtype=np.float32)
    if stack.ndim == 2:
        return stack[np.newaxis, :, :]
    if stack.ndim == 3:
        return stack
    raise ValueError(f"Unsupported stack shape {stack.shape}; expected YX or ZYX.")


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        if token.lower() in {"none", "nan", "null"}:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _format_bytes(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def _format_duration_hm(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{hours}:{minutes:02d}"


def _format_value(value, unit: str = "", digits: int = 4) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        text = f"{value:.{digits}g}"
    else:
        text = str(value)
    return f"{text} {unit}".rstrip()


def _array_stats(arr: np.ndarray) -> dict:
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


def _normalise_image(arr: np.ndarray) -> np.ndarray:
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
    if size <= limit:
        return slice(None)
    return slice(0, size, int(np.ceil(size / limit)))


def _metric_sample(arr: np.ndarray) -> np.ndarray:
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
    data = np.asarray(arr)
    sampled = _metric_sample(data)
    if data.shape == sampled.shape:
        return f"full shape {data.shape}"
    return f"full shape {data.shape} -> sampled {sampled.shape}"


def _metric_frequency_radius(shape: tuple[int, ...]) -> tuple[np.ndarray, float]:
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


def _deconvolution_effect_metrics(arr: np.ndarray) -> dict[str, float]:
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


def _quality_metrics(channels: list[np.ndarray]) -> dict[str, float | int]:
    metric_values: dict[str, list[float]] = {
        "detail_energy": [],
        "bright_detail_energy": [],
        "edge_strength": [],
        "signal_sparsity": [],
        "robust_range": [],
    }
    for channel in channels:
        metrics = _deconvolution_effect_metrics(channel)
        for key in metric_values:
            metric_values[key].append(metrics[key])
    out: dict[str, float | int] = {"channels_compared": len(channels)}
    for key, values in metric_values.items():
        out[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
    return out


def _runtime_environment_lines() -> list[str]:
    lines = [
        "Runtime environment",
        f"  Timestamp    : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"  Platform     : {platform.platform()}",
        f"  Python       : {platform.python_version()} ({sys.executable})",
        f"  NumPy        : {np.__version__}",
    ]
    try:
        import torch
        lines.append(f"  PyTorch      : {torch.__version__}")
        lines.append(f"  CUDA avail.  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"  CUDA version : {torch.version.cuda or '?'}")
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                lines.append(
                    f"  GPU {idx}       : {props.name} "
                    f"({_format_bytes(props.total_memory / (1024 * 1024))})"
                )
    except Exception as exc:
        lines.append(f"  PyTorch      : unavailable ({exc})")
    try:
        import psutil
        vm = psutil.virtual_memory()
        lines.append(
            f"  CPU cores    : {psutil.cpu_count(logical=False) or '?'} physical, "
            f"{psutil.cpu_count(logical=True) or '?'} logical"
        )
        lines.append(f"  System RAM   : {_format_bytes(vm.total / (1024 * 1024))}")
    except Exception:
        pass
    return lines


def _image_detail_lines(display_name: str, source_path: Optional[Path], meta: dict, images: list[np.ndarray]) -> list[str]:
    lines = [
        "",
        "Loaded image",
        f"  File       : {display_name}",
    ]
    if source_path is not None:
        lines.append(f"  Path       : {source_path}")
        if source_path.exists() and source_path.is_file():
            lines.append(f"  File size  : {_format_bytes(source_path.stat().st_size / (1024 * 1024))}")
    lines.extend([
        f"  Channels   : {len(images)}",
        f"  Dimensions : X={meta.get('size_x', '?')}  Y={meta.get('size_y', '?')}  "
        f"Z={meta.get('size_z', '?')}  C={meta.get('size_c', len(images))}  T={meta.get('size_t', '?')}",
        f"  Pixel size : XY={_format_value(meta.get('pixel_size_x'), 'um')}  "
        f"Z={_format_value(meta.get('pixel_size_z'), 'um')}",
        f"  Microscope : {meta.get('microscope_type', '?')}",
        f"  Objective  : NA={_format_value(meta.get('na'))}  "
        f"Mag={_format_value(meta.get('magnification'), 'x')}  Immersion={meta.get('immersion', '?')}",
        f"  RI         : immersion={_format_value(meta.get('refractive_index'))}  "
        f"sample={_format_value(meta.get('sample_refractive_index'))}",
    ])
    channels = meta.get("channels", [])
    names = meta.get("channel_names") or []
    for i, img in enumerate(images):
        stats = _array_stats(img)
        ch_meta = channels[i] if i < len(channels) else {}
        name = names[i] if i < len(names) else _default_channel_name(i)
        lines.append(
            f"  Ch{i} {name}: shape={stats['shape']} dtype={stats['dtype']} "
            f"data={_format_bytes(stats['bytes_mb'])}"
        )
        lines.append(
            f"    wavelengths: em={_format_value(ch_meta.get('emission_wavelength'), 'nm')}  "
            f"ex={_format_value(ch_meta.get('excitation_wavelength'), 'nm')}  "
            f"mode={ch_meta.get('acquisition_mode', '?')}"
        )
        lines.append(
            f"    pinhole    : size={_format_value(ch_meta.get('pinhole_size'), ch_meta.get('pinhole_size_unit') or '')}  "
            f"effective={_format_value(ch_meta.get('pinhole_airy_units'), 'AU')}"
        )
        lines.append(
            f"    intensity  : min={stats['min']:.4g} p1={stats['p1']:.4g} "
            f"median={stats['p50']:.4g} mean={stats['mean']:.4g} "
            f"p99={stats['p99']:.4g} max={stats['max']:.4g} "
            f"nonzero={stats['nonzero_percent']:.1f}%"
        )
    return lines


def _estimate_two_d_wf_psf_z_nm(
    wavelength_nm: float,
    na: float,
    sample_ri: float,
    pixel_size_xy_nm: float,
) -> float:
    """Return a practical hidden-volume Z step for 2D widefield PSFs."""
    na_safe = max(float(na), 1e-6)
    sample_safe = max(float(sample_ri), na_safe + 1e-6)
    denom = max(sample_safe - np.sqrt(max(sample_safe ** 2 - na_safe ** 2, 1e-9)), 1e-6)
    nyquist_nm = wavelength_nm / (2.0 * denom)
    return float(max(min(nyquist_nm, 1000.0), pixel_size_xy_nm / 2.0))


def _format_channel_values(
    channels: list[dict],
    key: str,
    default: float,
    *,
    digits: Optional[int] = None,
) -> str:
    values: list[str] = []
    for channel in channels:
        value = channel.get(key)
        try:
            numeric = float(value if value is not None else default)
        except (TypeError, ValueError):
            numeric = float(default)
        values.append(f"{numeric:.{digits}f}" if digits is not None else str(numeric))
    return ", ".join(values)


class _BaseTimepointSource:
    def __init__(self, metadata: dict):
        self.metadata = metadata

    def load_timepoint(
        self,
        t_index: int,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[np.ndarray]:
        raise NotImplementedError


class _BioImageTimepointSource(_BaseTimepointSource):
    def __init__(self, path_str: str):
        from bioio import BioImage

        self._path_str = str(path_str)
        self._img = BioImage(self._path_str)
        meta = _extract_bioio_metadata(self._img, self._path_str)
        size_c = _bioio_dim_size(self._img, "C", default=1)
        meta = _apply_metadata_defaults([None] * size_c, meta)
        meta["size_t"] = _bioio_dim_size(self._img, "T", default=1)
        meta["size_z"] = _bioio_dim_size(self._img, "Z", default=1)
        meta["size_y"] = _bioio_dim_size(self._img, "Y", default=1)
        meta["size_x"] = _bioio_dim_size(self._img, "X", default=1)
        meta["size_c"] = size_c
        meta["default_t"] = 0
        meta["default_z"] = meta["size_z"] // 2 if meta["size_z"] > 1 else 0
        super().__init__(meta)

    def load_timepoint(
        self,
        t_index: int,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[np.ndarray]:
        size_c = max(int(self.metadata.get("size_c", 1)), 1)
        size_t = max(int(self.metadata.get("size_t", 1)), 1)
        target_t = max(0, min(int(t_index), size_t - 1))
        channels: list[np.ndarray] = []
        total = size_c
        for c_index in range(size_c):
            if progress_cb is not None:
                progress_cb(c_index, total, f"Loading stack… channel {c_index + 1}/{total}")
            selector: dict[str, Any] = {}
            if size_t > 1:
                selector["T"] = target_t
            if size_c > 1:
                selector["C"] = c_index
            if hasattr(self._img, "get_image_dask_data"):
                arr = self._img.get_image_dask_data("ZYX", **selector).compute()
            else:
                arr = self._img.get_image_data("ZYX", **selector)
            channels.append(_normalize_stack_to_zyx(arr))
            if progress_cb is not None:
                progress_cb(c_index + 1, total, f"Loading stack… channel {c_index + 1}/{total}")
        return channels


class _HcsZarrTimepointSource(_BaseTimepointSource):
    def __init__(self, path_str: str):
        import zarr

        self._path = Path(path_str)
        self._store = zarr.open(str(self._path), mode="r")
        self._row, self._col, self._field = _first_hcs_zarr_field(self._path)
        self._field_group = self._store[f"{self._row}/{self._col}/{self._field}"]

        meta = _hcs_zarr_metadata(self._field_group, self._row, self._col, self._field)
        size_c = max(int(meta.get("size_c", 1)), 1)
        meta = _apply_metadata_defaults([None] * size_c, meta)
        super().__init__(meta)

    def load_timepoint(
        self,
        t_index: int,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[np.ndarray]:
        dataset = self._field_group["0"]
        shape = tuple(int(v) for v in dataset.shape)
        size_c = max(int(self.metadata.get("size_c", 1)), 1)
        size_t = max(int(self.metadata.get("size_t", 1)), 1)
        target_t = max(0, min(int(t_index), size_t - 1))

        if len(shape) == 5:
            data = np.asarray(dataset[target_t])
        elif len(shape) == 4:
            data = np.asarray(dataset[:])
        elif len(shape) == 3:
            data = np.asarray(dataset[:])
        else:
            data = np.asarray(dataset[:])
            while data.ndim > 5 and data.shape[0] == 1:
                data = data[0]
            if data.ndim == 5:
                data = data[target_t]

        channels: list[np.ndarray] = []
        total = size_c
        for c_index in range(size_c):
            if progress_cb is not None:
                progress_cb(c_index, total, f"Loading HCS field {self._row}/{self._col}/{self._field} channel {c_index + 1}/{total}")
            channel = data[c_index]
            channels.append(_normalize_stack_to_zyx(channel))
            if progress_cb is not None:
                progress_cb(c_index + 1, total, f"Loading HCS field {self._row}/{self._col}/{self._field} channel {c_index + 1}/{total}")
        return channels


class _OmeroTimepointSource(_BaseTimepointSource):
    def __init__(self, image):
        from omero_browser_qt import (
            PyramidTileProvider,
            RegularImagePlaneProvider,
            get_image_metadata,
            is_large_image,
        )

        self._image = image
        self.is_pyramidal = bool(is_large_image(image))
        self.tile_provider = PyramidTileProvider(image) if self.is_pyramidal else None
        self._provider = None if self.is_pyramidal else RegularImagePlaneProvider(image)
        meta = get_image_metadata(image)
        size_c = max(int(meta.get("size_c", 1)), 1)
        meta = _apply_metadata_defaults([None] * size_c, meta)
        meta["size_c"] = size_c
        meta["default_t"] = 0
        meta["default_z"] = max(int(meta.get("size_z", 1)) // 2, 0)
        super().__init__(meta)

    def load_timepoint(
        self,
        t_index: int,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[np.ndarray]:
        size_c = max(int(self.metadata.get("size_c", 1)), 1)
        size_t = max(int(self.metadata.get("size_t", 1)), 1)
        size_z = max(int(self.metadata.get("size_z", 1)), 1)
        target_t = max(0, min(int(t_index), size_t - 1))
        if self.is_pyramidal and self.tile_provider is not None:
            if progress_cb is not None:
                progress_cb(0, 1, "Loading OMERO pyramid overview…")
            z = max(0, min(int(self.metadata.get("default_z", 0)), size_z - 1))
            channels = [
                _normalize_stack_to_zyx(ch).astype(np.float32, copy=False)
                for ch in self.tile_provider.load_overview(z=z, t=target_t)
            ]
            if progress_cb is not None:
                progress_cb(1, 1, "Loaded OMERO pyramid overview")
            return channels

        channels: list[np.ndarray] = []
        total = max(size_c * size_z, 1)
        for c_index in range(size_c):
            base = c_index * size_z

            def _progress(done: int, channel_total: int, *, _c=c_index, _base=base) -> None:
                if progress_cb is None:
                    return
                progress_cb(
                    _base + done,
                    total,
                    f"Loading stack… channel {_c + 1}/{size_c}, plane {done}/{channel_total}",
                )

            stack = self._provider.get_stack(c_index, target_t, progress=_progress)
            channels.append(_normalize_stack_to_zyx(stack))
            if progress_cb is not None:
                progress_cb(base + size_z, total, f"Loading stack… channel {c_index + 1}/{size_c}")
        return channels


class _OmeroPyramidRegionSource:
    """Full-resolution OMERO region reader backed by PyramidTileProvider tiles."""

    def __init__(self, source: _OmeroTimepointSource):
        if not getattr(source, "is_pyramidal", False) or source.tile_provider is None:
            raise ValueError("OMERO source is not pyramidal")
        self._source = source
        self._provider = source.tile_provider
        self._level = max(int(self._provider.n_levels) - 1, 0)
        size_x, size_y = self._provider.level_size(self._level)
        self.shape = (
            max(int(source.metadata.get("size_t", 1)), 1),
            max(int(source.metadata.get("size_c", 1)), 1),
            max(int(source.metadata.get("size_z", 1)), 1),
            int(size_y),
            int(size_x),
        )
        self.metadata = dict(source.metadata)
        self.metadata["size_t"] = self.shape[0]
        self.metadata["size_c"] = self.shape[1]
        self.metadata["size_z"] = self.shape[2]
        self.metadata["size_y"] = self.shape[3]
        self.metadata["size_x"] = self.shape[4]
        image = getattr(source, "_image", None)
        image_id = image.getId() if image is not None and hasattr(image, "getId") else "unknown"
        self.source_id = f"omero:pyramid:{image_id}"

    def read_region(
        self,
        *,
        t: int,
        c: int,
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        size_t, size_c, size_z, size_y, size_x = self.shape
        t = max(0, min(int(t), size_t - 1))
        c = max(0, min(int(c), size_c - 1))
        z0, z1, z_step = z.indices(size_z)
        y0, y1, y_step = y.indices(size_y)
        x0, x1, x_step = x.indices(size_x)
        if z_step != 1 or y_step != 1 or x_step != 1:
            raise ValueError("OMERO pyramid streaming only supports contiguous regions")

        out = np.zeros((max(0, z1 - z0), max(0, y1 - y0), max(0, x1 - x0)), dtype=np.float32)
        if out.size == 0:
            return out

        tile_w, tile_h = self._provider.tile_size(self._level)
        tx0 = x0 // tile_w
        tx1 = (x1 + tile_w - 1) // tile_w
        ty0 = y0 // tile_h
        ty1 = (y1 + tile_h - 1) // tile_h

        for zi, z_abs in enumerate(range(z0, z1)):
            for ty in range(ty0, ty1):
                tile_y0 = ty * tile_h
                tile_y1 = min(size_y, tile_y0 + tile_h)
                oy0 = max(y0, tile_y0)
                oy1 = min(y1, tile_y1)
                if oy1 <= oy0:
                    continue
                for tx in range(tx0, tx1):
                    tile_x0 = tx * tile_w
                    tile_x1 = min(size_x, tile_x0 + tile_w)
                    ox0 = max(x0, tile_x0)
                    ox1 = min(x1, tile_x1)
                    if ox1 <= ox0:
                        continue
                    tile = self._provider.get_tile(self._level, c, z_abs, t, tx, ty)
                    if tile is None:
                        continue
                    out[
                        zi,
                        oy0 - y0:oy1 - y0,
                        ox0 - x0:ox1 - x0,
                    ] = np.asarray(
                        tile[oy0 - tile_y0:oy1 - tile_y0, ox0 - tile_x0:ox1 - tile_x0],
                        dtype=np.float32,
                    )
        return out


class _TimepointRegionSource:
    """Region-source adapter for sources that only expose full timepoint loads."""

    def __init__(self, source: _BaseTimepointSource, *, source_id: str = "timepoint"):
        self._source = source
        self.metadata = dict(source.metadata)
        self.shape = (
            max(int(self.metadata.get("size_t", 1)), 1),
            max(int(self.metadata.get("size_c", 1)), 1),
            max(int(self.metadata.get("size_z", 1)), 1),
            max(int(self.metadata.get("size_y", 1)), 1),
            max(int(self.metadata.get("size_x", 1)), 1),
        )
        self.source_id = source_id
        self._cache_t: Optional[int] = None
        self._cache_channels: list[np.ndarray] | None = None

    def _channels_for_t(self, t: int) -> list[np.ndarray]:
        t = max(0, min(int(t), self.shape[0] - 1))
        if self._cache_t != t or self._cache_channels is None:
            self._cache_channels = [
                _normalize_stack_to_zyx(ch).astype(np.float32, copy=False)
                for ch in self._source.load_timepoint(t)
            ]
            self._cache_t = t
        return self._cache_channels

    def read_region(
        self,
        *,
        t: int,
        c: int,
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        channels = self._channels_for_t(t)
        ci = max(0, min(int(c), len(channels) - 1))
        return np.asarray(channels[ci][z, y, x], dtype=np.float32)


def _leica_microscope_type(metadata: dict) -> str:
    raw = str(
        metadata.get("microscope_type")
        or metadata.get("mic_type2")
        or metadata.get("mic_type")
        or ""
    ).lower()
    if "conf" in raw or "scanner" in raw:
        return "confocal"
    return "widefield"


def _leica_channel_metadata(metadata: dict, size_c: int) -> list[dict]:
    emissions = metadata.get("emission")
    excitations = metadata.get("excitation")
    names = metadata.get("channel_names") or metadata.get("lutname") or []
    channels: list[dict] = []
    for idx in range(size_c):
        ch: dict = {}
        if isinstance(names, list) and idx < len(names) and names[idx]:
            ch["name"] = str(names[idx])
        else:
            ch["name"] = _default_channel_name(idx)
        if isinstance(emissions, list) and idx < len(emissions):
            try:
                value = float(emissions[idx])
                if value > 0:
                    ch["emission_wavelength"] = value
            except (TypeError, ValueError):
                pass
        if isinstance(excitations, list) and idx < len(excitations):
            try:
                value = float(excitations[idx])
                if value > 0:
                    ch["excitation_wavelength"] = value
            except (TypeError, ValueError):
                pass
        ch["acquisition_mode"] = _leica_microscope_type(metadata)
        pinhole = metadata.get("pinhole_airy")
        if pinhole is not None:
            ch["pinhole_airy_units"] = pinhole
        channels.append(ch)
    return channels


def _positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _leica_resolution_m_to_um(value: Any) -> Optional[float]:
    number = _positive_float(value)
    if number is None:
        return None
    return number * 1_000_000.0


class _LeicaTimepointSource(_BaseTimepointSource):
    def __init__(self, context):
        self._context = context
        self._handle = context.open()
        src = dict(context.metadata)
        size_c = max(int(context.size_c or src.get("channels") or 1), 1)
        meta = {
            "pixel_size_x": (
                _leica_resolution_m_to_um(src.get("xres"))
                or _positive_float(context.pixel_size_x_um)
                or _positive_float(src.get("xres2"))
            ),
            "pixel_size_y": (
                _leica_resolution_m_to_um(src.get("yres"))
                or _positive_float(context.pixel_size_y_um)
                or _positive_float(src.get("yres2"))
            ),
            "pixel_size_z": (
                _leica_resolution_m_to_um(src.get("zres"))
                or _positive_float(context.pixel_size_z_um)
                or _positive_float(src.get("zres2"))
            ),
            "na": src.get("na"),
            "refractive_index": src.get("refractiveindex") or src.get("refractive_index"),
            "magnification": src.get("magnification"),
            "microscope_type": _leica_microscope_type(src),
            "channel_names": list(context.channel_names or src.get("channel_names") or []),
            "channels": _leica_channel_metadata(src, size_c),
            "size_t": max(int(context.size_t or src.get("ts") or 1), 1),
            "size_z": max(int(context.size_z or src.get("zs") or 1), 1),
            "size_y": max(int(context.size_y or src.get("ys") or 1), 1),
            "size_x": max(int(context.size_x or src.get("xs") or 1), 1),
            "size_c": size_c,
            "default_t": 0,
            "default_z": max(int(context.size_z or src.get("zs") or 1) // 2, 0),
            "leica_context": context.to_dict(),
        }
        meta = {k: v for k, v in meta.items() if v is not None}
        meta = _apply_metadata_defaults([None] * size_c, meta)
        super().__init__(meta)

    def load_timepoint(
        self,
        t_index: int,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[np.ndarray]:
        size_c = max(int(self.metadata.get("size_c", 1)), 1)
        size_t = max(int(self.metadata.get("size_t", 1)), 1)
        target_t = max(0, min(int(t_index), size_t - 1))
        channels: list[np.ndarray] = []
        total = size_c
        for c_index in range(size_c):
            if progress_cb is not None:
                progress_cb(
                    c_index,
                    total,
                    f"Loading Leica stack… channel {c_index + 1}/{total}",
                )
            stack = self._handle.read_stack(c=c_index, t=target_t)
            channels.append(_normalize_stack_to_zyx(stack))
            if progress_cb is not None:
                progress_cb(
                    c_index + 1,
                    total,
                    f"Loading Leica stack… channel {c_index + 1}/{total}",
                )
        return channels


def _build_file_source(path_str: str) -> _BaseTimepointSource:
    path = Path(path_str)
    if path.is_dir() and path.suffix.lower() == ".zarr" and _is_hcs_zarr_plate(path):
        return _HcsZarrTimepointSource(path_str)
    return _BioImageTimepointSource(path_str)


def _build_omero_source(image) -> _BaseTimepointSource:
    return _OmeroTimepointSource(image)


def _build_leica_source(context) -> _BaseTimepointSource:
    return _LeicaTimepointSource(context)


# ---------------------------------------------------------------------------
# Live resource monitor (CPU / RAM / GPU / VRAM)
# ---------------------------------------------------------------------------

class _ResourceMonitor(QThread):
    """Background QThread that polls CPU/RAM/GPU metrics every 500 ms.

    Emits ``metrics_updated`` with a dict containing:
        cpu_pct, ram_used_gb, ram_total_gb, ram_swap_gb,
        gpu_pct, vram_used_gb, vram_total_gb, vram_spill_gb, has_gpu
    """

    metrics_updated = pyqtSignal(dict)
    _POLL_S = 0.5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()
        self._cpu_count = max(os.cpu_count() or 1, 1)

        # psutil
        self._psutil = None
        self._proc = None
        try:
            import psutil as _ps
            self._psutil = _ps
            self._proc = _ps.Process()
        except ImportError:
            pass

        # pynvml
        self._pynvml = None
        self._nvml_handle = None
        try:
            import pynvml as _nv
            _nv.nvmlInit()
            self._nvml_handle = _nv.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = _nv
        except Exception:
            pass

        # nvidia-smi fallback for GPU utilisation on Windows
        import shutil
        self._nvsmi = shutil.which("nvidia-smi")

        # torch GPU available?
        self._torch_cuda = False
        try:
            import torch
            self._torch_cuda = torch.cuda.is_available()
        except Exception:
            pass

    def run(self):
        self._stop.clear()
        # Prime cpu_percent (first call always returns 0)
        if self._proc:
            try:
                self._proc.cpu_percent()
            except Exception:
                pass
        while not self._stop.is_set():
            self.metrics_updated.emit(self._poll())
            self._stop.wait(self._POLL_S)

    def _poll(self) -> dict:
        m: dict = {
            "cpu_pct": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "ram_swap_gb": 0.0,
            "ram_swap_total_gb": 0.0,
            "gpu_pct": 0.0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "vram_spill_gb": 0.0,
            "has_gpu": False,
        }
        if self._psutil:
            try:
                # System-wide RAM (more meaningful than process RSS)
                vm = self._psutil.virtual_memory()
                m["ram_used_gb"] = vm.used / (1024 ** 3)
                m["ram_total_gb"] = vm.total / (1024 ** 3)
                # Swap / pagefile
                sw = self._psutil.swap_memory()
                m["ram_swap_gb"] = sw.used / (1024 ** 3)
                m["ram_swap_total_gb"] = sw.total / (1024 ** 3)
            except Exception:
                pass
            if self._proc:
                try:
                    proc_cpu_pct = float(self._proc.cpu_percent())
                    m["cpu_pct"] = min(
                        max(proc_cpu_pct / float(self._cpu_count), 0.0),
                        100.0,
                    )
                except Exception:
                    pass
        # --- GPU utilisation & VRAM ---
        gpu_pct_set = False

        # 1) Try pynvml (most reliable when available)
        if self._nvml_handle:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                m["gpu_pct"] = float(util.gpu)
                gpu_pct_set = True
                mi = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                m["vram_used_gb"] = mi.used / (1024 ** 3)
                m["vram_total_gb"] = mi.total / (1024 ** 3)
                m["has_gpu"] = True
            except Exception:
                pass

        # 2) nvidia-smi subprocess fallback for GPU % (works on Windows
        #    even when pynvml fails to report utilisation)
        if not gpu_pct_set and self._nvsmi:
            try:
                import subprocess
                out = subprocess.check_output(
                    [self._nvsmi,
                     "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    timeout=2,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                parts = out.decode().strip().split(",")
                if len(parts) >= 3:
                    m["gpu_pct"] = float(parts[0].strip())
                    gpu_pct_set = True
                    m["vram_used_gb"] = float(parts[1].strip()) / 1024
                    m["vram_total_gb"] = float(parts[2].strip()) / 1024
                    m["has_gpu"] = True
            except Exception:
                pass

        # 3) torch.cuda fallback (VRAM only, no utilisation %)
        if self._torch_cuda:
            try:
                import torch
                if not m["has_gpu"]:
                    m["has_gpu"] = True
                    props = torch.cuda.get_device_properties(0)
                    m["vram_total_gb"] = props.total_memory / (1024 ** 3)
                    m["vram_used_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
                if not gpu_pct_set:
                    m["gpu_pct"] = -1.0  # sentinel: "unknown"

                # VRAM spill: torch reserved memory exceeds physical VRAM
                # (Windows spills into pagefile/shared GPU memory)
                reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                if m["vram_total_gb"] > 0 and reserved_gb > m["vram_total_gb"]:
                    m["vram_spill_gb"] = reserved_gb - m["vram_total_gb"]
            except Exception:
                pass

        return m

    def request_stop(self):
        """Signal the polling loop to stop and wait up to 2 s."""
        self._stop.set()
        self.wait(2000)


class _RunMetricsMonitor:
    """Process/GPU resource sampler used for one deconvolution run."""

    def __init__(self, interval: float = 0.1):
        self._interval = float(interval)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu_percent: list[float] = []
        self._ram_bytes: list[int] = []
        self._gpu_util: list[float] = []
        self._gpu_mem_bytes: list[int] = []
        self._ram_baseline = 0
        self._gpu_mem_baseline = 0
        self._torch_baseline = 0
        self._t0 = 0.0
        self._t1 = 0.0

        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
        except Exception:
            pass

        self._nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def start(self) -> None:
        self._cpu_percent.clear()
        self._ram_bytes.clear()
        self._gpu_util.clear()
        self._gpu_mem_bytes.clear()
        self._stop_event.clear()

        if self._proc:
            try:
                self._proc.cpu_percent()
                self._ram_baseline = self._proc.memory_info().rss
            except Exception:
                self._ram_baseline = 0
        if self._nvml_handle:
            try:
                import pynvml
                self._gpu_mem_baseline = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used
            except Exception:
                self._gpu_mem_baseline = 0
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._torch_baseline = torch.cuda.memory_allocated()
        except Exception:
            self._torch_baseline = 0

        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            if self._proc:
                try:
                    self._cpu_percent.append(float(self._proc.cpu_percent()))
                    self._ram_bytes.append(int(self._proc.memory_info().rss))
                except Exception:
                    pass
            if self._nvml_handle:
                try:
                    import pynvml
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    self._gpu_util.append(float(util.gpu))
                    self._gpu_mem_bytes.append(int(pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used))
                except Exception:
                    pass
            self._stop_event.wait(self._interval)

    def stop(self) -> dict[str, float]:
        self._t1 = time.perf_counter()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        MB = 1024 * 1024
        m = {
            "time_s": self._t1 - self._t0,
            "cpu_percent_avg": 0.0,
            "cpu_percent_peak": 0.0,
            "ram_total_mb": 0.0,
            "ram_peak_mb": 0.0,
            "ram_avg_mb": 0.0,
            "ram_delta_peak_mb": 0.0,
            "ram_percent": 0.0,
            "gpu_util_avg": 0.0,
            "gpu_util_peak": 0.0,
            "gpu_total_mb": 0.0,
            "gpu_mem_peak_mb": 0.0,
            "gpu_mem_avg_mb": 0.0,
            "gpu_mem_delta_peak_mb": 0.0,
            "gpu_mem_percent": 0.0,
            "torch_gpu_peak_mb": 0.0,
            "torch_gpu_delta_mb": 0.0,
        }
        if self._cpu_percent:
            m["cpu_percent_avg"] = sum(self._cpu_percent) / len(self._cpu_percent)
            m["cpu_percent_peak"] = max(self._cpu_percent)
        try:
            import psutil
            m["ram_total_mb"] = psutil.virtual_memory().total / MB
        except Exception:
            pass
        if self._ram_bytes:
            peak = max(self._ram_bytes)
            m["ram_peak_mb"] = peak / MB
            m["ram_avg_mb"] = sum(self._ram_bytes) / len(self._ram_bytes) / MB
            m["ram_delta_peak_mb"] = (peak - self._ram_baseline) / MB
            if m["ram_total_mb"] > 0:
                m["ram_percent"] = m["ram_peak_mb"] / m["ram_total_mb"] * 100
        if self._gpu_util:
            m["gpu_util_avg"] = sum(self._gpu_util) / len(self._gpu_util)
            m["gpu_util_peak"] = max(self._gpu_util)
        if self._gpu_mem_bytes:
            peak = max(self._gpu_mem_bytes)
            m["gpu_mem_peak_mb"] = peak / MB
            m["gpu_mem_avg_mb"] = sum(self._gpu_mem_bytes) / len(self._gpu_mem_bytes) / MB
            m["gpu_mem_delta_peak_mb"] = (peak - self._gpu_mem_baseline) / MB
        if self._nvml_handle:
            try:
                import pynvml
                m["gpu_total_mb"] = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).total / MB
            except Exception:
                pass
        try:
            import torch
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated()
                m["torch_gpu_peak_mb"] = peak / MB
                m["torch_gpu_delta_mb"] = (peak - self._torch_baseline) / MB
        except Exception:
            pass
        if m["gpu_total_mb"] > 0 and m["gpu_mem_peak_mb"] > 0:
            m["gpu_mem_percent"] = m["gpu_mem_peak_mb"] / m["gpu_total_mb"] * 100
        return m


def _resource_metric_lines(metrics: dict[str, float]) -> list[str]:
    gpu_delta = (
        metrics.get("torch_gpu_delta_mb", 0.0)
        if metrics.get("torch_gpu_delta_mb", 0.0) > 0
        else metrics.get("gpu_mem_delta_peak_mb", 0.0)
    )
    lines = [
        "",
        "Resource metrics",
        f"  Deconv time : {_format_duration(metrics.get('time_s', 0.0))}",
        f"  CPU         : avg={metrics.get('cpu_percent_avg', 0.0):.0f}%  "
        f"peak={metrics.get('cpu_percent_peak', 0.0):.0f}%",
        f"  RAM         : peak={_format_bytes(metrics.get('ram_peak_mb', 0.0))}  "
        f"delta={_format_bytes(metrics.get('ram_delta_peak_mb', 0.0))}  "
        f"avg={_format_bytes(metrics.get('ram_avg_mb', 0.0))}",
    ]
    if metrics.get("gpu_total_mb", 0.0) > 0 or metrics.get("gpu_mem_peak_mb", 0.0) > 0:
        lines.extend([
            f"  GPU         : util avg={metrics.get('gpu_util_avg', 0.0):.0f}%  "
            f"peak={metrics.get('gpu_util_peak', 0.0):.0f}%",
            f"  VRAM        : peak={_format_bytes(metrics.get('gpu_mem_peak_mb', 0.0))}  "
            f"delta={_format_bytes(gpu_delta)}  "
            f"torch peak={_format_bytes(metrics.get('torch_gpu_peak_mb', 0.0))}",
        ])
    return lines


def _quality_comparison_lines(source_channels: list[np.ndarray], result_channels: list[np.ndarray]) -> list[str]:
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
    lines = [
        "",
        "Image metrics",
    ]
    n_channels = min(len(source_channels), len(result_channels))
    if n_channels:
        lines.append(
            f"  Metrics sample: {_metric_sample_summary(source_channels[0])} "
            f"(caps: Z<=32, Y/X<=512)"
        )
    for ch_idx in range(n_channels):
        source_q = _deconvolution_effect_metrics(source_channels[ch_idx])
        result_q = _deconvolution_effect_metrics(result_channels[ch_idx])
        lines.extend([
            f"  Channel {ch_idx}",
            "    Metric                 Source        Result      Change",
            "    -------------------------------------------------------",
        ])
        for key, label in metric_order:
            src = float(source_q.get(key, 0.0))
            res = float(result_q.get(key, 0.0))
            lines.append(f"    {label:<18} {src:>11.4g} {res:>13.4g} {_change(src, res):>10}")
    if n_channels > 1:
        source_mean = _quality_metrics(source_channels[:n_channels])
        result_mean = _quality_metrics(result_channels[:n_channels])
        lines.extend([
            "  Mean across channels",
            "    Metric                 Source        Result      Change",
            "    -------------------------------------------------------",
        ])
        for key, label in metric_order:
            mean_key = f"{key}_mean"
            src = float(source_mean.get(mean_key, 0.0))
            res = float(result_mean.get(mean_key, 0.0))
            lines.append(f"    {label:<18} {src:>11.4g} {res:>13.4g} {_change(src, res):>10}")
    return lines


# ---------------------------------------------------------------------------
# Single-metric horizontal bar widget (label | progress | value text)
# ---------------------------------------------------------------------------

class _MetricWidget(QWidget):
    """Compact bar displaying one resource metric."""

    _COLORS = {"ok": "#4CAF50", "warn": "#FF9800", "crit": "#F44336"}

    def __init__(self, label: str, bar_width: int = 80, val_width: int = 88, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(1, 0, 1, 0)
        layout.setSpacing(2)

        lbl = QLabel(label)
        lbl.setFixedWidth(36)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        f = lbl.font()
        f.setPointSize(8)
        lbl.setFont(f)
        layout.addWidget(lbl)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedWidth(bar_width)
        self._bar.setFixedHeight(13)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(self._style("#4CAF50"))
        layout.addWidget(self._bar)

        self._val = QLabel("—")
        self._val.setFixedWidth(val_width)
        f2 = self._val.font()
        f2.setPointSize(8)
        self._val.setFont(f2)
        layout.addWidget(self._val)

    @staticmethod
    def _style(color: str) -> str:
        return (
            f"QProgressBar {{ border: 1px solid #555; border-radius: 3px; "
            f"background-color: #2a2a2a; }}"
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}"
        )

    def set_value(self, pct: float, text: str):
        self._bar.setValue(int(min(max(pct, 0), 100)))
        self._val.setText(text)
        color = (
            self._COLORS["crit"] if pct >= 90
            else self._COLORS["warn"] if pct >= 70
            else self._COLORS["ok"]
        )
        self._bar.setStyleSheet(self._style(color))


class _LogDialog(QDialog):
    """Detached live log window."""

    metricsToggled = pyqtSignal(bool)
    chartsToggled = pyqtSignal(bool)

    # Default chart height when no image has been seen yet (assumes square image).
    _DEFAULT_PANEL_H = 230
    # Width used for convergence and sharpness charts
    _CHART_W = 260
    # max_image_w passed to _render_residual_pixmap — determines residual image pixel width
    _RESID_IMAGE_W = 230

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._convergence_series: list[Optional[float]] = []
        self._convergence_total: int = 0
        self._sharpness_series: list[Optional[float]] = []
        # Per-channel offset so multi-channel runs don't overwrite each other's data
        self._channel_offset: int = 0
        self._last_channel_index: int = -1
        # Chart height — updated after push_final_residual reveals the true aspect ratio
        self._panel_height: int = self._DEFAULT_PANEL_H
        self.setWindowTitle("CIDeconvolve Log")
        self.resize(900, 560)
        layout = QVBoxLayout(self)

        # --- Live chart panel (sits above the log text) ---
        self._chart_panel = QWidget()
        chart_row = QHBoxLayout(self._chart_panel)
        chart_row.setContentsMargins(4, 4, 4, 4)
        chart_row.setSpacing(8)

        self._conv_label = QLabel("Waiting for data\u2026")
        self._conv_label.setFixedSize(self._CHART_W, self._DEFAULT_PANEL_H)
        self._conv_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._conv_label.setStyleSheet(
            "background: #16161E; color: #888; border-radius: 4px; font-size: 9px;"
        )

        self._resid_label = QLabel("|raw \u2212 est| (final frame)")
        # Width = image area + colorbar; height = default (square assumption)
        _resid_default_w = self._RESID_IMAGE_W + 5 + 12 + 38   # gap+bar+labels
        self._resid_label.setFixedSize(_resid_default_w, self._DEFAULT_PANEL_H)
        self._resid_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._resid_label.setStyleSheet(
            "background: #16161E; color: #888; border-radius: 4px; font-size: 9px;"
        )

        self._sharp_label = QLabel("Waiting for data\u2026")
        self._sharp_label.setFixedSize(self._CHART_W, self._DEFAULT_PANEL_H)
        self._sharp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sharp_label.setStyleSheet(
            "background: #16161E; color: #888; border-radius: 4px; font-size: 9px;"
        )

        chart_row.addWidget(self._conv_label)
        chart_row.addWidget(self._resid_label)
        chart_row.addWidget(self._sharp_label)
        chart_row.addStretch()
        self._chart_panel.setVisible(False)
        layout.addWidget(self._chart_panel)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFont("Consolas")
        if not font.exactMatch():
            font = QFont("Courier New")
        font.setPointSize(9)
        self._text.setFont(font)
        self._text.installEventFilter(self)
        self._text.viewport().installEventFilter(self)
        layout.addWidget(self._text, stretch=1)

        buttons = QHBoxLayout()
        self._metrics_check = QCheckBox("Compute image metrics")
        self._metrics_check.setToolTip(
            "Compute FFT/gradient image metrics after deconvolution. "
            "This can take noticeable time on large images."
        )
        self._metrics_check.toggled.connect(self.metricsToggled.emit)
        buttons.addWidget(self._metrics_check)

        self._charts_check = QCheckBox("Live charts")
        self._charts_check.setToolTip(
            "Show live convergence graph and final residual image above the log text.\n"
            "Updates during deconvolution; residual only rendered at the end."
        )
        self._charts_check.toggled.connect(self._on_charts_toggled)
        self._charts_check.toggled.connect(self.chartsToggled.emit)
        buttons.addWidget(self._charts_check)

        self._conv_log_check = QCheckBox("Log scale")
        self._conv_log_check.setToolTip(
            "Display the convergence chart Y-axis on a logarithmic scale."
        )
        buttons.addWidget(self._conv_log_check)

        buttons.addStretch()
        self._save_button = QPushButton("Save\u2026")
        self._save_button.clicked.connect(self._save_log)
        buttons.addWidget(self._save_button)
        self._close_button = QPushButton("Close")
        self._close_button.clicked.connect(self.close)
        buttons.addWidget(self._close_button)
        layout.addLayout(buttons)

    def _on_charts_toggled(self, enabled: bool) -> None:
        self._chart_panel.setVisible(bool(enabled))

    def is_charts_enabled(self) -> bool:
        return self._charts_check.isChecked()

    def set_charts_enabled(self, enabled: bool) -> None:
        self._charts_check.setChecked(bool(enabled))

    def set_conv_log_scale(self, enabled: bool) -> None:
        self._conv_log_check.setChecked(bool(enabled))

    def push_iteration(self, payload: dict) -> None:
        """Update the live convergence and sharpness charts from an iteration callback payload."""
        if not self._charts_check.isChecked():
            return
        iteration = int(payload.get("iteration", 1))
        total = int(payload.get("total_iterations", iteration))
        channel_index = int(payload.get("channel_index", 0))

        # When the channel changes, start fresh so each channel's curve
        # is shown on its own scale (no dips from channel-to-channel jumps).
        if channel_index != self._last_channel_index:
            self._convergence_series = []
            self._convergence_total = 0
            self._sharpness_series = []
            self._channel_offset = 0
            self._last_channel_index = channel_index

        abs_idx = iteration - 1
        abs_total = total

        # For ci_sparse_hessian, prefer the raw data-fidelity loss (Poisson NLL,
        # analogous to I-div) over the combined normalised objective.
        convergence = payload.get("data_loss") if "data_loss" in payload else payload.get("convergence")
        while len(self._convergence_series) <= abs_idx:
            self._convergence_series.append(None)
        if convergence is not None:
            try:
                v = float(convergence)
                if np.isfinite(v):
                    self._convergence_series[abs_idx] = v
            except (TypeError, ValueError):
                pass
        self._convergence_total = max(self._convergence_total, abs_total)
        h = self._panel_height
        try:
            pix = _render_convergence_chart_pixmap(
                self._convergence_series, self._convergence_total,
                self._conv_log_check.isChecked(),
                width=self._CHART_W, height=h,
            )
            self._conv_label.setFixedSize(self._CHART_W, h)
            self._conv_label.setPixmap(pix)
            self._conv_label.setText("")
        except Exception:
            pass
        # Per-iteration sharpness trend
        image_arr = payload.get("image")
        if image_arr is not None:
            sharpness = _compute_sharpness(image_arr)
            if sharpness is not None:
                while len(self._sharpness_series) <= abs_idx:
                    self._sharpness_series.append(None)
                self._sharpness_series[abs_idx] = sharpness
                try:
                    spix = _render_sharpness_chart_pixmap(
                        self._sharpness_series, self._convergence_total,
                        width=self._CHART_W, height=h,
                    )
                    self._sharp_label.setFixedSize(self._CHART_W, h)
                    self._sharp_label.setPixmap(spix)
                    self._sharp_label.setText("")
                except Exception:
                    pass

    def push_final_residual(self, raw_plane: np.ndarray, est_plane: np.ndarray) -> None:
        """Render and display the |raw − est| residual for the final frame."""
        if not self._charts_check.isChecked():
            return
        try:
            pix, pw, ph = _render_residual_pixmap(
                raw_plane, est_plane, max_image_w=self._RESID_IMAGE_W
            )
            self._panel_height = ph
            self._resid_label.setFixedSize(pw, ph)
            self._resid_label.setPixmap(pix)
            self._resid_label.setText("")
            # Resize the other two charts to match the heatmap height
            if self._convergence_series:
                try:
                    p2 = _render_convergence_chart_pixmap(
                        self._convergence_series, self._convergence_total,
                        self._conv_log_check.isChecked(),
                        width=self._CHART_W, height=ph,
                    )
                    self._conv_label.setFixedSize(self._CHART_W, ph)
                    self._conv_label.setPixmap(p2)
                except Exception:
                    pass
            if self._sharpness_series:
                try:
                    p3 = _render_sharpness_chart_pixmap(
                        self._sharpness_series, self._convergence_total,
                        width=self._CHART_W, height=ph,
                    )
                    self._sharp_label.setFixedSize(self._CHART_W, ph)
                    self._sharp_label.setPixmap(p3)
                except Exception:
                    pass
        except Exception:
            pass

    def clear_charts(self) -> None:
        """Reset chart state for a new deconvolution run."""
        self._convergence_series = []
        self._convergence_total = 0
        self._sharpness_series = []
        self._channel_offset = 0
        self._last_channel_index = -1
        self._panel_height = self._DEFAULT_PANEL_H
        h = self._DEFAULT_PANEL_H
        _resid_default_w = self._RESID_IMAGE_W + 5 + 12 + 38
        self._conv_label.setFixedSize(self._CHART_W, h)
        self._conv_label.setPixmap(QPixmap())
        self._conv_label.setText("Waiting for data\u2026")
        self._resid_label.setFixedSize(_resid_default_w, h)
        self._resid_label.setPixmap(QPixmap())
        self._resid_label.setText("|raw \u2212 est| (final frame)")
        self._sharp_label.setFixedSize(self._CHART_W, h)
        self._sharp_label.setPixmap(QPixmap())
        self._sharp_label.setText("Waiting for data\u2026")

    def set_text(self, text: str) -> None:
        self._text.setPlainText(text)
        self._scroll_to_bottom()

    def append_line(self, line: str) -> None:
        self._text.appendPlainText(line)
        if self._running:
            self._scroll_to_bottom()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self._metrics_check.setEnabled(not self._running)
        if self._running:
            self._text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._scroll_to_bottom()
        else:
            self._text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def _scroll_to_bottom(self) -> None:
        self._text.moveCursor(QTextCursor.MoveOperation.End)

    def set_compute_metrics(self, enabled: bool) -> None:
        self._metrics_check.setChecked(bool(enabled))

    def eventFilter(self, obj, event) -> bool:
        if self._running and obj in (self._text, self._text.viewport()):
            if event.type() in (QEvent.Type.Wheel, QEvent.Type.KeyPress):
                self._scroll_to_bottom()
                return True
        return super().eventFilter(obj, event)

    def _save_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Log",
            str(Path.home() / "cideconvolve_log.txt"),
            "Text files (*.txt);;All files (*)",
        )
        if path:
            Path(path).write_text(self._text.toPlainText(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Composite status-bar panel showing all resource metrics
# ---------------------------------------------------------------------------

class ResourceMonitorBar(QWidget):
    """Status-bar widget with live CPU / RAM / GPU / VRAM bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(2)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        self._cpu_bar = _MetricWidget("CPU", bar_width=70, val_width=44)
        layout.addWidget(self._cpu_bar)

        self._ram_bar = _MetricWidget("RAM", bar_width=70, val_width=100)
        layout.addWidget(self._ram_bar)

        self._swap_bar = _MetricWidget("SWAP", bar_width=50, val_width=58)
        self._swap_bar.setToolTip("System swap / pagefile in use (RAM spillover)")
        layout.addWidget(self._swap_bar)

        # GPU / VRAM — always visible; show N/A when not available
        self._gpu_bar = _MetricWidget("GPU", bar_width=70, val_width=44)
        layout.addWidget(self._gpu_bar)

        self._vram_bar = _MetricWidget("VRAM", bar_width=70, val_width=100)
        layout.addWidget(self._vram_bar)

        self._vspill_bar = _MetricWidget("SPILL", bar_width=50, val_width=58)
        self._vspill_bar.setToolTip(
            "VRAM spillover — GPU memory reserved by PyTorch that exceeds\n"
            "physical VRAM (spills into system RAM / pagefile on Windows)"
        )
        layout.addWidget(self._vspill_bar)

        self._dot = QLabel("●")
        self._dot.setStyleSheet("color: #4CAF50; font-size: 9pt; padding: 0 2px;")
        self._dot.setToolTip("Deconvolution running")
        self._dot.setVisible(False)
        layout.addWidget(self._dot)

        self._has_gpu: bool = False

    def update_metrics(self, m: dict):
        """Slot — called from main thread via signal."""
        # CPU
        self._cpu_bar.set_value(m["cpu_pct"], f"{m['cpu_pct']:.0f}%")

        # System RAM
        if m["ram_total_gb"] > 0:
            ram_pct = m["ram_used_gb"] / m["ram_total_gb"] * 100
            self._ram_bar.set_value(
                ram_pct,
                f"{m['ram_used_gb']:.1f}/{m['ram_total_gb']:.0f} GB",
            )

        # RAM swap / pagefile (always visible)
        swap = m.get("ram_swap_gb", 0.0)
        swap_total = m.get("ram_swap_total_gb", 0.0)
        if swap_total > 0:
            swap_pct = swap / swap_total * 100
            self._swap_bar.set_value(swap_pct, f"{swap:.1f} GB")
        else:
            self._swap_bar.set_value(0, f"{swap:.1f} GB")

        # GPU / VRAM
        has_gpu = bool(m.get("has_gpu"))
        if has_gpu != self._has_gpu:
            self._has_gpu = has_gpu
        gpu_pct = m.get("gpu_pct", 0.0)
        if has_gpu:
            if gpu_pct < 0:
                self._gpu_bar.set_value(0, "N/A %")
            else:
                self._gpu_bar.set_value(gpu_pct, f"{gpu_pct:.0f}%")
            if m["vram_total_gb"] > 0:
                vram_pct = m["vram_used_gb"] / m["vram_total_gb"] * 100
                self._vram_bar.set_value(
                    vram_pct,
                    f"{m['vram_used_gb']:.1f}/{m['vram_total_gb']:.0f} GB",
                )
        else:
            self._gpu_bar.set_value(0, "no GPU")
            self._vram_bar.set_value(0, "no GPU")

        # VRAM spill (always visible) — max is half system RAM on Windows
        vspill = m.get("vram_spill_gb", 0.0)
        spill_max = m.get("ram_total_gb", 0) / 2
        if spill_max > 0:
            spill_pct = min(vspill / spill_max * 100, 100)
            self._vspill_bar.set_value(spill_pct, f"{vspill:.1f} GB")
        else:
            self._vspill_bar.set_value(0, f"{vspill:.1f} GB")

    def set_active(self, active: bool):
        """Show/hide the green activity dot (deconvolution running)."""
        self._dot.setVisible(active)


class CollapsibleSection(QWidget):
    """Simple inline section that can show or hide its child content."""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)
        layout.addWidget(self._content)

        self._on_toggled(expanded)

    def content_layout(self) -> QVBoxLayout:
        """Return the layout that should receive the collapsible content."""
        return self._content_layout

    def _on_toggled(self, expanded: bool):
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(expanded)


class NoWheelComboBox(QComboBox):
    """Combo box that ignores mouse-wheel changes."""

    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Spin box that ignores mouse-wheel changes."""

    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Double spin box that ignores mouse-wheel changes."""

    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


# ---------------------------------------------------------------------------
# Iteration movie helpers
# ---------------------------------------------------------------------------

_MOVIE_MIN_WIDTH = 1200
_MOVIE_MACRO_BLOCK = 16


def _movie_normalize_zyx(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Movie frame expected 2D or 3D data, got {arr.shape}.")


def _movie_project_stack(stack_zyx: np.ndarray, projection: str, z_index: int) -> np.ndarray:
    stack = _movie_normalize_zyx(stack_zyx)
    if projection == "Slice":
        z = max(0, min(int(z_index), stack.shape[0] - 1))
        return stack[z].astype(np.float32, copy=False)
    if projection == "MIP":
        return stack.max(axis=0).astype(np.float32, copy=False)
    if projection == "SUM":
        return stack.sum(axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported movie projection mode: {projection}")


def _movie_project_background(stack_zyx: np.ndarray, projection: str, background: float) -> float:
    stack = _movie_normalize_zyx(stack_zyx)
    bg = max(float(background), 0.0)
    if projection == "SUM":
        return bg * float(stack.shape[0])
    return bg


def _movie_percentile_levels(arr: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[float, float, float]:
    flat = np.asarray(arr).ravel()
    if flat.size == 0:
        return 0.0, 1.0, 1.0
    if flat.size > 1_000_000:
        step = max(1, int(np.ceil(flat.size / 1_000_000)))
        flat = flat[::step]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0, 1.0
    lo, hi = np.percentile(flat, [lo_pct, hi_pct])
    lo_f = float(lo)
    hi_f = float(hi)
    if hi_f <= lo_f:
        hi_f = lo_f + 1.0
    return lo_f, hi_f, 1.0


def _movie_composite_rgb(
    planes: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]],
    *,
    normalize_additive: bool = True,
) -> np.ndarray:
    if not planes:
        return np.zeros((2, 2, 3), dtype=np.uint8)
    height, width = planes[0][0].shape
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    for plane, (cr, cg, cb), (lo, hi, gamma) in planes:
        if hi <= lo:
            hi = lo + 1.0
        norm = (np.asarray(plane, dtype=np.float32) - np.float32(lo)) / np.float32(hi - lo)
        np.clip(norm, 0.0, 1.0, out=norm)
        gamma_safe = max(float(gamma), 1e-3)
        if abs(gamma_safe - 1.0) > 1e-6:
            np.power(norm, 1.0 / gamma_safe, out=norm)
        canvas[..., 0] += norm * (cr / 255.0)
        canvas[..., 1] += norm * (cg / 255.0)
        canvas[..., 2] += norm * (cb / 255.0)
    if normalize_additive:
        finite = canvas[np.isfinite(canvas)]
        if finite.size:
            exposure = float(np.percentile(finite, 99.95))
            if exposure > 1.0:
                canvas /= np.float32(exposure)
    np.clip(canvas, 0.0, 1.0, out=canvas)
    return np.ascontiguousarray((canvas * 255).astype(np.uint8))


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(np.ceil(value / multiple)) * multiple)


def _movie_resize_rgb(
    rgb: np.ndarray,
    min_width: int = _MOVIE_MIN_WIDTH,
    macro_block: int = _MOVIE_MACRO_BLOCK,
) -> np.ndarray:
    height, width = rgb.shape[:2]
    if width >= min_width and width % macro_block == 0 and height % macro_block == 0:
        return np.ascontiguousarray(rgb)
    from PIL import Image

    target_width = _round_up_to_multiple(max(width, min_width), macro_block)
    target_height = _round_up_to_multiple(
        max(1, int(round(height * (target_width / max(width, 1))))),
        macro_block,
    )
    image = Image.fromarray(rgb, mode="RGB")
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def _movie_pad_even_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[0] % 2 or rgb.shape[1] % 2:
        padded = np.zeros((rgb.shape[0] + rgb.shape[0] % 2, rgb.shape[1] + rgb.shape[1] % 2, 3), dtype=np.uint8)
        padded[:rgb.shape[0], :rgb.shape[1], :] = rgb
        rgb = padded
    return np.ascontiguousarray(rgb)


def _movie_letterbox_rgb(rgb: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    if rgb.shape[0] == target_h and rgb.shape[1] == target_w:
        return np.ascontiguousarray(rgb)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = max(0, (target_h - rgb.shape[0]) // 2)
    x0 = max(0, (target_w - rgb.shape[1]) // 2)
    h = min(rgb.shape[0], target_h)
    w = min(rgb.shape[1], target_w)
    canvas[y0:y0 + h, x0:x0 + w, :] = rgb[:h, :w, :]
    return np.ascontiguousarray(canvas)


def _movie_fit_rgb_to_box(rgb: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    from PIL import Image

    source_h, source_w = rgb.shape[:2]
    scale = max(target_width / max(source_w, 1), target_height / max(source_h, 1))
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    image = Image.fromarray(rgb, mode="RGB").resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.uint8)
    y0 = max(0, (resized_h - target_height) // 2)
    x0 = max(0, (resized_w - target_width) // 2)
    return np.ascontiguousarray(arr[y0:y0 + target_height, x0:x0 + target_width, :])


def _movie_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _movie_text_size(draw, text: str, font) -> tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    except Exception:
        return draw.textsize(text, font=font)


def _movie_overlay_rgb(
    rgb: np.ndarray,
    *,
    title: str,
    bottom_lines: list[str],
    convergence_series: Optional[list[Optional[float]]] = None,
    convergence_total_points: Optional[int] = None,
    convergence_log_scale: bool = False,
) -> np.ndarray:
    if not title and not bottom_lines and not convergence_series:
        return rgb
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    pad = max(8, min(width, height) // 80)
    title_font = _movie_font(max(15, min(width, height) // 44), bold=True)
    info_font = _movie_font(max(11, min(width, height) // 62), bold=False)

    if title:
        title_text = title.strip()
        tw, th = _movie_text_size(draw, title_text, title_font)
        x = max(pad, (width - tw) // 2)
        y = pad
        draw.rounded_rectangle(
            (x - pad, y - pad // 2, x + tw + pad, y + th + pad),
            radius=max(4, pad // 2),
            fill=(0, 0, 0, 150),
        )
        draw.text((x, y), title_text, font=title_font, fill=(255, 255, 255, 235))

    if bottom_lines:
        sizes = [_movie_text_size(draw, line, info_font) for line in bottom_lines]
        line_gap = max(2, pad // 4)
        block_w = max((size[0] for size in sizes), default=0)
        block_h = sum(size[1] for size in sizes) + line_gap * max(len(sizes) - 1, 0)
        x = pad
        y = max(pad, height - block_h - pad)
        draw.rounded_rectangle(
            (x - pad // 2, y - pad // 2, min(width - pad // 2, x + block_w + pad), y + block_h + pad // 2),
            radius=max(4, pad // 2),
            fill=(0, 0, 0, 155),
        )
        cursor_y = y
        for line, (_, line_h) in zip(bottom_lines, sizes):
            draw.text((x, cursor_y), line, font=info_font, fill=(255, 255, 255, 230))
            cursor_y += line_h + line_gap

    if convergence_series:
        values = [
            float(value) for value in convergence_series
            if value is not None and np.isfinite(float(value))
        ]
        if values:
            if convergence_log_scale:
                plotted = [np.log10(max(value, 1e-12)) for value in values]
            else:
                plotted = values
            plot_w = min(max(220, width // 5), width - 2 * pad)
            plot_h = min(max(110, height // 7), height - 2 * pad)
            x0 = width - plot_w - pad
            y0 = height - plot_h - pad
            x1 = width - pad
            y1 = height - pad
            draw.rounded_rectangle(
                (x0, y0, x1, y1),
                radius=max(4, pad // 2),
                fill=(0, 0, 0, 155),
            )
            axis_pad = max(10, pad)
            px0 = x0 + axis_pad
            py0 = y0 + axis_pad
            px1 = x1 - axis_pad // 2
            py1 = y1 - axis_pad
            draw.line((px0, py1, px1, py1), fill=(255, 255, 255, 95), width=1)
            draw.line((px0, py0, px0, py1), fill=(255, 255, 255, 95), width=1)
            vmin = min(plotted)
            vmax = max(plotted)
            flat_series = vmax <= vmin
            if flat_series:
                vmax = vmin + 1.0
            points: list[tuple[float, float]] = []
            denom_x = max(int(convergence_total_points or len(convergence_series)) - 1, 1)
            for idx, value in enumerate(convergence_series):
                if value is None or not np.isfinite(float(value)):
                    continue
                plot_value = np.log10(max(float(value), 1e-12)) if convergence_log_scale else float(value)
                norm = 1.0 if flat_series else (plot_value - vmin) / (vmax - vmin)
                x = px0 + (px1 - px0) * (idx / denom_x)
                y = py1 - (py1 - py0) * norm
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=(117, 211, 255, 235), width=max(2, pad // 4), joint="curve")
            if points:
                r = max(3, pad // 4)
                x, y = points[-1]
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 245))
            label = "log convergence" if convergence_log_scale else "convergence"
            lw, lh = _movie_text_size(draw, label, info_font)
            draw.text((x0 + axis_pad, y0 + max(4, axis_pad // 3)), label, font=info_font, fill=(255, 255, 255, 210))
            latest = f"{values[-1]:.3g}"
            vw, _ = _movie_text_size(draw, latest, info_font)
            draw.text((x1 - axis_pad - vw, y0 + max(4, axis_pad // 3)), latest, font=info_font, fill=(255, 255, 255, 210))

    composed = Image.alpha_composite(image, overlay).convert("RGB")
    return np.ascontiguousarray(np.asarray(composed, dtype=np.uint8))


def _movie_difference_inset_rgb(
    rgb: np.ndarray,
    current_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    mode: str,
) -> np.ndarray:
    mode = str(mode or "None")
    if mode == "None":
        return rgb
    from PIL import Image, ImageDraw

    current = np.asarray(current_rgb, dtype=np.float32)
    reference = np.asarray(reference_rgb, dtype=np.float32)
    if current.shape != reference.shape:
        ref_image = Image.fromarray(np.asarray(reference_rgb, dtype=np.uint8), mode="RGB")
        ref_image = ref_image.resize((current.shape[1], current.shape[0]), Image.Resampling.BILINEAR)
        reference = np.asarray(ref_image, dtype=np.float32)

    if mode.startswith("Ratio"):
        current_luma = (
            0.2126 * current[..., 0]
            + 0.7152 * current[..., 1]
            + 0.0722 * current[..., 2]
        )
        reference_luma = (
            0.2126 * reference[..., 0]
            + 0.7152 * reference[..., 1]
            + 0.0722 * reference[..., 2]
        )
        signal = np.maximum(current_luma, reference_luma)
        finite_signal = signal[np.isfinite(signal)]
        signal_floor = float(np.percentile(finite_signal, 35.0)) if finite_signal.size else 0.0
        signal_mask = signal > max(signal_floor, 1.0)
        eps = max(float(np.percentile(finite_signal, 99.0)) * 0.01 if finite_signal.size else 1.0, 1.0)
        data_2d = np.log2((current_luma + eps) / (reference_luma + eps))
        finite = data_2d[signal_mask & np.isfinite(data_2d)]
        if finite.size:
            center = float(np.median(finite))
            spread = max(float(np.percentile(np.abs(finite - center), 98.0)), 1e-6)
        else:
            center = 0.0
            spread = 1.0
        norm_2d = np.clip((data_2d - center) / spread, -1.0, 1.0)
        norm_2d[~signal_mask] = 0.0
        norm = np.repeat(norm_2d[:, :, np.newaxis], 3, axis=2)
        label = mode
    else:
        data = current - reference
        finite = data[np.isfinite(data)]
        spread = max(float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0, 1e-6)
        norm = np.clip(data / spread, -1.0, 1.0)
        label = mode

    inset_rgb = np.zeros_like(current, dtype=np.uint8)
    pos = np.clip(norm, 0.0, 1.0)
    neg = np.clip(-norm, 0.0, 1.0)
    inset_rgb[..., 0] = np.clip(pos.max(axis=2) * 255.0, 0, 255).astype(np.uint8)
    inset_rgb[..., 1] = np.clip(neg.max(axis=2) * 180.0, 0, 255).astype(np.uint8)
    inset_rgb[..., 2] = np.clip(neg.max(axis=2) * 255.0, 0, 255).astype(np.uint8)

    base = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size
    pad = max(8, min(width, height) // 80)
    inset_w = min(max(220, width // 5), width - 2 * pad)
    inset_h = max(80, int(inset_w * current.shape[0] / max(current.shape[1], 1)))
    inset_h = min(inset_h, max(80, height // 5))
    x0 = width - inset_w - pad
    y0 = pad
    inset = Image.fromarray(inset_rgb, mode="RGB").resize((inset_w, inset_h), Image.Resampling.BILINEAR)
    draw.rounded_rectangle(
        (x0 - pad // 2, y0 - pad // 2, x0 + inset_w + pad // 2, y0 + inset_h + pad * 2),
        radius=max(4, pad // 2),
        fill=(0, 0, 0, 155),
    )
    overlay.alpha_composite(inset.convert("RGBA"), (x0, y0))
    font = _movie_font(max(11, min(width, height) // 62), bold=False)
    draw.text((x0, y0 + inset_h + max(2, pad // 4)), label, font=font, fill=(255, 255, 255, 220))
    return np.ascontiguousarray(np.asarray(Image.alpha_composite(base, overlay).convert("RGB"), dtype=np.uint8))


def _movie_raw_estimated_ratio_inset_rgb(
    rgb: np.ndarray,
    observed_plane: np.ndarray,
    estimated_plane: np.ndarray,
    background: float,
    mode: str = "ratio",
) -> np.ndarray:
    """Overlay a mini-panel comparing raw vs PSF*deconvolved using a glow colormap.

    Both modes render the *magnitude* of the mismatch with the 'inferno' glow
    colormap (black = perfect match / zero error, bright = large error).  A
    colorbar strip on the right shows the absolute scale so the panel naturally
    fades toward black as the algorithm converges.

    mode="ratio"      — |log2(observed/estimated)|, auto-normalised per frame.
    mode="difference" — |observed − estimated| / p99(raw).  Fixed scale so
                        brightness genuinely decreases toward zero at convergence.
    """
    from PIL import Image, ImageDraw
    import matplotlib

    observed = np.asarray(observed_plane, dtype=np.float32)
    estimated = np.asarray(estimated_plane, dtype=np.float32)
    if observed.shape != estimated.shape:
        est_image = Image.fromarray(estimated, mode="F")
        est_image = est_image.resize((observed.shape[1], observed.shape[0]), Image.Resampling.BILINEAR)
        estimated = np.asarray(est_image, dtype=np.float32)

    bg = max(float(background), 0.0)
    observed_signal = np.clip(observed - bg, 0.0, None)
    estimated_signal = np.clip(estimated - bg, 0.0, None)
    # Base signal mask and scale exclusively on the *observed* (raw) signal so
    # that border artefacts produced by the DL model (high estimated values
    # outside the real sample) do not inflate the normalisation scale.
    obs_finite = observed_signal[np.isfinite(observed_signal)]
    signal_floor = max(float(np.percentile(obs_finite, 35.0)), 1e-6) if obs_finite.size else 1e-6
    signal_mask = observed_signal > signal_floor

    cmap = matplotlib.colormaps["inferno"]

    if mode == "difference":
        raw_scale = max(float(np.percentile(obs_finite, 99.0)), 1e-6) if obs_finite.size else 1.0
        abs_diff = np.abs(observed_signal - estimated_signal)
        mag = np.clip(abs_diff / raw_scale, 0.0, 1.0)
        mag[~signal_mask] = 0.0
        label = "Raw \u2212 est  (abs)"
        cbar_top_str = f"{raw_scale:.0f}"
        cbar_mid_str = f"{raw_scale / 2.0:.0f}"
    else:
        eps = max(float(np.percentile(obs_finite, 99.0)) * 0.01, 1e-6) if obs_finite.size else 1e-6
        ratio_log2 = np.log2((observed_signal + eps) / (estimated_signal + eps))
        finite_ratio = ratio_log2[signal_mask & np.isfinite(ratio_log2)]
        if finite_ratio.size:
            spread = max(float(np.percentile(np.abs(finite_ratio), 98.0)), 1e-6)
        else:
            spread = 1.0
        mag = np.clip(np.abs(ratio_log2) / spread, 0.0, 1.0)
        mag[~signal_mask] = 0.0
        label = "Raw / est  (log2)"
        cbar_top_str = f"{spread:.2f}"
        cbar_mid_str = f"{spread / 2.0:.2f}"

    # Apply inferno glow: black=0 (no error), yellow/white=1 (max error)
    rgba_mapped = cmap(mag)  # (H, W, 4) float 0–1
    inset_rgb = (rgba_mapped[..., :3] * 255.0).astype(np.uint8)
    inset_rgb[~signal_mask] = 0

    base = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size
    pad = max(8, min(width, height) // 80)

    font = _movie_font(max(11, min(width, height) // 62), bold=False)
    font_small = _movie_font(max(9, min(width, height) // 75), bold=False)
    font_sz = max(9, min(width, height) // 75)

    cbar_w = max(12, pad + 4)
    cbar_gap = max(3, pad // 3)
    cbar_label_w = max(40, pad * 5)

    inset_w = min(
        max(180, width // 5),
        max(60, width - 2 * pad - cbar_gap - cbar_w - cbar_label_w),
    )
    inset_h = max(80, int(inset_w * observed.shape[0] / max(observed.shape[1], 1)))
    inset_h = min(inset_h, max(80, height // 5))

    total_w = inset_w + cbar_gap + cbar_w + cbar_label_w
    x0 = width - total_w - pad
    y0 = pad
    label_row_h = max(16, font_sz + 6)

    # Dark background rectangle covering inset + colorbar + labels
    draw.rounded_rectangle(
        (x0 - pad // 2, y0 - pad // 2,
         x0 + total_w + pad // 2, y0 + inset_h + label_row_h + pad),
        radius=max(4, pad // 2),
        fill=(0, 0, 0, 155),
    )

    # Inset glow image
    inset = Image.fromarray(inset_rgb, mode="RGB").resize((inset_w, inset_h), Image.Resampling.BILINEAR)
    overlay.alpha_composite(inset.convert("RGBA"), (x0, y0))

    # Colorbar gradient strip: top = max (bright), bottom = 0 (black)
    cbar_x = x0 + inset_w + cbar_gap
    cbar_vals = np.linspace(1.0, 0.0, inset_h).reshape(-1, 1)
    cbar_rgba = cmap(cbar_vals)
    cbar_strip = (cbar_rgba[..., :3] * 255.0).astype(np.uint8)
    cbar_strip = np.repeat(cbar_strip, cbar_w, axis=1)
    overlay.alpha_composite(
        Image.fromarray(cbar_strip, mode="RGB").convert("RGBA"),
        (cbar_x, y0),
    )

    # Colorbar border
    draw.rectangle(
        (cbar_x, y0, cbar_x + cbar_w - 1, y0 + inset_h - 1),
        outline=(200, 200, 200, 180),
        width=1,
    )

    # Tick marks and labels: top, mid, bottom
    txt_x = cbar_x + cbar_w + max(2, pad // 4)
    tick_col = (220, 220, 220, 200)
    lbl_col = (255, 240, 200, 220)
    # Top tick
    draw.line([(cbar_x - 3, y0 + 1), (cbar_x, y0 + 1)], fill=tick_col, width=1)
    draw.text((txt_x, y0), cbar_top_str, font=font_small, fill=lbl_col)
    # Mid tick
    mid_y = y0 + inset_h // 2
    draw.line([(cbar_x - 3, mid_y), (cbar_x, mid_y)], fill=tick_col, width=1)
    draw.text((txt_x, mid_y - font_sz // 2), cbar_mid_str, font=font_small, fill=(220, 220, 200, 200))
    # Bottom tick
    bot_y = y0 + inset_h - 1
    draw.line([(cbar_x - 3, bot_y), (cbar_x, bot_y)], fill=tick_col, width=1)
    draw.text((txt_x, bot_y - font_sz), "0", font=font_small, fill=lbl_col)

    # Label below inset
    draw.text((x0, y0 + inset_h + max(2, pad // 4)), label, font=font, fill=(255, 255, 255, 220))

    return np.ascontiguousarray(
        np.asarray(Image.alpha_composite(base, overlay).convert("RGB"), dtype=np.uint8)
    )


def _movie_metrics_line(channels: list[np.ndarray]) -> str:
    metrics = _quality_metrics(channels)
    return (
        f"detail={metrics['detail_energy_mean']:.3f}  "
        f"bright={metrics['bright_detail_energy_mean']:.3f}  "
        f"edge={metrics['edge_strength_mean']:.3f}  "
        f"sparse={metrics['signal_sparsity_mean']:.3f}  "
        f"range={metrics['robust_range_mean']:.3f}"
    )


# ---------------------------------------------------------------------------
# Live log-panel chart renderers (standalone QPixmap helpers)
# ---------------------------------------------------------------------------

def _render_convergence_chart_pixmap(
    series: list[Optional[float]],
    total_points: int,
    log_scale: bool,
    width: int = 260,
    height: int = 130,
) -> QPixmap:
    """Render a convergence line chart as a QPixmap for the live log panel."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (22, 22, 28))
    draw = ImageDraw.Draw(img)
    pad = max(6, min(width, height) // 20)
    font = _movie_font(max(7, min(width, height) // 20), bold=False)

    values = [float(v) for v in series if v is not None and np.isfinite(float(v))]
    # Log scale only makes sense for strictly positive values; fall back to
    # linear when any value is ≤ 0 (e.g. negative Poisson NLL from sparse hessian).
    use_log = log_scale and bool(values) and all(v > 0 for v in values)
    label = "log convergence" if use_log else "convergence"
    draw.text((pad, pad // 2), label, font=font, fill=(170, 170, 185))

    if not values:
        draw.text((pad, height // 2 - 6), "Waiting for data\u2026", font=font, fill=(110, 110, 130))
    else:
        plotted = [np.log10(v) for v in values] if use_log else values
        px0, py0 = pad, pad + 18
        px1, py1 = width - pad, height - pad
        draw.line([(px0, py1), (px1, py1)], fill=(70, 70, 90), width=1)
        draw.line([(px0, py0), (px0, py1)], fill=(70, 70, 90), width=1)
        vmin, vmax_v = min(plotted), max(plotted)
        if vmax_v <= vmin:
            vmax_v = vmin + 1.0
        denom_x = max(int(total_points or len(series)) - 1, 1)
        points: list[tuple[float, float]] = []
        for idx, sv in enumerate(series):
            if sv is None or not np.isfinite(float(sv)):
                continue
            pv = np.log10(float(sv)) if use_log else float(sv)
            norm = (pv - vmin) / (vmax_v - vmin)
            px = px0 + (px1 - px0) * (idx / denom_x)
            py = py1 - (py1 - py0) * norm
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=(117, 211, 255), width=max(2, pad // 3), joint="curve")
        if points:
            r = max(3, pad // 3)
            lx, ly = points[-1]
            draw.ellipse((lx - r, ly - r, lx + r, ly + r), fill=(255, 255, 255))
        latest = f"{values[-1]:.3g}"
        lw, _ = _movie_text_size(draw, latest, font)
        draw.text((px1 - lw, pad // 2), latest, font=font, fill=(255, 255, 200))

    raw = img.tobytes("raw", "RGB")
    qimg = QImage(raw, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _render_residual_pixmap(
    raw_plane: np.ndarray,
    est_plane: np.ndarray,
    max_image_w: int = 230,
) -> tuple["QPixmap", int, int]:
    """Render |raw − est| as an inferno-coloured QPixmap with a colorbar.

    Height is derived from the actual image aspect ratio so the heatmap
    matches the data geometry.  Returns ``(pixmap, total_width, total_height)``.
    """
    from PIL import Image, ImageDraw

    diff = np.abs(
        np.asarray(raw_plane, dtype=np.float32) - np.asarray(est_plane, dtype=np.float32)
    )
    finite = diff[np.isfinite(diff)]
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    norm = np.clip(diff / vmax, 0.0, 1.0)

    try:
        import matplotlib as _mpl
        _cmap = _mpl.colormaps["inferno"]
        rgba = _cmap(norm)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        _use_mpl = True
    except Exception:
        _use_mpl = False
        r = np.clip(norm * 220 + norm ** 2 * 35, 0, 255).astype(np.uint8)
        g = np.clip(norm ** 2 * 110, 0, 255).astype(np.uint8)
        b = np.clip(norm * 200 * (1.0 - norm * 0.5), 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)

    # Preserve image aspect ratio
    h_img, w_img = diff.shape[:2]
    aspect = w_img / max(h_img, 1)
    content_h = int(round(max_image_w / aspect))
    content_h = max(60, min(content_h, 280))

    # Colorbar layout: gap | bar | gap | labels
    CBAR_GAP = 5
    CBAR_W   = 12
    LABEL_W  = 38
    total_w  = max_image_w + CBAR_GAP + CBAR_W + LABEL_W
    total_h  = content_h

    content_img = Image.fromarray(rgb, "RGB").resize(
        (max_image_w, content_h), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (total_w, total_h), (22, 22, 28))
    canvas.paste(content_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    font_sm = _movie_font(max(6, content_h // 23), bold=False)

    # Draw colorbar gradient strip
    bar_x0 = max_image_w + CBAR_GAP
    bar_x1 = bar_x0 + CBAR_W
    for row in range(total_h):
        t = 1.0 - row / max(total_h - 1, 1)   # 1 → vmax (top), 0 → 0 (bottom)
        if _use_mpl:
            cr, cg, cb, _ = _cmap(t)
            cr, cg, cb = int(cr * 255), int(cg * 255), int(cb * 255)
        else:
            cr = int(min(t * 220 + t ** 2 * 35, 255))
            cg = int(min(t ** 2 * 110, 255))
            cb = int(min(t * 200 * (1.0 - t * 0.5), 255))
        draw.line([(bar_x0, row), (bar_x1, row)], fill=(cr, cg, cb))
    draw.rectangle([bar_x0, 0, bar_x1 - 1, total_h - 1], outline=(80, 80, 95))

    # Scale labels
    def _fmt_val(v: float) -> str:
        if v == 0.0:
            return "0"
        if abs(v) >= 1e4 or abs(v) < 0.01:
            return f"{v:.1e}"
        if abs(v) >= 100:
            return f"{v:.0f}"
        return f"{v:.2g}"

    label_x = bar_x1 + 2
    _, lh = _movie_text_size(draw, "0", font_sm)
    draw.text((label_x, 0),                            _fmt_val(vmax),       font=font_sm, fill=(220, 220, 200))
    draw.text((label_x, total_h // 2 - lh // 2),      _fmt_val(vmax * 0.5), font=font_sm, fill=(180, 180, 165))
    draw.text((label_x, max(total_h - lh - 1, 0)),    "0",                  font=font_sm, fill=(130, 130, 115))

    # Title overlay
    draw.text((3, 1), "|raw\u2212est|", font=font_sm, fill=(200, 200, 200))

    raw_bytes = canvas.tobytes("raw", "RGB")
    qimg = QImage(raw_bytes, total_w, total_h, total_w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg), total_w, total_h


def _compute_sharpness(image_arr: np.ndarray) -> Optional[float]:
    """Return the Laplacian-variance of the central Z-plane — a lightweight sharpness proxy.

    Downsamples to ≤ 512 px per axis before computing so the overhead per
    iteration is well under 5 ms even on large 3-D stacks.
    """
    arr = np.asarray(image_arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]                       # CZYX → ZYX
    if arr.ndim == 3:
        plane = arr[arr.shape[0] // 2]
    elif arr.ndim == 2:
        plane = arr
    else:
        return None
    if plane.shape[0] > 512 or plane.shape[1] > 512:
        plane = plane[::2, ::2]            # 2× downsample
    # Discrete 5-point Laplacian
    lap = (
        plane[:-2, 1:-1] + plane[2:, 1:-1]
        + plane[1:-1, :-2] + plane[1:-1, 2:]
        - 4.0 * plane[1:-1, 1:-1]
    )
    return float(np.var(lap))


def _render_sharpness_chart_pixmap(
    series: list[Optional[float]],
    total_points: int,
    width: int = 260,
    height: int = 130,
) -> QPixmap:
    """Render a per-iteration Laplacian-variance sharpness trend as a QPixmap."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (22, 22, 28))
    draw = ImageDraw.Draw(img)
    pad = max(6, min(width, height) // 20)
    font = _movie_font(max(7, min(width, height) // 20), bold=False)
    draw.text((pad, pad // 2), "sharpness (Laplacian var)", font=font, fill=(155, 220, 155))

    values = [float(v) for v in series if v is not None and np.isfinite(float(v))]
    if not values:
        draw.text((pad, height // 2 - 6), "Waiting for data\u2026", font=font, fill=(100, 140, 100))
    else:
        px0, py0 = pad, pad + 18
        px1, py1 = width - pad, height - pad
        draw.line([(px0, py1), (px1, py1)], fill=(70, 70, 90), width=1)
        draw.line([(px0, py0), (px0, py1)], fill=(70, 70, 90), width=1)
        vmin, vmax_v = min(values), max(values)
        if vmax_v <= vmin:
            vmax_v = vmin + 1.0
        denom_x = max(int(total_points or len(series)) - 1, 1)
        points: list[tuple[float, float]] = []
        for idx, sv in enumerate(series):
            if sv is None or not np.isfinite(float(sv)):
                continue
            norm = (float(sv) - vmin) / (vmax_v - vmin)
            px = px0 + (px1 - px0) * (idx / denom_x)
            py = py1 - (py1 - py0) * norm
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=(100, 220, 130), width=max(2, pad // 3), joint="curve")
        if points:
            r = max(3, pad // 3)
            lx, ly = points[-1]
            draw.ellipse((lx - r, ly - r, lx + r, ly + r), fill=(255, 255, 255))
        latest = f"{values[-1]:.3g}"
        lw, _ = _movie_text_size(draw, latest, font)
        draw.text((px1 - lw, pad // 2), latest, font=font, fill=(200, 255, 200))

    raw = img.tobytes("raw", "RGB")
    qimg = QImage(raw, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class _IterationMovieRecorder:
    """Stage projected iteration planes and encode a composite MP4 at the end."""

    def __init__(
        self,
        initial_channels: list[np.ndarray],
        render_state: dict[str, object],
        niter_list: list[int],
        output_path: str,
        fps: int,
        method: str,
        start_mode: str,
        title_text: str,
        show_info_metrics: bool,
        hold_endpoints: bool,
        layout_mode: str,
        difference_inset: str,
        convergence_log_scale: bool,
        progress_cb: Callable[[str], None],
    ) -> None:
        self.output_path = output_path
        self.fps = int(fps)
        self.method = str(method)
        self.start_mode = str(start_mode or "").strip()
        self.title_text = str(title_text or "").strip()
        self.show_info_metrics = bool(show_info_metrics)
        self.hold_endpoints = bool(hold_endpoints)
        self.layout_mode = str(layout_mode or "Standard")
        self.difference_inset = str(difference_inset or "None")
        self.convergence_log_scale = bool(convergence_log_scale)
        self.progress_cb = progress_cb
        self.projection = str(render_state.get("projection", "Slice"))
        self.z_index = int(render_state.get("z_index", 0))
        self.active_channels = [int(v) for v in render_state.get("active_channels", [])]
        self.channel_colors = [
            tuple(int(c) for c in color)
            for color in render_state.get("channel_colors", [])
        ]
        self.advanced_scaling_active = bool(render_state.get("advanced_scaling_active", False))
        self.channel_scaling = list(render_state.get("channel_scaling", []))
        self.lo_pct = float(render_state.get("lo_percentile", 0.1))
        self.hi_pct = float(render_state.get("hi_percentile", 100.0))
        self.temp_dir = tempfile.mkdtemp(prefix="cideconvolve_movie_")
        self.initial_planes: dict[int, np.ndarray] = {}
        self.initial_levels: dict[int, tuple[float, float, float]] = {}
        self.memmaps: dict[int, np.memmap] = {}
        self.estimated_memmaps: dict[int, np.memmap] = {}
        self.iter_counts: dict[int, int] = {}
        self.frame_labels: dict[tuple[int, int], str] = {}
        self.frame_levels: dict[tuple[int, int], tuple[float, float, float]] = {}
        self.frame_convergence: dict[tuple[int, int], Optional[float]] = {}
        self.frame_background: dict[tuple[int, int], float] = {}
        self._closed = False

        if not self.active_channels:
            raise ValueError("Movie export has no visible channels to render.")

        for ch_idx in self.active_channels:
            if ch_idx >= len(initial_channels):
                continue
            initial = _movie_normalize_zyx(initial_channels[ch_idx])
            initial_plane = np.ascontiguousarray(_movie_project_stack(initial, self.projection, self.z_index))
            self.initial_planes[ch_idx] = initial_plane.astype(np.float32, copy=False)
            self.initial_levels[ch_idx] = self._levels_for(ch_idx, self.initial_planes[ch_idx], "original", initial)
            niter = niter_list[ch_idx] if ch_idx < len(niter_list) else niter_list[-1]
            staged_frames = max(int(niter), 1) + (1 if self.method == "ci_rl_dl" else 0)
            stage_path = Path(self.temp_dir) / f"channel_{ch_idx:03d}.dat"
            self.memmaps[ch_idx] = np.memmap(
                stage_path,
                dtype=np.float32,
                mode="w+",
                shape=(staged_frames, initial_plane.shape[0], initial_plane.shape[1]),
            )
            estimated_path = Path(self.temp_dir) / f"channel_{ch_idx:03d}_estimated.dat"
            self.estimated_memmaps[ch_idx] = np.memmap(
                estimated_path,
                dtype=np.float32,
                mode="w+",
                shape=(staged_frames, initial_plane.shape[0], initial_plane.shape[1]),
            )
            self.iter_counts[ch_idx] = 0

    def _levels_for(
        self,
        ch_idx: int,
        plane: np.ndarray,
        pane: str,
        contrast_source: Optional[np.ndarray] = None,
    ) -> tuple[float, float, float]:
        if self.advanced_scaling_active and ch_idx < len(self.channel_scaling):
            state = dict(self.channel_scaling[ch_idx])
            pane_levels = dict(state.get(pane) or {})
            pane_max = max(
                float(pane_levels.get("max", 0.0)),
                float(np.nanmax(plane)) if plane.size else 0.0,
                0.0,
            )
            lo = min(max(float(pane_levels.get("min", 0.0)), 0.0), pane_max)
            hi = min(max(float(pane_levels.get("max", pane_max)), lo), pane_max)
            gamma = min(max(float(state.get("gamma", 1.0)), 0.10), 5.00)
            return lo, hi, gamma
        source = plane
        if contrast_source is not None and self.projection != "SUM":
            source = contrast_source
        return _movie_percentile_levels(source, self.lo_pct, self.hi_pct)

    def capture(self, payload: dict[str, Any]) -> None:
        ch_idx = int(payload.get("channel_index", 0))
        if ch_idx not in self.memmaps:
            return
        iteration = max(1, int(payload.get("iteration", 1)))
        mm = self.memmaps[ch_idx]
        if iteration > mm.shape[0]:
            return
        stack = _movie_normalize_zyx(payload["image"])
        plane = np.ascontiguousarray(
            _movie_project_stack(stack, self.projection, self.z_index)
        )
        mm[iteration - 1, :, :] = plane
        if ch_idx in self.estimated_memmaps and "estimated" in payload:
            estimated_stack = _movie_normalize_zyx(payload["estimated"])
            estimated_plane = np.ascontiguousarray(
                _movie_project_stack(estimated_stack, self.projection, self.z_index)
            )
            self.estimated_memmaps[ch_idx][iteration - 1, :, :] = estimated_plane
            bg = _movie_project_background(
                estimated_stack,
                self.projection,
                float(payload.get("background", 0.0)),
            )
            self.frame_background[(ch_idx, iteration - 1)] = bg
        self.iter_counts[ch_idx] = max(self.iter_counts.get(ch_idx, 0), iteration)
        self.frame_levels[(ch_idx, iteration - 1)] = self._levels_for(ch_idx, plane, "deconvolved", stack)
        convergence = payload.get("convergence")
        if convergence is not None:
            self.frame_convergence[(ch_idx, iteration - 1)] = float(convergence)
        label = str(payload.get("stage_label") or "").strip()
        if label:
            self.frame_labels[(ch_idx, iteration - 1)] = label
        if iteration == 1 or iteration % 10 == 0 or bool(payload.get("is_final", False)):
            if label:
                self.progress_cb(f"  Movie staged Ch{ch_idx + 1} {label}")
            else:
                self.progress_cb(f"  Movie staged Ch{ch_idx + 1} iteration {iteration}")

    def _latest_convergence_for_channel(self, ch_idx: int, frame_idx: int) -> Optional[float]:
        count = self.iter_counts.get(ch_idx, 0)
        if count <= 0:
            return None
        latest_frame = min(frame_idx, count - 1)
        for idx in range(latest_frame, -1, -1):
            value = self.frame_convergence.get((ch_idx, idx))
            if value is not None:
                return value
        return None

    def _convergence_values_at(self, frame_idx: int) -> list[float]:
        return [
            value
            for ch_idx in self.active_channels
            for value in [self._latest_convergence_for_channel(ch_idx, frame_idx)]
            if value is not None
        ]

    def _convergence_series_for(self, frame_idx: int) -> list[Optional[float]]:
        series: list[Optional[float]] = []
        for idx in range(frame_idx + 1):
            values = self._convergence_values_at(idx)
            series.append(float(np.mean(values)) if values else None)
        return series

    def _bottom_lines_for(self, frame_idx: int, frame_count: int, channels: list[np.ndarray]) -> list[str]:
        labels = [
            self.frame_labels.get((ch_idx, frame_idx))
            for ch_idx in self.active_channels
            if self.frame_labels.get((ch_idx, frame_idx))
        ] if frame_idx >= 0 else []
        if labels:
            label = labels[0]
            if self.show_info_metrics:
                return [
                    f"{self.method}  {label}  frame {frame_idx + 1}/{frame_count}",
                    _movie_metrics_line(channels),
                ]
            return [f"{self.method}  {label}"]
        if not self.show_info_metrics:
            return []
        if frame_idx < 0:
            return [
                f"{self.method}  original input",
                _movie_metrics_line(channels),
            ]
        convergence_values = self._convergence_values_at(frame_idx)
        convergence_text = (
            f"{float(np.mean(convergence_values)):.6g}"
            if convergence_values
            else "pending"
        )
        start_text = f"  start={self.start_mode}" if frame_idx == 0 and self.start_mode else ""
        return [
            f"{self.method}  iteration {frame_idx + 1}/{frame_count}{start_text}  convergence {convergence_text}",
            _movie_metrics_line(channels),
        ]

    def _levels_for_frame(self, ch_idx: int, frame_idx: int) -> tuple[float, float, float]:
        if (ch_idx, frame_idx) in self.frame_levels:
            return self.frame_levels[(ch_idx, frame_idx)]
        return self.initial_levels[ch_idx]

    def _frame_payload(self, frame_idx: int) -> tuple[list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]], list[np.ndarray]]:
        planes = []
        metric_channels = []
        for ch_idx in self.active_channels:
            color = self.channel_colors[ch_idx] if ch_idx < len(self.channel_colors) else _FALLBACK_COLORS[ch_idx % len(_FALLBACK_COLORS)]
            count = self.iter_counts.get(ch_idx, 0)
            if frame_idx < 0:
                plane = self.initial_planes[ch_idx]
                levels = self.initial_levels[ch_idx]
            elif count <= 0:
                plane = self.initial_planes[ch_idx]
                levels = self.initial_levels[ch_idx]
            elif frame_idx >= count:
                plane = self.memmaps[ch_idx][count - 1]
                levels = self._levels_for_frame(ch_idx, count - 1)
            else:
                plane = self.memmaps[ch_idx][frame_idx]
                levels = self._levels_for_frame(ch_idx, frame_idx)
            planes.append((plane, color, levels))
            metric_channels.append(np.asarray(plane, dtype=np.float32))
        return planes, metric_channels

    def _standard_rgb_for_frame(self, frame_idx: int) -> tuple[np.ndarray, list[np.ndarray]]:
        planes, metric_channels = self._frame_payload(frame_idx)
        return _movie_composite_rgb(planes), metric_channels

    def _final_frame_index(self, frame_count: int) -> int:
        return max(0, frame_count - 1)

    def _sequence_frame_indices(self, frame_count: int) -> list[int]:
        hold_frames = max(1, int(round(self.fps))) if self.hold_endpoints else 1
        indices = [-1] * hold_frames
        indices.extend(range(frame_count))
        if frame_count > 0 and hold_frames > 1:
            indices.extend([frame_count - 1] * (hold_frames - 1))
        return indices

    def _layout_sequence(self) -> list[str]:
        mode = self.layout_mode.strip().lower()
        if mode == "split-screen":
            return ["split"]
        if mode == "standard + split-screen":
            return ["standard", "split"]
        return ["standard"]

    def _difference_reference_rgb(self, frame_count: int) -> Optional[np.ndarray]:
        mode = self.difference_inset.strip().lower()
        if mode == "none" or "estimated" in mode:
            return None
        if "final" in mode:
            return self._standard_rgb_for_frame(self._final_frame_index(frame_count))[0]
        return self._standard_rgb_for_frame(-1)[0]

    def _raw_estimated_ratio_payload(self, frame_idx: int) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
        observed_planes: list[np.ndarray] = []
        estimated_planes: list[np.ndarray] = []
        backgrounds: list[float] = []
        for ch_idx in self.active_channels:
            count = self.iter_counts.get(ch_idx, 0)
            if count <= 0 or ch_idx not in self.estimated_memmaps:
                continue
            idx = min(max(frame_idx, 0), count - 1)
            key = (ch_idx, idx)
            if key not in self.frame_background:
                continue
            observed_planes.append(np.asarray(self.initial_planes[ch_idx], dtype=np.float32))
            estimated_planes.append(np.asarray(self.estimated_memmaps[ch_idx][idx], dtype=np.float32))
            backgrounds.append(float(self.frame_background[key]))
        if not observed_planes or not estimated_planes:
            return None
        observed = np.mean(np.stack(observed_planes, axis=0), axis=0)
        estimated = np.mean(np.stack(estimated_planes, axis=0), axis=0)
        background = float(np.mean(backgrounds)) if backgrounds else 0.0
        return observed, estimated, background

    def _render_frame(
        self,
        frame_idx: int,
        frame_count: int,
        layout: str,
        reference_rgb: Optional[np.ndarray],
        target_shape: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        current_rgb, metric_channels = self._standard_rgb_for_frame(frame_idx)
        if layout == "split":
            original_rgb = self._standard_rgb_for_frame(-1)[0]
            frame_h, frame_w = current_rgb.shape[:2]
            left_w = max(1, frame_w // 2)
            right_w = max(1, frame_w - left_w)
            rgb = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            if original_rgb.shape != current_rgb.shape:
                original_rgb = _movie_fit_rgb_to_box(original_rgb, frame_w, frame_h)
            rgb[:, :left_w, :] = original_rgb[:, :left_w, :]
            rgb[:, left_w:, :] = current_rgb[:, left_w:, :]
        else:
            rgb = current_rgb

        rgb = _movie_resize_rgb(rgb)
        physical_ratio_mode = "estimated" in self.difference_inset.strip().lower()
        if physical_ratio_mode and frame_idx >= 0:
            ratio_payload = self._raw_estimated_ratio_payload(frame_idx)
            if ratio_payload is not None:
                observed_plane, estimated_plane, background = ratio_payload
                inset_mode = "difference" if "\u2212" in self.difference_inset or "raw -" in self.difference_inset.lower() else "ratio"
                rgb = _movie_raw_estimated_ratio_inset_rgb(
                    rgb,
                    observed_plane,
                    estimated_plane,
                    background,
                    mode=inset_mode,
                )
        if reference_rgb is not None and frame_idx >= 0:
            # For the DL refinement frame with a "vs final" inset, compare against
            # the last RL iteration (frame_idx - 1) instead of the DL result itself
            # (which would produce an all-black difference).
            is_dl_frame = (
                "final" in self.difference_inset.lower()
                and frame_idx > 0
                and any(
                    "dl" in (self.frame_labels.get((ch_idx, frame_idx)) or "").lower()
                    for ch_idx in self.active_channels
                )
            )
            eff_reference = (
                self._standard_rgb_for_frame(frame_idx - 1)[0]
                if is_dl_frame
                else reference_rgb
            )
            rgb = _movie_difference_inset_rgb(
                rgb,
                current_rgb,
                eff_reference,
                self.difference_inset if not is_dl_frame else self.difference_inset.replace("vs final", "DL vs last RL"),
            )
        has_stage_label = frame_idx >= 0 and any(
            self.frame_labels.get((ch_idx, frame_idx))
            for ch_idx in self.active_channels
        )
        if self.title_text or self.show_info_metrics or has_stage_label:
            rgb = _movie_overlay_rgb(
                rgb,
                title=self.title_text,
                bottom_lines=self._bottom_lines_for(frame_idx, frame_count, metric_channels),
                convergence_series=(
                    self._convergence_series_for(frame_idx)
                    if self.show_info_metrics and frame_idx >= 0
                    else None
                ),
                convergence_total_points=frame_count,
                convergence_log_scale=self.convergence_log_scale,
            )
        if target_shape is not None:
            rgb = _movie_letterbox_rgb(rgb, target_shape)
        return rgb

    def encode(self) -> None:
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Movie export requires imageio and imageio-ffmpeg. "
                "Install the GUI requirements again, or run: pip install \"imageio[ffmpeg]\""
            ) from exc

        frame_count = max(self.iter_counts.values(), default=0)
        if frame_count <= 0:
            raise RuntimeError("No movie frames were captured.")
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame_indices = self._sequence_frame_indices(frame_count)
        layouts = self._layout_sequence()
        total_frames = len(frame_indices) * len(layouts)
        reference_rgb = self._difference_reference_rgb(frame_count)
        sample_shapes = [
            self._render_frame(frame_indices[0], frame_count, layout, reference_rgb).shape[:2]
            for layout in layouts
        ]
        target_shape = (
            max(shape[0] for shape in sample_shapes),
            max(shape[1] for shape in sample_shapes),
        )
        self.progress_cb(f"Encoding MP4 movie ({total_frames} frames at {self.fps} fps)…")
        with imageio.get_writer(
            str(output),
            fps=self.fps,
            codec="libx264",
            quality=10,
            ffmpeg_params=["-crf", "16"],
        ) as writer:
            movie_frame_idx = 0
            for layout in layouts:
                for frame_idx in frame_indices:
                    movie_frame_idx += 1
                    rgb = self._render_frame(frame_idx, frame_count, layout, reference_rgb, target_shape)
                    writer.append_data(_movie_pad_even_rgb(rgb))
                    if movie_frame_idx % 25 == 0 or movie_frame_idx == total_frames:
                        self.progress_cb(f"  Movie encoded frame {movie_frame_idx}/{total_frames}")
        self.progress_cb(f"Movie saved: {output}")

    def encode_downsized_gif(self, *, scale: float = 0.5, colors: int = 128) -> Optional[Path]:
        """Convert the recorded MP4 to an optimised animated GIF.

        Parameters
        ----------
        scale:
            Spatial scale factor applied to each frame (default 0.5 = half size).
        colors:
            Palette size for colour quantisation (2–256).  Fewer colours compress
            better; 128 is a good balance for microscopy images.  The palette is
            derived from the first frame and reused for all subsequent frames so
            that LZW can exploit inter-frame similarity.
        """
        try:
            import imageio.v2 as imageio
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "GIF export requires imageio and Pillow. Install the GUI requirements again."
            ) from exc

        output = Path(self.output_path)
        if not output.exists():
            return None
        gif_path = output.with_suffix(".gif")
        self.progress_cb(f"Creating half-size GIF from MP4: {gif_path}")
        reader = imageio.get_reader(str(output))
        meta = reader.get_meta_data() or {}
        fps = float(meta.get("fps") or self.fps or 10)
        # GIF duration is in milliseconds per frame
        duration_ms = int(round(1000.0 / max(fps, 1e-6)))
        frames: list = []
        palette_image: Optional[Image.Image] = None
        frame_idx = 0
        try:
            for frame in reader:
                frame_idx += 1
                rgb = np.asarray(frame[:, :, :3], dtype=np.uint8)
                h, w = rgb.shape[:2]
                new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                resized = Image.fromarray(rgb).resize(new_size, Image.Resampling.LANCZOS)
                if palette_image is None:
                    # Derive palette from first frame; reuse for all frames so
                    # the LZW stream can compress repeated colour indices well.
                    palette_image = resized.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=0)
                    frames.append(palette_image)
                else:
                    frames.append(resized.quantize(palette=palette_image, dither=0))
                if frame_idx % 25 == 0:
                    self.progress_cb(f"  GIF converted frame {frame_idx}")
        finally:
            reader.close()

        if not frames:
            return None

        frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration_ms,
            optimize=True,
        )
        self.progress_cb(f"GIF saved: {gif_path}")
        return gif_path

    def close(self, *, remove_output: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        for mm in self.memmaps.values():
            try:
                mm.flush()
            except Exception:
                pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if remove_output:
            try:
                Path(self.output_path).unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Worker thread for deconvolution
# ---------------------------------------------------------------------------

def _deconvolve_channel_stacks(
    channels_zyx: list[np.ndarray],
    metadata: dict,
    params: dict,
    t_index: int,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[np.ndarray]:
    """Deconvolve one already-loaded timepoint and return per-channel ZYX output."""
    del metadata  # Metadata is carried indirectly via params and normalized shapes.

    def _progress(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    def _stopped() -> bool:
        return bool(should_stop and should_stop())

    try:
        from core.deconvolve_ci import (
            ci_generate_psf,
            ci_rl_deconvolve,
            ci_sparse_hessian_deconvolve,
        )
        from core.deconvolve_ci_dl import deconvolve_ci_rl_dl
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load deconvolve_ci (torch DLL error).\n\n"
            f"Your PyTorch installation appears broken. Try:\n"
            f"  conda install pytorch torchvision torchaudio "
            f"pytorch-cuda=12.1 -c pytorch -c nvidia\n\n"
            f"Original error: {exc}"
        ) from exc

    results: list[np.ndarray] = []
    n_channels = len(channels_zyx)
    movie_recorder: Optional[_IterationMovieRecorder] = None
    movie_params = dict(params.get("movie") or {})
    if movie_params.get("enabled"):
        movie_recorder = _IterationMovieRecorder(
            channels_zyx,
            dict(movie_params.get("render_state") or {}),
            params["niter_list"],
            str(movie_params.get("path") or ""),
            int(movie_params.get("fps", 10)),
            str(params.get("method", "")),
            str(params.get("start", "")),
            str(movie_params.get("title_text") or ""),
            bool(movie_params.get("show_info_metrics", False)),
            bool(movie_params.get("hold_endpoints", False)),
            str(movie_params.get("layout_mode") or "Standard"),
            str(movie_params.get("difference_inset") or "None"),
            bool(movie_params.get("convergence_log_scale", False)),
            _progress,
        )
        _progress(
            f"Movie export enabled: {movie_recorder.output_path} "
            f"({movie_recorder.projection}, {movie_recorder.fps} fps)"
        )
        if movie_params.get("create_gif"):
            _progress("Movie GIF export enabled: half-size animated GIF will be created after MP4")

        # Check whether the image will be tiled — movie is incompatible with tiling.
        # Do this early so the user gets a clear error before any computation starts.
        if channels_zyx:
            from core.deconvolve_ci import _auto_n_tiles
            _sample = _channel_stack_to_solver_input(channels_zyx[0])
            _n_tiles_check = _auto_n_tiles(_sample.shape, device=params.get("device"), psf_xy_est=65)
            if _n_tiles_check > 1:
                raise RuntimeError(
                    f"Movie export is not supported for large images that require tiling "
                    f"(image is {_sample.shape[1]}×{_sample.shape[2]} px and would be split into "
                    f"{_n_tiles_check} tiles). Please uncheck 'Record movie' and try again."
                )

    try:
        for ci, channel_zyx in enumerate(channels_zyx):
            if _stopped():
                raise RuntimeError("Stopped by user")

            ch_data = _channel_stack_to_solver_input(channel_zyx)
            em_list = params["emission_wavelengths"]
            em_wl = em_list[ci] if ci < len(em_list) else em_list[-1] if em_list else 520.0
            ex_list = params["excitation_wavelengths"]
            ex_wl = ex_list[ci] if ci < len(ex_list) else ex_list[-1] if ex_list else None
            if params["microscope_type"] != "confocal":
                ex_wl = None
            pinhole_list = params["pinhole_airy_units"]
            pinhole_airy = (
                pinhole_list[ci] if ci < len(pinhole_list)
                else pinhole_list[-1] if pinhole_list
                else _DEFAULT_PINHOLE_AIRY_UNITS
            )

            use_2d_wf_auto = (
                ch_data.ndim == 2
                and params["microscope_type"] == "widefield"
                and params["method"] in ("ci_rl", "ci_rl_tv", "ci_rl_dl")
                and params["two_d_mode"] == "auto"
            )

            psf_pixel_size_z_nm = params["pixel_size_z_nm"]
            if use_2d_wf_auto:
                n_z_psf = 65
                psf_pixel_size_z_nm = _estimate_two_d_wf_psf_z_nm(
                    em_wl,
                    params["na"],
                    params["ri_sample"],
                    params["pixel_size_xy_nm"],
                )
            elif ch_data.ndim == 3:
                nz_img, _, _ = ch_data.shape
                n_z_psf = max(2 * nz_img - 1, 1) | 1
            else:
                n_z_psf = 1

            airy_radius_nm = 0.61 * em_wl / params["na"]
            airy_radius_px = airy_radius_nm / params["pixel_size_xy_nm"]
            n_xy_psf = int(max(64, 2 * int(4 * airy_radius_px) + 1))
            if n_xy_psf % 2 == 0:
                n_xy_psf += 1

            _progress(
                f"T {t_index + 1} — generating PSF for channel {ci + 1}/{n_channels} (λ={em_wl:.0f} nm)…"
            )
            t_psf = time.time()
            psf = ci_generate_psf(
                na=params["na"],
                wavelength_nm=em_wl,
                pixel_size_xy_nm=params["pixel_size_xy_nm"],
                pixel_size_z_nm=psf_pixel_size_z_nm,
                n_xy=n_xy_psf,
                n_z=n_z_psf,
                ri_immersion=params["ri_immersion"],
                ri_sample=params["ri_sample"],
                ri_coverslip=params["ri_immersion"],
                ri_coverslip_design=params["ri_immersion"],
                ri_immersion_design=params["ri_immersion"],
                t_g=params["t_g"],
                t_g0=params["t_g0"],
                t_i0=params["t_i0"],
                z_p=params["z_p"],
                microscope_type=params["microscope_type"],
                excitation_nm=ex_wl,
                pinhole_airy_units=pinhole_airy,
                integrate_pixels=params["integrate_pixels"],
                n_subpixels=params["n_subpixels"],
                n_pupil=params["n_pupil"],
                device=params["device"],
            )
            _progress(
                f"  Ch{ci}: PSF shape={psf.shape} sum={float(psf.sum()):.6g} "
                f"peak={float(psf.max()):.6g} generated in {_format_duration(time.time() - t_psf)}"
            )

            if ch_data.ndim == 2 and psf.ndim == 3 and not use_2d_wf_auto:
                if psf.shape[0] != 1:
                    raise ValueError(
                        f"Expected a singleton-Z PSF for 2D data, got shape {psf.shape}."
                    )
                psf = psf[0]

            if psf.ndim == 3 and ch_data.ndim == 3 and psf.shape[0] > ch_data.shape[0]:
                start = (psf.shape[0] - ch_data.shape[0]) // 2
                psf = psf[start:start + ch_data.shape[0]]

            gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass

            niter_list = params["niter_list"]
            niter = niter_list[ci] if ci < len(niter_list) else niter_list[-1]
            if _stopped():
                raise RuntimeError("Stopped by user")

            _progress(
                f"T {t_index + 1} — deconvolving channel {ci + 1}/{n_channels} "
                f"(image={ch_data.shape}, psf={psf.shape}, {niter} iter)…"
            )
            t_deconv = time.time()
            _live_cb = params.get("live_callback")

            def _make_iter_cb(
                _movie_rec=movie_recorder, _live=_live_cb
            ):
                if _movie_rec is None and _live is None:
                    return None
                def _cb(payload: dict) -> None:
                    if _movie_rec is not None:
                        _movie_rec.capture(payload)
                    if _live is not None:
                        _live(payload)
                return _cb

            common = dict(
                niter=niter,
                offset=params["offset"],
                prefilter_sigma=params["prefilter_sigma"],
                start=params["start"],
                background=params["background"],
                convergence=params["convergence"],
                rel_threshold=params["rel_threshold"],
                check_every=params["check_every"],
                pixel_size_xy=params["pixel_size_xy_nm"],
                pixel_size_z=psf_pixel_size_z_nm if use_2d_wf_auto else params["pixel_size_z_nm"],
                device=params["device"],
                iteration_callback=_make_iter_cb(),
                channel_index=ci,
            )
            if params["method"] == "ci_sparse_hessian":
                out = ci_sparse_hessian_deconvolve(
                    ch_data,
                    psf,
                    sparse_hessian_weight=params["sparse_hessian_weight"],
                    sparse_hessian_reg=params["sparse_hessian_reg"],
                    **common,
                )
            elif params["method"] == "ci_rl_dl":
                out = deconvolve_ci_rl_dl(
                    ch_data,
                    psf,
                    model_path=params.get("dl_model_path") or None,
                    device=params["device"],
                    rl_kwargs={
                        **common,
                        "tv_lambda": 0.0,
                        "damping": params["damping"],
                        "microscope_type": params["microscope_type"],
                        "two_d_mode": params["two_d_mode"],
                        "two_d_wf_aggressiveness": params["two_d_wf_aggressiveness"],
                        "two_d_wf_bg_radius_um": params["two_d_wf_bg_radius_um"],
                        "two_d_wf_bg_scale": params["two_d_wf_bg_scale"],
                    },
                    dl_kwargs={
                        "z_radius": params["dl_z_context"],
                        "batch_size": params["dl_batch_size"],
                        "mixed_precision": params["dl_mixed_precision"],
                        "residual_strength": params["dl_residual_strength"],
                    },
                    return_diagnostics=True,
                )
                if (movie_recorder is not None or _live_cb is not None) and params.get("dl_model_path"):
                    dl_frame_idx = int(niter) + 1
                    movie_payload: dict[str, Any] = {
                        "channel_index": int(ci),
                        "iteration": dl_frame_idx,
                        "total_iterations": dl_frame_idx,
                        "image": out["result"],
                        "stage_label": "DL refinement",
                        "is_final": True,
                    }
                    if out.get("reconvolved_prediction") is not None:
                        movie_payload["estimated"] = out["reconvolved_prediction"]
                        movie_payload["background"] = 0.0
                    if movie_recorder is not None:
                        movie_recorder.capture(movie_payload)
                    if _live_cb is not None:
                        _live_cb(movie_payload)
            else:
                out = ci_rl_deconvolve(
                    ch_data,
                    psf,
                    tv_lambda=params["tv_lambda"],
                    damping=params["damping"],
                    microscope_type=params["microscope_type"],
                    two_d_mode=params["two_d_mode"],
                    two_d_wf_aggressiveness=params["two_d_wf_aggressiveness"],
                    two_d_wf_bg_radius_um=params["two_d_wf_bg_radius_um"],
                    two_d_wf_bg_scale=params["two_d_wf_bg_scale"],
                    **common,
                )
            iterations_used = out.get("iterations_used", niter)
            convergence_history = out.get("convergence") or []
            conv_text = f", last objective={convergence_history[-1]:.6g}" if convergence_history else ""
            _progress(
                f"  Ch{ci}: done in {_format_duration(time.time() - t_deconv)} "
                f"(iterations used={iterations_used}{conv_text})"
            )
            results.append(_solver_output_to_zyx(out["result"].copy()))

            del psf
            del out
            gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass

        if movie_recorder is not None:
            if _stopped():
                raise RuntimeError("Stopped by user")
            movie_recorder.encode()
            if movie_params.get("create_gif"):
                movie_recorder.encode_downsized_gif(scale=0.5)
    except Exception:
        if movie_recorder is not None:
            movie_recorder.close(remove_output=True)
            movie_recorder = None
        raise
    finally:
        if movie_recorder is not None:
            movie_recorder.close()

    return results


def _write_ome_tiff(data: np.ndarray, path: str, metadata: dict) -> None:
    from bioio.writers import OmeTiffWriter

    physical_pixel_sizes = None
    px_x = metadata.get("pixel_size_x")
    px_z = metadata.get("pixel_size_z")
    if px_x or px_z:
        from bioio_base.types import PhysicalPixelSizes
        physical_pixel_sizes = PhysicalPixelSizes(
            Z=px_z or 1.0, Y=px_x or 1.0, X=px_x or 1.0
        )

    channel_names = _current_channel_names(metadata, data.shape[1])
    OmeTiffWriter.save(
        data.astype(np.float32),
        path,
        dim_order="TCZYX",
        physical_pixel_sizes=physical_pixel_sizes,
        channel_names=channel_names,
    )


class _DeconvolveWorker(QThread):
    """Preview deconvolution for a single timepoint."""

    finished = pyqtSignal(object)
    progress = pyqtSignal(str)
    liveUpdate = pyqtSignal(dict)

    def __init__(
        self,
        channels: list[np.ndarray],
        metadata: dict,
        params: dict,
        t_index: int,
        parent=None,
    ):
        super().__init__(parent)
        self.channels = channels
        self.metadata = metadata
        self.params = params
        self.t_index = int(t_index)

    def run(self):
        monitor = None
        try:
            self.progress.emit("")
            self.progress.emit("Deconvolution")
            self.progress.emit(f"  Timepoint   : {self.t_index + 1}")
            self.progress.emit(f"  Method      : {self.params['method']}")
            self.progress.emit(f"  Iterations  : {', '.join(str(n) for n in self.params['niter_list'])}")
            self.progress.emit(f"  Device      : {self.params['device'] or 'auto'}")
            self.progress.emit(f"  Background  : {self.params['background']}")
            self.progress.emit(f"  Offset      : {self.params['offset']}")
            self.progress.emit(f"  Start       : {self.params['start']}")
            self.progress.emit(
                f"  Convergence : {self.params['convergence']} "
                f"(threshold={self.params['rel_threshold']}, every={self.params['check_every']})"
            )
            if self.params["method"] == "ci_rl_tv":
                self.progress.emit(f"  TV lambda   : {self.params['tv_lambda']}")
            if self.params["method"] in ("ci_rl", "ci_rl_tv", "ci_rl_dl"):
                self.progress.emit(f"  Damping     : {self.params['damping']}")
                if self.params["microscope_type"] == "confocal":
                    self.progress.emit(
                        f"  Pinhole     : {_format_float_list(self.params['pinhole_airy_units'])} AU"
                    )
                self.progress.emit(
                    f"  2D WF mode  : {self.params['two_d_mode']} "
                    f"(aggr={self.params['two_d_wf_aggressiveness']}, "
                    f"bg radius={self.params['two_d_wf_bg_radius_um']} um, "
                    f"bg scale={self.params['two_d_wf_bg_scale']})"
                )
            if self.params["method"] == "ci_rl_dl":
                self.progress.emit(
                    f"  DL model    : {self.params.get('dl_model_path') or '(none; ci_rl unchanged)'}"
                )
                self.progress.emit(
                    f"  DL params   : z-context={self.params['dl_z_context']}, "
                    f"batch={self.params['dl_batch_size']}, "
                    f"mixed_precision={self.params['dl_mixed_precision']}, "
                    f"residual_strength={self.params['dl_residual_strength']}"
                )
            if self.params["method"] == "ci_sparse_hessian":
                self.progress.emit(
                    f"  Sparse      : weight={self.params['sparse_hessian_weight']}, "
                    f"reg={self.params['sparse_hessian_reg']}"
                )
            if self.params["prefilter_sigma"] > 0.0:
                self.progress.emit(f"  Prefilter   : sigma={self.params['prefilter_sigma']}")
            self.progress.emit(f"  Image metrics: {'enabled' if self.params.get('compute_metrics') else 'disabled'}")
            movie_params = self.params.get("movie") or {}
            if movie_params.get("enabled"):
                self.progress.emit(
                    f"  Movie       : {movie_params.get('path')} at {movie_params.get('fps', 10)} fps"
                    f"{' + GIF' if movie_params.get('create_gif') else ''}"
                )

            # Expose live iteration payloads to the main thread via signal.
            # The signal connection (queued by default across threads) ensures
            # the chart update runs on the main thread without any extra locking.
            self.params["live_callback"] = self.liveUpdate.emit
            monitor = _RunMetricsMonitor()
            monitor.start()
            results = _deconvolve_channel_stacks(
                self.channels,
                self.metadata,
                self.params,
                self.t_index,
                progress_cb=self.progress.emit,
                should_stop=self.isInterruptionRequested,
            )
            metrics = monitor.stop()
            for line in _resource_metric_lines(metrics):
                self.progress.emit(line)
            if self.params.get("compute_metrics"):
                t_metrics = time.time()
                self.progress.emit("")
                self.progress.emit("Computing image metrics...")
                for line in _quality_comparison_lines(self.channels, results):
                    self.progress.emit(line)
                self.progress.emit(f"Image metrics computed in {_format_duration(time.time() - t_metrics)}")
            else:
                self.progress.emit("")
                self.progress.emit("Image metrics skipped (disabled).")
            self.finished.emit({
                "timepoint": self.t_index,
                "channels": results,
                "metrics": metrics,
            })
        except Exception as exc:
            if monitor is not None:
                try:
                    metrics = monitor.stop()
                    for line in _resource_metric_lines(metrics):
                        self.progress.emit(line)
                except Exception:
                    pass
            traceback.print_exc()
            self.finished.emit(exc)
        finally:
            _release_cuda_memory(synchronize=True)


class _FitPsfWorker(QThread):
    """Search for best PSF parameters via short RL trials."""

    # (trial_idx, total_trials, trial_params, i_div)
    progress = pyqtSignal(int, int, object, float)
    finished = pyqtSignal(object)  # dict result or Exception

    def __init__(
        self,
        channels: list[np.ndarray],
        base_params: dict,
        niter_fit: int = 40,
        parent=None,
    ):
        super().__init__(parent)
        self.channels = channels
        self.base_params = dict(base_params)
        self.niter_fit = int(niter_fit)

    def run(self):
        try:
            from core.deconvolve_ci import ci_fit_psf_params

            n_channels = len(self.channels)
            if n_channels == 0:
                raise ValueError("No channels provided for PSF fitting.")

            em_wl_list = self.base_params.get("emission_wavelengths", [520])
            if not isinstance(em_wl_list, (list, tuple)):
                em_wl_list = [em_wl_list]
            ex_wl_list = self.base_params.get("excitation_wavelengths", [488])
            if not isinstance(ex_wl_list, (list, tuple)):
                ex_wl_list = [ex_wl_list]
            pinhole_list = self.base_params.get("pinhole_airy_units", [1.0])
            if not isinstance(pinhole_list, (list, tuple)):
                pinhole_list = [pinhole_list]
            coarse_ri_grid = [1.330, 1.358, 1.385, 1.410, 1.435, 1.460, 1.480, 1.500, 1.515]
            fine_grid_count = 9
            fine_half_width = 0.015
            total_trials = n_channels * (len(coarse_ri_grid) + fine_grid_count)
            score_label = "normalized residual + normalized roughness (coarse-to-fine RI scan)"

            def _merge_logs(logs_per_channel: list[list[dict]]) -> list[dict]:
                merged_by_key: dict[tuple[float, float], dict] = {}
                for log_entries in logs_per_channel:
                    for entry in log_entries:
                        params = entry.get("params", {})
                        ph = float(params.get("pinhole_airy_units", 1.0))
                        ri = float(params.get("ri_sample", 1.33))
                        key = (round(ph, 6), round(ri, 6))
                        bucket = merged_by_key.setdefault(
                            key,
                            {
                                "params": {
                                    "pinhole_airy_units": ph,
                                    "ri_sample": ri,
                                },
                                "residual": [],
                                "roughness": [],
                                "score": [],
                                "i_div": [],
                            },
                        )
                        for metric_key in ("residual", "roughness", "score", "i_div"):
                            if metric_key in entry:
                                bucket[metric_key].append(float(entry[metric_key]))

                merged_log = []
                for key in sorted(merged_by_key.keys(), key=lambda item: (item[0], item[1])):
                    bucket = merged_by_key[key]
                    merged_entry = {"params": bucket["params"]}
                    for metric_key in ("residual", "roughness", "score", "i_div"):
                        values = bucket.get(metric_key, [])
                        if values:
                            merged_entry[metric_key] = float(np.mean(values))
                    merged_log.append(merged_entry)
                return merged_log

            def _run_stage(
                ri_grid: list[float],
                *,
                trial_offset: int,
                emit_progress: bool = True,
            ) -> tuple[list[list[dict]], list[float], dict, str]:
                stage_logs: list[list[dict]] = []
                stage_baselines: list[float] = []
                stage_grid_info: dict = {}
                stage_score_label = score_label
                stage_total = len(ri_grid)

                for ci, ch_data in enumerate(self.channels):
                    if self.isInterruptionRequested():
                        raise RuntimeError("Stopped by user")
                    em_wl = float(em_wl_list[ci] if ci < len(em_wl_list) else em_wl_list[-1])
                    ex_wl = float(ex_wl_list[ci] if ci < len(ex_wl_list) else ex_wl_list[-1])
                    ph = float(pinhole_list[ci] if ci < len(pinhole_list) else pinhole_list[-1])

                    ch_base = dict(self.base_params)
                    ch_base["emission_nm"] = em_wl
                    ch_base["excitation_nm"] = ex_wl
                    ch_base["pinhole_airy_units"] = ph

                    def _cb(idx: int, total: int, params: dict, i_div: float,
                            ci_=ci, offset_=trial_offset, stage_len=stage_total):
                        global_idx = offset_ + ci_ * stage_len + idx
                        self.progress.emit(global_idx, total_trials, params, i_div)

                    result = ci_fit_psf_params(
                        ch_data,
                        ch_base,
                        pinhole_grid=[ph],
                        ri_sample_grid=list(ri_grid),
                        niter_fit=self.niter_fit,
                        callback=_cb if emit_progress else None,
                        should_stop=self.isInterruptionRequested,
                    )
                    stage_logs.append(result["search_log"])
                    stage_baselines.append(float(result.get("baseline_i_div", float("inf"))))
                    stage_grid_info = result["grid"]
                    stage_score_label = result.get("score_label", stage_score_label)

                return stage_logs, stage_baselines, stage_grid_info, stage_score_label

            coarse_logs, _, coarse_grid_info, coarse_score_label = _run_stage(
                coarse_ri_grid,
                trial_offset=0,
            )
            coarse_merged = _merge_logs(coarse_logs)
            if not coarse_merged:
                raise RuntimeError("No coarse RI search results were produced.")

            coarse_best = min(coarse_merged, key=lambda e: e.get("i_div", float("inf")))
            coarse_best_ri = float(coarse_best["params"].get("ri_sample", 1.47))
            fine_lo = max(1.330, coarse_best_ri - fine_half_width)
            fine_hi = min(1.515, coarse_best_ri + fine_half_width)
            fine_ri_grid = [round(float(v), 4) for v in np.linspace(fine_lo, fine_hi, fine_grid_count)]

            fine_logs, _, fine_grid_info, fine_score_label = _run_stage(
                fine_ri_grid,
                trial_offset=n_channels * len(coarse_ri_grid),
            )
            fine_merged = _merge_logs(fine_logs)
            merged_log = _merge_logs([coarse_merged, fine_merged])
            base_score_label = fine_score_label or coarse_score_label or "normalized residual + normalized roughness"
            score_label = f"{base_score_label} (coarse-to-fine RI scan)"

            orig_ri = float(self.base_params.get("ri_sample", 1.47))
            baseline_logs, _, _, _ = _run_stage(
                [orig_ri],
                trial_offset=0,
                emit_progress=False,
            )
            baseline_merged = _merge_logs(baseline_logs)
            baseline_entry = baseline_merged[0] if baseline_merged else None

            def _normalize_metric(values: list[float]) -> list[float]:
                arr = np.asarray([float(v) for v in values], dtype=np.float64)
                finite = np.isfinite(arr)
                if not np.any(finite):
                    return [float("inf")] * len(values)
                vmin = float(np.min(arr[finite]))
                vmax = float(np.max(arr[finite]))
                if vmax <= vmin + 1e-12:
                    return [0.0 if is_finite else float("inf") for is_finite in finite]
                norm = (arr - vmin) / (vmax - vmin)
                return [float(v) if is_finite else float("inf") for v, is_finite in zip(norm, finite)]

            scored_entries = [dict(entry) for entry in merged_log]
            if baseline_entry is not None:
                scored_entries.append(dict(baseline_entry))

            residual_norm = _normalize_metric([float(entry.get("residual", float("inf"))) for entry in scored_entries])
            roughness_norm = _normalize_metric([float(entry.get("roughness", float("inf"))) for entry in scored_entries])
            for entry, resid_n, rough_n in zip(scored_entries, residual_norm, roughness_norm):
                score = resid_n + rough_n if np.isfinite(resid_n) and np.isfinite(rough_n) else float("inf")
                entry["score"] = score
                entry["i_div"] = score

            if baseline_entry is not None:
                baseline_scored = scored_entries[-1]
                merged_log = scored_entries[:-1]
            else:
                baseline_scored = None
                merged_log = scored_entries

            ri_values = sorted({float(entry["params"].get("ri_sample", 1.33)) for entry in merged_log})
            pinhole_values = sorted({float(entry["params"].get("pinhole_airy_units", 1.0)) for entry in merged_log})
            grid_info = {
                "pinhole": pinhole_values or coarse_grid_info.get("pinhole", fine_grid_info.get("pinhole", [])),
                "ri_sample": ri_values or coarse_grid_info.get("ri_sample", fine_grid_info.get("ri_sample", [])),
            }

            # Find best from merged candidates and compare to the original parameters.
            candidate_best = min(merged_log, key=lambda e: e["i_div"])
            if baseline_scored is not None and baseline_scored["i_div"] <= candidate_best["i_div"]:
                best = baseline_scored
            else:
                best = candidate_best

            baseline_i_div = float(baseline_scored["i_div"]) if baseline_scored is not None else float("inf")
            improvement_pct = (
                max(0.0, (baseline_i_div - best["i_div"]) / baseline_i_div * 100.0)
                if baseline_i_div > 0 and baseline_i_div != float("inf")
                else 0.0
            )

            self.finished.emit({
                "best_params": best["params"],
                "best_i_div": best["i_div"],
                "grid_best_params": candidate_best["params"],
                "grid_best_i_div": candidate_best["i_div"],
                "baseline_i_div": baseline_i_div,
                "improvement_pct": improvement_pct,
                "search_log": merged_log,
                "grid": grid_info,
                "score_label": score_label,
            })
        except Exception as exc:
            if "Stopped by user" not in str(exc):
                traceback.print_exc()
            self.finished.emit(exc)
        finally:
            _release_cuda_memory(synchronize=True)


def _guess_fit_zp_um(channels, pixel_size_z_nm):
    """Guess emitter depth from the active stack depth for RI fitting."""
    px_z_nm = float(pixel_size_z_nm)
    if px_z_nm <= 0.0:
        return None
    z_sizes = [int(ch.shape[0]) for ch in channels if getattr(ch, "ndim", 0) == 3 and ch.shape[0] > 1]
    if not z_sizes:
        return None
    nz = min(z_sizes)
    stack_depth_um = max(nz - 1, 0) * px_z_nm / 1000.0
    guess_um = max(stack_depth_um * 0.5, px_z_nm / 1000.0)
    return guess_um, nz, stack_depth_um


class _SaveTSeriesWorker(QThread):
    """Export a full deconvolved T-series to OME-TIFF."""

    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(
        self,
        source: _BaseTimepointSource,
        metadata: dict,
        params: dict,
        output_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.metadata = metadata
        self.params = params
        self.output_path = output_path

    def run(self):
        stage_path = None
        try:
            size_t = max(int(self.metadata.get("size_t", 1)), 1)
            size_c = max(int(self.metadata.get("size_c", 1)), 1)
            size_z = max(int(self.metadata.get("size_z", 1)), 1)
            size_y = max(int(self.metadata.get("size_y", 1)), 1)
            size_x = max(int(self.metadata.get("size_x", 1)), 1)

            fd, stage_path = tempfile.mkstemp(prefix="cideconvolve_", suffix=".dat")
            os.close(fd)
            staged = np.memmap(
                stage_path,
                dtype=np.float32,
                mode="w+",
                shape=(size_t, size_c, size_z, size_y, size_x),
            )

            for t_index in range(size_t):
                if self.isInterruptionRequested():
                    raise RuntimeError("Stopped by user")
                self.progress.emit(f"Saving T-series — loading timepoint {t_index + 1}/{size_t}…")
                channels = self.source.load_timepoint(
                    t_index,
                    progress_cb=lambda done, total, text: self.progress.emit(
                        f"{text} ({done}/{total})"
                    ),
                )
                self.progress.emit(f"Saving T-series — processing timepoint {t_index + 1}/{size_t}…")
                results = _deconvolve_channel_stacks(
                    channels,
                    self.metadata,
                    self.params,
                    t_index,
                    progress_cb=self.progress.emit,
                    should_stop=self.isInterruptionRequested,
                )
                for c_index, result in enumerate(results):
                    staged[t_index, c_index, :, :, :] = result
                staged.flush()

            self.progress.emit("Writing OME-TIFF…")
            _write_ome_tiff(staged, self.output_path, self.metadata)
            self.finished.emit({"path": self.output_path})
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit(exc)
        finally:
            if stage_path and os.path.exists(stage_path):
                try:
                    os.remove(stage_path)
                except OSError:
                    pass


class _StreamingOmeroWorker(QThread):
    """Stream a pyramidal OMERO image directly to OME-Zarr."""

    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(
        self,
        source: _OmeroTimepointSource,
        metadata: dict,
        params: dict,
        output_path: str,
        *,
        tile_size: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.metadata = dict(metadata)
        self.params = dict(params)
        self.output_path = output_path
        self.tile_size = int(tile_size)

    def run(self):
        monitor = None
        try:
            if self.params.get("method") == "ci_rl_dl":
                raise RuntimeError("Streaming ci_rl_dl is not enabled yet. Choose ci_rl, ci_rl_tv, or ci_sparse_hessian.")

            from core.streaming import (
                ZarrPyramidSink,
                deconvolve_streaming,
                save_streaming_provenance,
                suggest_streaming_tile_size,
            )
            from core.deconvolve_ci import (
                ci_generate_psf,
                ci_rl_deconvolve,
                ci_sparse_hessian_deconvolve,
            )

            region_source = _OmeroPyramidRegionSource(self.source)
            region_source.metadata.update(self.metadata)
            region_source.metadata["size_t"] = region_source.shape[0]
            region_source.metadata["size_c"] = region_source.shape[1]
            region_source.metadata["size_z"] = region_source.shape[2]
            region_source.metadata["size_y"] = region_source.shape[3]
            region_source.metadata["size_x"] = region_source.shape[4]
            if self.tile_size <= 0:
                self.tile_size = suggest_streaming_tile_size(
                    region_source.shape,
                    psf_xy_est=_estimate_psf_xy_for_params(self.params),
                    method=str(self.params.get("method", "ci_rl")),
                    device=self.params.get("device"),
                )

            sink = ZarrPyramidSink(
                self.output_path,
                shape=region_source.shape,
                metadata=region_source.metadata,
                resume=True,
            )
            psf_cache: dict[int, np.ndarray] = {}
            psf_pixel_size_z_cache: dict[int, float] = {}
            self.progress.emit("")
            self.progress.emit("Streaming OMERO deconvolution")
            self.progress.emit(
                f"  Shape       : T={region_source.shape[0]} C={region_source.shape[1]} "
                f"Z={region_source.shape[2]} Y={region_source.shape[3]} X={region_source.shape[4]}"
            )
            self.progress.emit(f"  Method      : {self.params['method']}")
            self.progress.emit(f"  Iterations  : {', '.join(str(n) for n in self.params['niter_list'])}")
            self.progress.emit(f"  Tile size   : {self.tile_size} x {self.tile_size} px")
            self.progress.emit(f"  Output      : {self.output_path}")

            monitor = _RunMetricsMonitor()
            monitor.start()

            def _stopped() -> bool:
                return bool(self.isInterruptionRequested())

            def _psf_for_channel(ci: int) -> np.ndarray:
                cached = psf_cache.get(ci)
                if cached is not None:
                    return cached
                if _stopped():
                    raise RuntimeError("Stopped by user")
                em_list = self.params["emission_wavelengths"]
                em_wl = em_list[ci] if ci < len(em_list) else em_list[-1] if em_list else 520.0
                ex_list = self.params["excitation_wavelengths"]
                ex_wl = ex_list[ci] if ci < len(ex_list) else ex_list[-1] if ex_list else None
                if self.params["microscope_type"] != "confocal":
                    ex_wl = None
                pinhole_list = self.params["pinhole_airy_units"]
                pinhole_airy = (
                    pinhole_list[ci] if ci < len(pinhole_list)
                    else pinhole_list[-1] if pinhole_list
                    else _DEFAULT_PINHOLE_AIRY_UNITS
                )
                is_2d = region_source.shape[2] == 1
                use_2d_wf_auto = (
                    is_2d
                    and self.params["microscope_type"] == "widefield"
                    and self.params["method"] in ("ci_rl", "ci_rl_tv")
                    and self.params["two_d_mode"] == "auto"
                )
                psf_pixel_size_z_nm = self.params["pixel_size_z_nm"]
                if use_2d_wf_auto:
                    n_z_psf = 65
                    psf_pixel_size_z_nm = _estimate_two_d_wf_psf_z_nm(
                        em_wl,
                        self.params["na"],
                        self.params["ri_sample"],
                        self.params["pixel_size_xy_nm"],
                    )
                elif not is_2d:
                    n_z_psf = max(2 * region_source.shape[2] - 1, 1) | 1
                else:
                    n_z_psf = 1

                airy_radius_nm = 0.61 * em_wl / self.params["na"]
                airy_radius_px = airy_radius_nm / self.params["pixel_size_xy_nm"]
                n_xy_psf = int(max(64, 2 * int(4 * airy_radius_px) + 1))
                if n_xy_psf % 2 == 0:
                    n_xy_psf += 1
                self.progress.emit(
                    f"  Generating PSF for channel {ci + 1}/{region_source.shape[1]} "
                    f"(lambda={em_wl:.0f} nm)..."
                )
                psf = ci_generate_psf(
                    na=self.params["na"],
                    wavelength_nm=em_wl,
                    pixel_size_xy_nm=self.params["pixel_size_xy_nm"],
                    pixel_size_z_nm=psf_pixel_size_z_nm,
                    n_xy=n_xy_psf,
                    n_z=n_z_psf,
                    ri_immersion=self.params["ri_immersion"],
                    ri_sample=self.params["ri_sample"],
                    ri_coverslip=self.params["ri_immersion"],
                    ri_coverslip_design=self.params["ri_immersion"],
                    ri_immersion_design=self.params["ri_immersion"],
                    t_g=self.params["t_g"],
                    t_g0=self.params["t_g0"],
                    t_i0=self.params["t_i0"],
                    z_p=self.params["z_p"],
                    microscope_type=self.params["microscope_type"],
                    excitation_nm=ex_wl,
                    pinhole_airy_units=pinhole_airy,
                    integrate_pixels=self.params["integrate_pixels"],
                    n_subpixels=self.params["n_subpixels"],
                    n_pupil=self.params["n_pupil"],
                    device=self.params["device"],
                )
                self.progress.emit(
                    f"    Ch{ci}: PSF shape={psf.shape} sum={float(psf.sum()):.6g}"
                )
                psf_cache[ci] = psf
                psf_pixel_size_z_cache[ci] = float(psf_pixel_size_z_nm)
                return psf

            def _deconvolve_tile(tile_img: np.ndarray, psf: np.ndarray, ci: int) -> np.ndarray:
                if _stopped():
                    raise RuntimeError("Stopped by user")
                effective_psf = psf
                use_2d_wf_auto = (
                    tile_img.ndim == 2
                    and self.params["microscope_type"] == "widefield"
                    and self.params["method"] in ("ci_rl", "ci_rl_tv")
                    and self.params["two_d_mode"] == "auto"
                )
                if tile_img.ndim == 2 and effective_psf.ndim == 3 and not use_2d_wf_auto:
                    if effective_psf.shape[0] != 1:
                        raise ValueError(f"Expected singleton-Z PSF for 2D tile, got {effective_psf.shape}")
                    effective_psf = effective_psf[0]
                elif tile_img.ndim == 3 and effective_psf.ndim == 2:
                    effective_psf = effective_psf[np.newaxis, :, :]
                if tile_img.ndim == 3 and effective_psf.ndim == 3 and effective_psf.shape[0] > tile_img.shape[0]:
                    start = (effective_psf.shape[0] - tile_img.shape[0]) // 2
                    effective_psf = effective_psf[start:start + tile_img.shape[0]]

                niter_list = self.params["niter_list"]
                niter = niter_list[ci] if ci < len(niter_list) else niter_list[-1]
                common = dict(
                    niter=niter,
                    offset=self.params["offset"],
                    prefilter_sigma=self.params["prefilter_sigma"],
                    start=self.params["start"],
                    background=self.params["background"],
                    convergence=self.params["convergence"],
                    rel_threshold=self.params["rel_threshold"],
                    check_every=self.params["check_every"],
                    pixel_size_xy=self.params["pixel_size_xy_nm"],
                    pixel_size_z=(
                        psf_pixel_size_z_cache.get(ci, self.params["pixel_size_z_nm"])
                        if use_2d_wf_auto
                        else self.params["pixel_size_z_nm"]
                    ),
                    device=self.params["device"],
                    tiling="none",
                    iteration_callback=None,
                    channel_index=ci,
                )
                if self.params["method"] == "ci_sparse_hessian":
                    out = ci_sparse_hessian_deconvolve(
                        tile_img,
                        effective_psf,
                        sparse_hessian_weight=self.params["sparse_hessian_weight"],
                        sparse_hessian_reg=self.params["sparse_hessian_reg"],
                        **common,
                    )
                else:
                    out = ci_rl_deconvolve(
                        tile_img,
                        effective_psf,
                        tv_lambda=self.params["tv_lambda"] if self.params["method"] == "ci_rl_tv" else 0.0,
                        damping=self.params["damping"],
                        microscope_type=self.params["microscope_type"],
                        two_d_mode=self.params["two_d_mode"],
                        two_d_wf_aggressiveness=self.params["two_d_wf_aggressiveness"],
                        two_d_wf_bg_radius_um=self.params["two_d_wf_bg_radius_um"],
                        two_d_wf_bg_scale=self.params["two_d_wf_bg_scale"],
                        **common,
                    )
                if _stopped():
                    raise RuntimeError("Stopped by user")
                result = out["result"]
                del out
                _release_cuda_memory()
                return result

            def _progress(payload: dict) -> None:
                if _stopped():
                    raise RuntimeError("Stopped by user")
                event = payload.get("event")
                if event == "tile_start":
                    self.progress.emit(
                        f"  Tile {int(payload.get('done', 0)) + 1}/{int(payload.get('total', 0))}: "
                        f"T={int(payload.get('timepoint', 0)) + 1} "
                        f"C={int(payload.get('channel', 0)) + 1} "
                        f"{payload.get('core')}"
                    )
                elif event == "tile_done":
                    self.progress.emit(
                        f"    Done {int(payload.get('done', 0))}/{int(payload.get('total', 0))} tiles"
                    )
                elif event == "pyramid_start":
                    self.progress.emit("  Building OME-Zarr pyramid levels...")

            summary = deconvolve_streaming(
                region_source,
                sink,
                psf_for_channel=_psf_for_channel,
                deconvolve_tile=_deconvolve_tile,
                tile_yx=(self.tile_size, self.tile_size),
                progress=_progress,
                resume=True,
                build_pyramids=True,
            )
            provenance = save_streaming_provenance(
                str(Path(self.output_path).with_suffix(Path(self.output_path).suffix + ".provenance.json")),
                source=region_source,
                sink=sink,
                params={
                    "method": self.params["method"],
                    "iterations": self.params["niter_list"],
                    "tile_size": self.tile_size,
                    "convergence": self.params["convergence"],
                    "rel_threshold": self.params["rel_threshold"],
                    "background": self.params["background"],
                    "offset": self.params["offset"],
                    "prefilter_sigma": self.params["prefilter_sigma"],
                    "start": self.params["start"],
                },
                summary=summary,
            )
            metrics = monitor.stop() if monitor is not None else {}
            for line in _resource_metric_lines(metrics):
                self.progress.emit(line)
            self.finished.emit({
                "streaming_output": self.output_path,
                "provenance": str(provenance),
                "summary": summary,
                "metrics": metrics,
            })
        except Exception as exc:
            if monitor is not None:
                try:
                    metrics = monitor.stop()
                    for line in _resource_metric_lines(metrics):
                        self.progress.emit(line)
                except Exception:
                    pass
            if "Stopped by user" not in str(exc):
                traceback.print_exc()
            self.finished.emit(exc)


def _streaming_output_metadata(base_metadata: dict, params: dict, *, source_name: str = "") -> dict:
    meta = dict(base_metadata or {})
    if source_name:
        meta.setdefault("name", source_name)
    meta["pixel_size_x"] = float(params.get("pixel_size_xy_nm", 1000.0)) / 1000.0
    meta["pixel_size_y"] = float(params.get("pixel_size_xy_nm", 1000.0)) / 1000.0
    meta["pixel_size_z"] = float(params.get("pixel_size_z_nm", 1000.0)) / 1000.0
    meta["na"] = params.get("na")
    meta["refractive_index"] = params.get("ri_immersion")
    meta["sample_refractive_index"] = params.get("ri_sample")
    meta["microscope_type"] = params.get("microscope_type")
    raw_channels = meta.get("channels", [])
    if isinstance(raw_channels, dict):
        raw_channels = [raw_channels]
    elif not isinstance(raw_channels, Sequence) or isinstance(raw_channels, (str, bytes, bytearray)):
        raw_channels = []
    channels = [dict(ch) if isinstance(ch, dict) else {} for ch in raw_channels]
    size_c = max(int(meta.get("size_c", len(channels) or 1)), 1)
    if len(channels) < size_c:
        channels.extend({} for _ in range(size_c - len(channels)))
    raw_names = meta.get("channel_names") or []
    names = list(raw_names) if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes, bytearray)) else []

    def _param_value(key: str, ci: int):
        values = list(params.get(key) or [])
        if not values:
            return None
        return values[ci] if ci < len(values) else values[-1]

    for ci in range(size_c):
        ch = channels[ci]
        ch.setdefault("name", names[ci] if ci < len(names) else _default_channel_name(ci))
        emission = _param_value("emission_wavelengths", ci)
        excitation = _param_value("excitation_wavelengths", ci)
        pinhole = _param_value("pinhole_airy_units", ci)
        if emission is not None:
            ch["emission_wavelength"] = emission
        if excitation is not None:
            ch["excitation_wavelength"] = excitation
        if pinhole is not None:
            ch["pinhole_airy_units"] = pinhole
    meta["channels"] = channels[:size_c]
    meta["channel_names"] = [
        str(ch.get("name") or ch.get("label") or _default_channel_name(i))
        for i, ch in enumerate(meta["channels"])
    ]
    meta["cideconvolve_processing"] = {
        "method": params.get("method"),
        "iterations": list(params.get("niter_list", [])),
        "convergence": params.get("convergence"),
        "background": params.get("background"),
        "offset": params.get("offset"),
        "prefilter_sigma": params.get("prefilter_sigma"),
        "tile_streaming": True,
    }
    return meta


def _validate_channel_parameter_lists(params: dict, size_c: int) -> None:
    # The interactive GUI already accepts short channel lists by reusing the
    # final value for extra channels.  Batch mode follows that behaviour so one
    # saved 3-channel settings file can still process a mixed 3/4-channel list.
    return None


def _estimate_psf_xy_for_params(params: dict) -> int:
    na = max(float(params.get("na", 1.4) or 1.4), 1e-6)
    pixel_size_xy_nm = max(float(params.get("pixel_size_xy_nm", 65.0) or 65.0), 1e-6)
    emissions = list(params.get("emission_wavelengths") or [520.0])
    max_xy = 65
    for value in emissions:
        try:
            em_wl = float(value)
        except (TypeError, ValueError):
            em_wl = 520.0
        airy_radius_nm = 0.61 * em_wl / na
        airy_radius_px = airy_radius_nm / pixel_size_xy_nm
        n_xy_psf = int(max(64, 2 * int(4 * airy_radius_px) + 1))
        if n_xy_psf % 2 == 0:
            n_xy_psf += 1
        max_xy = max(max_xy, n_xy_psf)
    return int(max_xy)


def _batch_metadata_overlay(metadata: dict[str, Any]) -> dict[str, Any]:
    overlay = dict(metadata or {})
    raw_channels = overlay.get("channels")
    if raw_channels is not None and (
        not isinstance(raw_channels, Sequence)
        or isinstance(raw_channels, (str, bytes, bytearray))
    ):
        overlay.pop("channels", None)
    raw_names = overlay.get("channel_names")
    if raw_names is not None and (
        not isinstance(raw_names, Sequence)
        or isinstance(raw_names, (str, bytes, bytearray))
    ):
        overlay.pop("channel_names", None)
    return overlay


def _run_streaming_deconvolution_job(
    region_source,
    *,
    params: dict,
    output_path: str,
    output_format: str,
    tile_size: int,
    projection_mode: str = "Full stack",
    progress: Optional[Callable[[dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    if params.get("method") == "ci_rl_dl":
        raise RuntimeError("Batch streaming does not support ci_rl_dl yet. Choose ci_rl, ci_rl_tv, or ci_sparse_hessian.")

    from core.streaming import (
        ProjectionPyramidSink,
        TiledOmeTiffSink,
        ZarrPyramidSink,
        deconvolve_streaming,
        save_streaming_provenance,
        suggest_streaming_tile_size,
    )
    from core.deconvolve_ci import (
        ci_generate_psf,
        ci_rl_deconvolve,
        ci_sparse_hessian_deconvolve,
    )

    shape = tuple(int(v) for v in region_source.shape)
    _validate_channel_parameter_lists(params, shape[1])
    region_source.metadata = _streaming_output_metadata(region_source.metadata, params)
    for axis, value in zip(("size_t", "size_c", "size_z", "size_y", "size_x"), shape):
        region_source.metadata[axis] = int(value)

    output_format_key = output_format.strip().lower()
    projection_key = str(projection_mode or "Full stack").strip().lower()
    project_output = shape[2] > 1 and projection_key not in {"", "full stack", "full", "none"}
    sink_shape = (shape[0], shape[1], 1 if project_output else shape[2], shape[3], shape[4])
    sink_metadata = dict(region_source.metadata)
    if project_output:
        sink_metadata["size_z"] = 1
        sink_metadata["default_z"] = 0
        sink_metadata["projection"] = projection_key
        sink_metadata["cideconvolve_processing"] = dict(sink_metadata.get("cideconvolve_processing") or {})
        sink_metadata["cideconvolve_processing"]["projection"] = projection_key

    psf_xy_est = _estimate_psf_xy_for_params(params)
    effective_tile_size = int(tile_size)
    if effective_tile_size <= 0:
        effective_tile_size = suggest_streaming_tile_size(
            shape,
            psf_xy_est=psf_xy_est,
            method=str(params.get("method", "ci_rl")),
            device=params.get("device"),
        )
        if progress is not None:
            progress({
                "event": "message",
                "message": f"Auto tile size: {effective_tile_size} x {effective_tile_size} px",
            })
    else:
        effective_tile_size = max(128, effective_tile_size)

    def _make_sink():
        if "tiff" in output_format_key:
            base_sink = TiledOmeTiffSink(
                output_path,
                shape=sink_shape,
                metadata=sink_metadata,
                tile_yx=(512, 512),
            )
        else:
            base_sink = ZarrPyramidSink(
                output_path,
                shape=sink_shape,
                metadata=sink_metadata,
                resume=False,
            )
        return (
            ProjectionPyramidSink(base_sink, source_shape=shape, mode=projection_key)
            if project_output
            else base_sink
        )

    def _is_out_of_memory(exc: Exception) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda oom" in text or ("cublas" in text and "alloc" in text)

    def _clear_cuda_cache() -> None:
        _release_cuda_memory(synchronize=True)

    def _stopped() -> bool:
        return bool(should_stop() if should_stop is not None else False)

    psf_cache: dict[int, np.ndarray] = {}
    psf_pixel_size_z_cache: dict[int, float] = {}

    def _psf_for_channel(ci: int) -> np.ndarray:
        cached = psf_cache.get(ci)
        if cached is not None:
            return cached
        if _stopped():
            raise RuntimeError("Stopped by user")
        em_list = params["emission_wavelengths"]
        em_wl = em_list[ci] if ci < len(em_list) else em_list[-1] if em_list else 520.0
        ex_list = params["excitation_wavelengths"]
        ex_wl = ex_list[ci] if ci < len(ex_list) else ex_list[-1] if ex_list else None
        if params["microscope_type"] != "confocal":
            ex_wl = None
        pinhole_list = params["pinhole_airy_units"]
        pinhole_airy = (
            pinhole_list[ci] if ci < len(pinhole_list)
            else pinhole_list[-1] if pinhole_list
            else _DEFAULT_PINHOLE_AIRY_UNITS
        )
        is_2d = shape[2] == 1
        use_2d_wf_auto = (
            is_2d
            and params["microscope_type"] == "widefield"
            and params["method"] in ("ci_rl", "ci_rl_tv")
            and params["two_d_mode"] == "auto"
        )
        psf_pixel_size_z_nm = params["pixel_size_z_nm"]
        if use_2d_wf_auto:
            n_z_psf = 65
            psf_pixel_size_z_nm = _estimate_two_d_wf_psf_z_nm(
                em_wl,
                params["na"],
                params["ri_sample"],
                params["pixel_size_xy_nm"],
            )
        elif not is_2d:
            n_z_psf = max(2 * shape[2] - 1, 1) | 1
        else:
            n_z_psf = 1

        airy_radius_nm = 0.61 * em_wl / params["na"]
        airy_radius_px = airy_radius_nm / params["pixel_size_xy_nm"]
        n_xy_psf = int(max(64, 2 * int(4 * airy_radius_px) + 1))
        if n_xy_psf % 2 == 0:
            n_xy_psf += 1
        if progress is not None:
            progress({"event": "message", "message": f"Generating PSF for channel {ci + 1}/{shape[1]}..."})
        psf = ci_generate_psf(
            na=params["na"],
            wavelength_nm=em_wl,
            pixel_size_xy_nm=params["pixel_size_xy_nm"],
            pixel_size_z_nm=psf_pixel_size_z_nm,
            n_xy=n_xy_psf,
            n_z=n_z_psf,
            ri_immersion=params["ri_immersion"],
            ri_sample=params["ri_sample"],
            ri_coverslip=params["ri_immersion"],
            ri_coverslip_design=params["ri_immersion"],
            ri_immersion_design=params["ri_immersion"],
            t_g=params["t_g"],
            t_g0=params["t_g0"],
            t_i0=params["t_i0"],
            z_p=params["z_p"],
            microscope_type=params["microscope_type"],
            excitation_nm=ex_wl,
            pinhole_airy_units=pinhole_airy,
            integrate_pixels=params["integrate_pixels"],
            n_subpixels=params["n_subpixels"],
            n_pupil=params["n_pupil"],
            device=params["device"],
        )
        psf_cache[ci] = psf
        psf_pixel_size_z_cache[ci] = float(psf_pixel_size_z_nm)
        return psf

    def _deconvolve_tile(tile_img: np.ndarray, psf: np.ndarray, ci: int) -> np.ndarray:
        if _stopped():
            raise RuntimeError("Stopped by user")
        effective_psf = psf
        use_2d_wf_auto = (
            tile_img.ndim == 2
            and params["microscope_type"] == "widefield"
            and params["method"] in ("ci_rl", "ci_rl_tv")
            and params["two_d_mode"] == "auto"
        )
        if tile_img.ndim == 2 and effective_psf.ndim == 3 and not use_2d_wf_auto:
            if effective_psf.shape[0] != 1:
                raise ValueError(f"Expected singleton-Z PSF for 2D tile, got {effective_psf.shape}")
            effective_psf = effective_psf[0]
        elif tile_img.ndim == 3 and effective_psf.ndim == 2:
            effective_psf = effective_psf[np.newaxis, :, :]
        if tile_img.ndim == 3 and effective_psf.ndim == 3 and effective_psf.shape[0] > tile_img.shape[0]:
            start = (effective_psf.shape[0] - tile_img.shape[0]) // 2
            effective_psf = effective_psf[start:start + tile_img.shape[0]]

        niter_list = params["niter_list"]
        niter = niter_list[ci] if ci < len(niter_list) else niter_list[-1]
        common = dict(
            niter=niter,
            offset=params["offset"],
            prefilter_sigma=params["prefilter_sigma"],
            start=params["start"],
            background=params["background"],
            convergence=params["convergence"],
            rel_threshold=params["rel_threshold"],
            check_every=params["check_every"],
            pixel_size_xy=params["pixel_size_xy_nm"],
            pixel_size_z=(
                psf_pixel_size_z_cache.get(ci, params["pixel_size_z_nm"])
                if use_2d_wf_auto
                else params["pixel_size_z_nm"]
            ),
            device=params["device"],
            tiling="none",
            iteration_callback=None,
            channel_index=ci,
        )
        if params["method"] == "ci_sparse_hessian":
            out = ci_sparse_hessian_deconvolve(
                tile_img,
                effective_psf,
                sparse_hessian_weight=params["sparse_hessian_weight"],
                sparse_hessian_reg=params["sparse_hessian_reg"],
                **common,
            )
        else:
            out = ci_rl_deconvolve(
                tile_img,
                effective_psf,
                tv_lambda=params["tv_lambda"] if params["method"] == "ci_rl_tv" else 0.0,
                damping=params["damping"],
                microscope_type=params["microscope_type"],
                two_d_mode=params["two_d_mode"],
                two_d_wf_aggressiveness=params["two_d_wf_aggressiveness"],
                two_d_wf_bg_radius_um=params["two_d_wf_bg_radius_um"],
                two_d_wf_bg_scale=params["two_d_wf_bg_scale"],
                **common,
            )
        if _stopped():
            raise RuntimeError("Stopped by user")
        result = out["result"]
        del out
        _release_cuda_memory()
        return result

    def _progress(payload: dict) -> None:
        if _stopped():
            raise RuntimeError("Stopped by user")
        if progress is not None:
            progress(payload)

    attempt_tile_size = max(128, int(effective_tile_size))
    while True:
        sink = _make_sink()
        try:
            summary = deconvolve_streaming(
                region_source,
                sink,
                psf_for_channel=_psf_for_channel,
                deconvolve_tile=_deconvolve_tile,
                tile_yx=(attempt_tile_size, attempt_tile_size),
                progress=_progress,
                resume=False,
                build_pyramids=True,
            )
            provenance = save_streaming_provenance(
                str(Path(output_path).with_suffix(Path(output_path).suffix + ".provenance.json")),
                source=region_source,
                sink=sink,
                params={
                    "method": params.get("method"),
                    "iterations": params.get("niter_list"),
                    "tile_size": attempt_tile_size,
                    "output_format": output_format,
                    "projection": projection_key if project_output else "full_stack",
                },
                summary=summary,
            )
            break
        except Exception as exc:
            abort = getattr(sink, "abort", None)
            if callable(abort):
                abort()
            if not _is_out_of_memory(exc) or attempt_tile_size <= 256:
                raise
            _clear_cuda_cache()
            _delete_output_path(output_path)
            next_tile_size = max(256, (attempt_tile_size // 2 // 64) * 64)
            if next_tile_size >= attempt_tile_size:
                next_tile_size = max(256, attempt_tile_size // 2)
            if progress is not None:
                progress({
                    "event": "message",
                    "message": f"GPU memory retry: reducing tile to {next_tile_size} x {next_tile_size} px",
                })
            attempt_tile_size = next_tile_size
    return {"summary": summary, "provenance": str(provenance), "output": output_path}


@dataclass
class _BatchItem:
    source_type: str
    display_name: str
    locator: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_obj: Any = None
    output_dir: str = ""
    status: str = "Queued"
    progress: int = 0
    output_path: str = ""
    message: str = ""

    def stable_key(self) -> str:
        return f"{self.source_type}:{self.locator}"

    def row_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "progress": self.progress,
            "source_type": self.source_type,
            "display_name": self.display_name,
            "shape": _batch_shape_label(self.metadata),
            "output_dir": self.output_dir,
            "output_path": self.output_path,
            "message": self.message,
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "display_name": self.display_name,
            "locator": self.locator,
            "metadata": dict(self.metadata),
            "output_dir": self.output_dir,
            "status": self.status,
            "progress": self.progress,
            "output_path": self.output_path,
            "message": self.message,
        }


def _batch_shape_label(metadata: dict) -> str:
    try:
        t = int(metadata.get("size_t", 1))
        c = int(metadata.get("size_c", 1))
        z = int(metadata.get("size_z", 1))
        y = int(metadata.get("size_y", 0))
        x = int(metadata.get("size_x", 0))
        if y and x:
            return f"T{t} C{c} Z{z} {y}x{x}"
    except Exception:
        pass
    return ""


def _format_batch_output_display(path_text: str, *, max_chars: int = 96) -> str:
    text = str(path_text or "")
    if not text:
        return ""
    name = text.rstrip("\\/").replace("\\", "/").split("/")[-1]
    if len(name) <= max_chars:
        return name
    keep = max(8, max_chars - 3)
    head = max(4, keep // 3)
    tail = max(4, keep - head)
    return f"{name[:head]}...{name[-tail:]}"


def _format_batch_folder_display(path_text: str, *, max_chars: int = 44) -> str:
    text = str(path_text or "").strip().rstrip("\\/")
    if not text:
        return ""
    parts = [p for p in text.replace("\\", "/").split("/") if p]
    if not parts:
        return text
    leaf = parts[-1]
    if len(parts) == 1 or len(leaf) >= max_chars:
        name = leaf
    else:
        parent = parts[-2] if len(parts) >= 2 else ""
        name = f".../{parent}/{leaf}" if parent else leaf
        if len(name) > max_chars:
            name = f".../{leaf}"
    if len(name) <= max_chars:
        return name
    keep = max(8, max_chars - 3)
    head = max(4, keep // 3)
    tail = max(4, keep - head)
    return f"{name[:head]}...{name[-tail:]}"


def _batch_source_stem(name: str) -> str:
    stem = str(name or "image").strip().rstrip("\\/").replace("\\", "/").split("/")[-1]
    lower = stem.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".ome.zarr", ".tiff", ".tif", ".zarr"):
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return _safe_filename_stem(stem or "image")


def _batch_output_base_name(item: _BatchItem) -> str:
    if item.source_type == "leica":
        save_child_name = item.metadata.get("save_child_name")
        leica_context = item.metadata.get("leica_context")
        if not save_child_name and isinstance(leica_context, dict):
            save_child_name = leica_context.get("save_child_name")
        if save_child_name:
            return str(save_child_name)
    return item.display_name


def _batch_output_path(
    item: _BatchItem,
    output_dir: str | Path,
    output_format: str,
    params: dict,
    projection_mode: str = "Full stack",
) -> str:
    del params
    projection_key = str(projection_mode or "Full stack").strip().lower()
    size_z = int(item.metadata.get("size_z", 1) or 1)
    projection_tag = ""
    if size_z > 1 and projection_key not in {"", "full stack", "full", "none"}:
        projection_tag = f"_{_safe_filename_stem(projection_key)}"
    stem = _safe_filename_stem(f"{_batch_source_stem(_batch_output_base_name(item))}_decon{projection_tag}")
    suffix = ".ome.tiff" if "tiff" in output_format.lower() else ".ome.zarr"
    return str(Path(output_dir) / f"{stem}{suffix}")


def _delete_output_path(path_text: str) -> None:
    path = Path(path_text)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _metadata_from_region_source(source) -> dict[str, Any]:
    meta = dict(getattr(source, "metadata", {}) or {})
    shape = tuple(int(v) for v in getattr(source, "shape", (1, 1, 1, 0, 0)))
    for axis, value in zip(("size_t", "size_c", "size_z", "size_y", "size_x"), shape):
        meta[axis] = value
    return meta


def _file_or_zarr_batch_item(path_text: str, source_type: str) -> _BatchItem:
    path = Path(path_text)
    meta: dict[str, Any] = {"name": path.name}
    message = ""
    try:
        from core.streaming import open_region_source
        source = open_region_source(path)
        meta.update(_metadata_from_region_source(source))
    except Exception as exc:
        message = f"Metadata probe deferred: {exc}"
    return _BatchItem(
        source_type=source_type,
        display_name=path.name,
        locator=str(path),
        metadata=meta,
        message=message,
    )


class _BatchDeconvolveWorker(QThread):
    item_changed = pyqtSignal(int, object)
    log = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(
        self,
        items: list[_BatchItem],
        params: dict,
        output_dir: str,
        output_format: str,
        *,
        projection_mode: str = "Full stack",
        tile_size: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.items = items
        self.params = dict(params)
        self.output_dir = output_dir
        self.output_format = output_format
        self.projection_mode = str(projection_mode or "Full stack")
        self.tile_size = int(tile_size)

    def _emit_item(self, index: int, **updates: Any) -> None:
        item = self.items[index]
        for key, value in updates.items():
            setattr(item, key, value)
        self.item_changed.emit(index, item.row_payload())

    def _open_region_source(self, item: _BatchItem):
        if item.source_type in ("file", "zarr"):
            from core.streaming import open_region_source
            return open_region_source(item.locator)
        if item.source_type == "omero":
            source = _build_omero_source(item.source_obj)
            if bool(getattr(source, "is_pyramidal", False)):
                return _OmeroPyramidRegionSource(source)
            return _TimepointRegionSource(source, source_id=f"omero:{item.locator}")
        if item.source_type == "leica":
            source = _build_leica_source(item.source_obj)
            return _TimepointRegionSource(source, source_id=f"leica:{item.locator}")
        raise ValueError(f"Unsupported source type: {item.source_type}")

    def run(self):
        cancelled = False
        for index, item in enumerate(self.items):
            if self.isInterruptionRequested():
                cancelled = True
                break
            if item.status == "Done":
                continue
            monitor = None
            try:
                item_output_dir = item.output_dir or self.output_dir
                Path(item_output_dir).expanduser().mkdir(parents=True, exist_ok=True)
                output_path = _batch_output_path(
                    item,
                    item_output_dir,
                    self.output_format,
                    self.params,
                    self.projection_mode,
                )
                self._emit_item(index, status="Running", progress=0, output_path=output_path, message="Opening source")
                self.log.emit(f"Batch: opening {item.display_name}")
                source = self._open_region_source(item)
                self._emit_item(index, message="Validating metadata", progress=0)
                source.metadata.update(_batch_metadata_overlay(item.metadata))
                source.metadata.update(_streaming_output_metadata(source.metadata, self.params, source_name=item.display_name))
                item.metadata.update(_metadata_from_region_source(source))
                _validate_channel_parameter_lists(self.params, int(source.shape[1]))
                self._emit_item(index, message="Preparing output", progress=0)

                _delete_output_path(output_path)
                monitor = _RunMetricsMonitor()
                monitor.start()

                def _progress(payload: dict) -> None:
                    event = payload.get("event")
                    if event == "tile_done":
                        done = int(payload.get("done", 0))
                        total = max(int(payload.get("total", 1)), 1)
                        tile_index = int(payload.get("tile_index", 0)) + 1
                        tile_count = max(int(payload.get("tile_count", 1)), 1)
                        self._emit_item(
                            index,
                            progress=min(99, int(done * 100 / total)),
                            message=f"Done tile {tile_index}/{tile_count} ({done}/{total} total)",
                        )
                    elif event == "tile_start":
                        done = int(payload.get("done", 0))
                        total = max(int(payload.get("total", 1)), 1)
                        tile_index = int(payload.get("tile_index", 0)) + 1
                        tile_count = max(int(payload.get("tile_count", 1)), 1)
                        timepoint = int(payload.get("timepoint", 0)) + 1
                        channel = int(payload.get("channel", 0)) + 1
                        self._emit_item(
                            index,
                            progress=min(98, int(done * 100 / total)),
                            message=f"Processing tile {tile_index}/{tile_count} (T{timepoint} C{channel})",
                        )
                    elif event == "pyramid_start":
                        self._emit_item(index, progress=99, message="Building pyramid")
                    elif event == "message":
                        self._emit_item(index, message=str(payload.get("message", "")))

                result = _run_streaming_deconvolution_job(
                    source,
                    params=self.params,
                    output_path=output_path,
                    output_format=self.output_format,
                    tile_size=self.tile_size,
                    projection_mode=self.projection_mode,
                    progress=_progress,
                    should_stop=self.isInterruptionRequested,
                )
                metrics = monitor.stop() if monitor is not None else {}
                self._emit_item(index, status="Done", progress=100, message="Done", output_path=output_path)
                self.log.emit(f"Batch: done {item.display_name} -> {output_path}")
                for line in _resource_metric_lines(metrics):
                    self.log.emit(f"  {line}")
                self.log.emit(f"  Provenance: {result.get('provenance')}")
            except Exception as exc:
                if monitor is not None:
                    try:
                        monitor.stop()
                    except Exception:
                        pass
                if "Stopped by user" in str(exc) or self.isInterruptionRequested():
                    self._emit_item(index, status="Cancelled", progress=max(item.progress, 0), message="Cancelled")
                    cancelled = True
                    break
                message = str(exc) or exc.__class__.__name__
                self._emit_item(index, status="Failed", message=message)
                self.log.emit(f"Batch: failed {item.display_name}: {message}")
                self.log.emit(traceback.format_exc())
                continue
            finally:
                _release_cuda_memory(synchronize=True)
        self.finished.emit({"cancelled": cancelled})


class _BatchDeconvolverDialog(QDialog):
    _STATUS_STYLE = {
        "Queued": ("#2b2b2b", "#d9d9d9", "#8f9aa6"),
        "Running": ("#172b3a", "#d7ecff", "#5db2ff"),
        "Done": ("#193226", "#dff5e5", "#58c878"),
        "Failed": ("#3a2023", "#ffd9d9", "#ff7676"),
        "Cancelled": ("#3a321f", "#ffe6ad", "#e5b84f"),
    }

    def __init__(self, host, parent=None):
        super().__init__(parent or host)
        self._host = host
        self._items: list[_BatchItem] = []
        self._worker: Optional[_BatchDeconvolveWorker] = None
        self._batch_started_at: Optional[float] = None
        self._batch_finished_at: Optional[float] = None
        self._batch_active_rows: list[int] = []
        self._batch_status_label = "Ready"
        self._batch_timer = QTimer(self)
        self._batch_timer.setInterval(1000)
        self._batch_timer.timeout.connect(self._update_batch_status)
        self.setWindowTitle("Batch Deconvolver")
        self.setMinimumSize(1120, 620)
        self.setModal(False)
        self._build_ui()

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            QMessageBox.information(self, "Batch Running", "Stop or wait for the current batch before closing this dialog.")
            event.ignore()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        add_bar = QHBoxLayout()
        btn_open = QPushButton("Open\u2026")
        btn_open.clicked.connect(self._on_add_files)
        add_bar.addWidget(btn_open)
        btn_zarr = QPushButton("Open Zarr\u2026")
        btn_zarr.clicked.connect(self._on_add_zarr)
        add_bar.addWidget(btn_zarr)
        btn_leica = QPushButton("Open Leica\u2026")
        btn_leica.clicked.connect(self._on_add_leica)
        add_bar.addWidget(btn_leica)
        btn_omero = QPushButton("Open OMERO\u2026")
        btn_omero.clicked.connect(self._on_add_omero)
        add_bar.addWidget(btn_omero)
        add_bar.addStretch()
        layout.addLayout(add_bar)

        settings_row = QHBoxLayout()
        self._settings_path = QLineEdit()
        self._settings_path.setPlaceholderText("Saved settings JSON")
        settings_row.addWidget(QLabel("Settings:"))
        settings_row.addWidget(self._settings_path, stretch=1)
        btn_settings = QPushButton("Browse\u2026")
        btn_settings.clicked.connect(self._choose_settings)
        settings_row.addWidget(btn_settings)
        layout.addLayout(settings_row)

        output_row = QHBoxLayout()
        self._output_dir = QLineEdit(str(_default_downloads_dir()))
        output_row.addWidget(QLabel("Output folder:"))
        output_row.addWidget(self._output_dir, stretch=1)
        btn_output = QPushButton("Browse\u2026")
        btn_output.clicked.connect(self._choose_output_dir)
        output_row.addWidget(btn_output)
        self._format_combo = QComboBox()
        self._format_combo.addItems(["OME-TIFF", "OME-Zarr"])
        output_row.addWidget(QLabel("Format:"))
        output_row.addWidget(self._format_combo)
        self._projection_combo = QComboBox()
        self._projection_combo.addItems(["Full stack", "MIP", "SUM", "Mean"])
        self._projection_combo.setToolTip("For 3D images, save only a Z projection instead of the full Z stack.")
        output_row.addWidget(QLabel("Z output:"))
        output_row.addWidget(self._projection_combo)
        layout.addLayout(output_row)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels([
            "Status", "Progress", "Source", "Name", "Shape", "Folder", "Output", "Message"
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._table.verticalHeader().setDefaultSectionSize(30)
        self._table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: #3a3a3a;
                alternate-background-color: #252525;
                selection-background-color: #244a64;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #343434;
                color: #f0f0f0;
                padding: 5px 7px;
                border: 0;
                border-right: 1px solid #464646;
                border-bottom: 1px solid #464646;
                font-weight: 600;
            }
            QTableCornerButton::section {
                background-color: #343434;
                border: 0;
            }
            """
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, stretch=1)

        action_row = QHBoxLayout()
        self._btn_remove = QPushButton("Remove Selected")
        self._btn_remove.clicked.connect(self._remove_selected)
        action_row.addWidget(self._btn_remove)
        self._btn_reset = QPushButton("Reset Selected")
        self._btn_reset.clicked.connect(self._reset_selected)
        action_row.addWidget(self._btn_reset)
        self._btn_clear = QPushButton("Clear Completed/Failed")
        self._btn_clear.clicked.connect(self._clear_finished)
        action_row.addWidget(self._btn_clear)
        action_row.addStretch()
        self._btn_start = QPushButton("Start")
        self._btn_start.clicked.connect(self._on_start_stop_batch)
        action_row.addWidget(self._btn_start)
        layout.addLayout(action_row)

        self._batch_status = QStatusBar(self)
        self._batch_status.setSizeGripEnabled(False)
        self._batch_status.showMessage("Ready")
        layout.addWidget(self._batch_status)

    def _selected_rows(self) -> list[int]:
        return sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)

    def _set_running(self, running: bool) -> None:
        self._btn_start.setEnabled(True)
        self._btn_start.setText("Stop" if running else "Start")
        self._btn_start.setToolTip(
            "Stop after the current tile/plane checkpoint"
            if running else
            "Start the queued batch"
        )
        self._btn_start.setStyleSheet(
            "QPushButton { background-color: #8a3b3b; color: white; font-weight: bold; padding: 6px 14px; }"
            if running else
            ""
        )
        self._btn_remove.setEnabled(not running)
        self._btn_reset.setEnabled(not running)
        self._btn_clear.setEnabled(not running)

    def _batch_completed_units(self) -> float:
        units = 0.0
        active_rows = self._batch_active_rows or [
            i for i, item in enumerate(self._items) if item.status != "Done"
        ]
        for row in active_rows:
            if row < 0 or row >= len(self._items):
                continue
            item = self._items[row]
            if item.status in {"Done", "Failed", "Cancelled"}:
                units += 1.0
            elif item.status == "Running":
                units += max(0.0, min(float(item.progress) / 100.0, 0.99))
        return units

    def _reset_batch_status_clock_if_idle(self) -> None:
        if self._worker is not None:
            return
        self._batch_timer.stop()
        self._batch_started_at = None
        self._batch_finished_at = None
        self._batch_active_rows = []
        self._batch_status_label = "Ready"

    def _update_batch_status(self) -> None:
        status_bar = getattr(self, "_batch_status", None)
        if status_bar is None:
            return
        if self._batch_started_at is None:
            queued = sum(1 for item in self._items if item.status == "Queued")
            status_bar.showMessage(f"Ready | {queued} queued")
            return

        now = self._batch_finished_at or time.time()
        elapsed = max(0.0, now - self._batch_started_at)
        total = max(len(self._batch_active_rows), 1)
        completed = min(self._batch_completed_units(), float(total))

        if self._batch_finished_at is not None:
            eta = 0.0
            end_ts = self._batch_finished_at
        elif completed > 0.0:
            seconds_per_image = elapsed / completed
            eta = max(0.0, seconds_per_image * (total - completed))
            end_ts = time.time() + eta
        else:
            eta = None
            end_ts = None

        eta_text = _format_duration_hm(eta) if eta is not None else "--:--"
        end_text = time.strftime("%Y-%m-%d %H:%M", time.localtime(end_ts)) if end_ts else "--"
        status_bar.showMessage(
            f"{self._batch_status_label} | "
            f"Elapsed {_format_duration_hm(elapsed)} | "
            f"ETA {eta_text} | "
            f"End {end_text} | "
            f"Images {completed:.1f}/{total}"
        )

    def _on_start_stop_batch(self) -> None:
        if self._worker is not None:
            self._stop_batch()
            return
        self._start_batch()

    def _add_items(self, items: Sequence[_BatchItem]) -> None:
        existing = {item.stable_key() for item in self._items}
        output_dir = self._output_dir.text().strip()
        for item in items:
            if item.stable_key() in existing:
                continue
            if not item.output_dir:
                item.output_dir = output_dir
            self._items.append(item)
            existing.add(item.stable_key())
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._update_row(row, item.row_payload())
        self._reset_batch_status_clock_if_idle()
        self._update_batch_status()

    def _update_row(self, row: int, payload: dict[str, Any]) -> None:
        values = [
            str(payload.get("status", "")),
            f"{int(payload.get('progress', 0))}%",
            str(payload.get("source_type", "")),
            str(payload.get("display_name", "")),
            str(payload.get("shape", "")),
            _format_batch_folder_display(str(payload.get("output_dir", ""))),
            _format_batch_output_display(str(payload.get("output_path", ""))),
            str(payload.get("message", "")),
        ]
        full_values = [
            str(payload.get("status", "")),
            f"{int(payload.get('progress', 0))}%",
            str(payload.get("source_type", "")),
            str(payload.get("display_name", "")),
            str(payload.get("shape", "")),
            str(payload.get("output_dir", "")),
            str(payload.get("output_path", "")),
            str(payload.get("message", "")),
        ]
        status = values[0]
        row_bg, row_fg, accent = self._STATUS_STYLE.get(status, self._STATUS_STYLE["Queued"])
        bg_brush = QBrush(QColor(row_bg))
        fg_brush = QBrush(QColor(row_fg))
        accent_brush = QBrush(QColor(accent))
        for col, text in enumerate(values):
            item = self._table.item(row, col)
            if item is None:
                item = QTableWidgetItem()
                self._table.setItem(row, col, item)
            item.setText(text)
            item.setBackground(bg_brush)
            item.setForeground(accent_brush if col in (0, 1) else fg_brush)
            item.setToolTip(full_values[col])
            if col in (0, 1, 2, 4):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            font = item.font()
            font.setBold(col == 0)
            item.setFont(font)

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Images",
            self._host._last_open_dir,
            "Images (*.ome.tiff *.ome.tif *.tiff *.tif *.nd2 *.czi *.lif);;All Files (*)",
        )
        if not paths:
            return
        self._host._last_open_dir = str(Path(paths[0]).parent)
        self._add_items([_file_or_zarr_batch_item(path, "file") for path in paths])

    def _on_add_zarr(self) -> None:
        dialog = QFileDialog(self, "Add OME-Zarr Folders", self._host._last_zarr_dir)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        for view_type in (QListView, QTreeView):
            for view in dialog.findChildren(view_type):
                view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        paths = [p for p in dialog.selectedFiles() if p]
        if not paths:
            return
        self._host._last_zarr_dir = str(Path(paths[0]).parent)
        self._add_items([_file_or_zarr_batch_item(path, "zarr") for path in paths])

    def _on_add_leica(self) -> None:
        try:
            from leica_browser_qt import LeicaBrowserDialog
        except ImportError:
            QMessageBox.critical(self, "Leica Browser Missing", "leica-browser-qt is not installed.")
            return
        start_dir = _accessible_directory_or_home(self._host._last_leica_dir)
        dialog = LeicaBrowserDialog(roots=[start_dir], selection_mode="multiple", parent=self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        browsed_root = self._host._leica_dialog_current_root(dialog)
        if browsed_root is not None:
            self._host._set_last_leica_dir(browsed_root, persist=True)
        if not accepted:
            return
        if hasattr(dialog, "selected_contexts"):
            contexts = list(dialog.selected_contexts() or [])
        else:
            context = dialog.selected_context()
            contexts = [context] if context is not None else []
        items = []
        for ctx in contexts:
            meta = dict(getattr(ctx, "metadata", {}) or {})
            meta.update({
                "name": getattr(ctx, "name", "Leica image"),
                "size_t": max(int(getattr(ctx, "size_t", 1) or meta.get("ts", 1)), 1),
                "size_c": max(int(getattr(ctx, "size_c", 1) or meta.get("channels", 1)), 1),
                "size_z": max(int(getattr(ctx, "size_z", 1) or meta.get("zs", 1)), 1),
                "size_y": max(int(getattr(ctx, "size_y", 0) or meta.get("ys", 0)), 0),
                "size_x": max(int(getattr(ctx, "size_x", 0) or meta.get("xs", 0)), 0),
                "leica_context": ctx.to_dict() if hasattr(ctx, "to_dict") else {},
            })
            locator = f"{getattr(ctx, 'container_path', '')}::{getattr(ctx, 'internal_path', getattr(ctx, 'name', ''))}"
            display_name = _batch_output_base_name(
                _BatchItem("leica", str(getattr(ctx, "name", "Leica image")), locator, meta)
            )
            items.append(_BatchItem("leica", display_name, locator, meta, source_obj=ctx))
        self._add_items(items)

    def _on_add_omero(self) -> None:
        try:
            from omero_browser_qt import LoginDialog, OmeroBrowserDialog, OmeroGateway
        except ImportError:
            QMessageBox.warning(self, "OMERO not available", "omero-browser-qt is not installed.")
            return
        if self._host._omero_gw is None:
            try:
                self._host._omero_gw = OmeroGateway()
            except RuntimeError:
                from PyQt6.QtCore import QObject as _QObj
                inst = OmeroGateway._instance
                _QObj.__init__(inst)
                OmeroGateway.__init__(inst)
                self._host._omero_gw = inst
        gw = self._host._omero_gw
        if gw.is_connected() and not self._host._omero_session_is_reusable():
            gw.disconnect()
            self._host._omero_session_deadline = 0.0
        if not self._host._omero_session_is_reusable():
            dlg = LoginDialog(self, gateway=gw)
            self._host._configure_omero_login_dialog(dlg)
            if dlg.exec() != LoginDialog.DialogCode.Accepted:
                return
            self._host._refresh_omero_session_deadline()
        else:
            self._host._refresh_omero_session_deadline()
        browser = OmeroBrowserDialog(self, gateway=gw, multiselect=True)
        if browser.exec() != OmeroBrowserDialog.DialogCode.Accepted:
            return
        self._host._refresh_omero_session_deadline()
        images = list(browser.get_selected_images() or [])
        items = []
        for image in images:
            image_id = image.getId() if hasattr(image, "getId") else id(image)
            name = image.getName() if hasattr(image, "getName") else f"OMERO image {image_id}"
            meta = {"name": name, "id": image_id}
            try:
                source = _build_omero_source(image)
                meta.update(dict(source.metadata))
            except Exception as exc:
                meta["probe_error"] = str(exc)
            items.append(_BatchItem("omero", str(name), str(image_id), meta, source_obj=image))
        self._add_items(items)

    def _choose_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Batch Settings",
            self._host._last_settings_dir,
            "JSON Settings (*.json);;All Files (*)",
        )
        if path:
            self._host._last_settings_dir = str(Path(path).parent)
            self._settings_path.setText(path)

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Output Folder", self._output_dir.text().strip())
        if path:
            self._output_dir.setText(path)

    def _remove_selected(self) -> None:
        for row in self._selected_rows():
            del self._items[row]
            self._table.removeRow(row)
        self._reset_batch_status_clock_if_idle()
        self._update_batch_status()

    def _reset_selected(self) -> None:
        rows = self._selected_rows()
        if not rows:
            rows = list(range(len(self._items) - 1, -1, -1))
        for row in rows:
            item = self._items[row]
            item.status = "Queued"
            item.progress = 0
            item.message = ""
            self._update_row(row, item.row_payload())
        self._reset_batch_status_clock_if_idle()
        self._update_batch_status()

    def _clear_finished(self) -> None:
        for row in range(len(self._items) - 1, -1, -1):
            if self._items[row].status in {"Done", "Failed", "Cancelled"}:
                del self._items[row]
                self._table.removeRow(row)
        self._reset_batch_status_clock_if_idle()
        self._update_batch_status()

    def _load_batch_params(self) -> dict:
        settings_path = Path(self._settings_path.text().strip())
        if not settings_path.is_file():
            raise ValueError("Choose a saved settings JSON before starting the batch.")
        with settings_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._host._apply_settings(data)
        params = self._host._collect_params()
        params["movie"] = {"enabled": False}
        params["compute_metrics"] = False
        return params

    def _start_batch(self) -> None:
        if self._worker is not None:
            return
        if not self._items:
            QMessageBox.information(self, "Batch Deconvolver", "Add at least one image first.")
            return
        try:
            params = self._load_batch_params()
            default_output_dir = Path(self._output_dir.text().strip()).expanduser()
            default_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Batch Setup Error", str(exc))
            return
        worker_items = []
        try:
            for row, item in enumerate(self._items):
                item_output_dir = str(Path(item.output_dir or default_output_dir).expanduser())
                Path(item_output_dir).mkdir(parents=True, exist_ok=True)
                if not item.output_dir:
                    item.output_dir = item_output_dir
                    self._update_row(row, item.row_payload())
                worker_items.append(_BatchItem(
                    item.source_type,
                    item.display_name,
                    item.locator,
                    dict(item.metadata),
                    source_obj=item.source_obj,
                    output_dir=item_output_dir,
                    status=item.status,
                    progress=item.progress,
                    output_path=item.output_path,
                    message=item.message,
                ))
        except Exception as exc:
            QMessageBox.critical(self, "Batch Setup Error", f"Could not create output folder: {exc}")
            return
        self._batch_started_at = time.time()
        self._batch_finished_at = None
        self._batch_active_rows = [
            row for row, item in enumerate(self._items) if item.status != "Done"
        ]
        self._batch_status_label = "Running"
        self._batch_timer.start()
        self._update_batch_status()
        self._set_running(True)
        self._worker = _BatchDeconvolveWorker(
            worker_items,
            params,
            str(default_output_dir),
            self._format_combo.currentText(),
            projection_mode=self._projection_combo.currentText(),
            parent=self,
        )
        self._worker.item_changed.connect(self._on_worker_item_changed)
        self._worker.log.connect(self._host._log)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _stop_batch(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()
            _release_cuda_memory(synchronize=True)
            self._batch_status_label = "Stopping"
            self._update_batch_status()
            self._btn_start.setEnabled(False)
            self._btn_start.setText("Stopping...")
            self._btn_start.setToolTip("Waiting for the current checkpoint to stop")

    def _on_worker_item_changed(self, row: int, payload: object) -> None:
        if not isinstance(payload, dict) or row < 0 or row >= len(self._items):
            return
        item = self._items[row]
        item.status = str(payload.get("status", item.status))
        item.progress = int(payload.get("progress", item.progress) or 0)
        item.output_dir = str(payload.get("output_dir", item.output_dir))
        item.output_path = str(payload.get("output_path", item.output_path))
        item.message = str(payload.get("message", item.message))
        self._update_row(row, payload)
        self._update_batch_status()

    def _on_worker_finished(self, result: object) -> None:
        self._worker = None
        self._set_running(False)
        _release_cuda_memory(synchronize=True)
        self._batch_timer.stop()
        self._batch_finished_at = time.time()
        if isinstance(result, dict) and result.get("cancelled"):
            self._batch_status_label = "Cancelled"
            self._host._status.showMessage("Batch cancelled", 5000)
        else:
            self._batch_status_label = "Complete"
            self._host._status.showMessage("Batch complete", 5000)
        self._update_batch_status()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

def _detect_gpu_info() -> str:
    """Return a short string with torch + GPU version info for the title bar."""
    try:
        import torch
        parts = [f"torch {torch.__version__}"]
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda or "?"
            parts.append(f"{name}  CUDA {cuda_ver}")
        else:
            parts.append("CPU only")
        return "  |  ".join(parts)
    except Exception:
        return "torch not available"


class DeconvolveCIWindow(QMainWindow):
    def __init__(self, *, movie_available: bool = False, fitpsf_available: bool = False, dlref_available: bool = False):
        super().__init__()
        gpu_info = _detect_gpu_info()
        self.setWindowTitle(f"CI Deconvolve — {gpu_info}")
        app_icon = _load_app_icon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        self.setMinimumSize(1430, 700)

        # State
        self._input_channels: list[np.ndarray] = []
        self._input_source: Optional[_BaseTimepointSource] = None
        self._input_source_factory: Optional[Callable[[], _BaseTimepointSource]] = None
        self._loaded_timepoint: Optional[int] = None
        self._preview_outputs_by_t: dict[int, list[np.ndarray]] = {}
        self._metadata: dict = {}
        self._worker: Optional[_DeconvolveWorker] = None
        self._save_worker: Optional[_SaveTSeriesWorker] = None
        self._fit_psf_worker: Optional[_FitPsfWorker] = None
        self._fit_psf_started_at: Optional[float] = None
        self._monitor: Optional[_ResourceMonitor] = None
        self._log_dialog: Optional[_LogDialog] = None
        self._batch_dialog: Optional[_BatchDeconvolverDialog] = None
        self._log_lines: list[str] = []
        self._log_running = False
        self._compute_image_metrics = False
        self._live_charts_enabled = False
        self._live_log_scale = False
        self._last_live_update_s: float = 0.0
        self._last_final_iteration_payload: Optional[dict] = None
        self._movie_available = bool(movie_available)
        self._fitpsf_available = bool(fitpsf_available)
        self._dlref_available = bool(dlref_available)
        self._log_emitter = _GuiLogEmitter(self)
        self._log_handler = _QtLogHandler(self._log_emitter)
        self._log_emitter.line.connect(self._log_from_logging)
        logging.getLogger().addHandler(self._log_handler)
        self._input_path: Optional[Path] = None
        self._last_open_dir: str = _default_settings_dir()
        self._last_leica_dir: str = _default_settings_dir()
        self._last_zarr_dir: str = _default_settings_dir()
        self._last_save_dir: str = _default_settings_dir()
        self._last_settings_dir: str = _default_settings_dir()
        self._restore_last_browse_dirs()
        self._movie_default_path: Optional[str] = None
        self._omero_gw = None  # OmeroGateway instance (lazy)
        self._omero_session_deadline: float = 0.0
        self._excitation_saved: str = "488"  # remembered when field is disabled
        self._pinhole_airy_saved: str = str(_DEFAULT_PINHOLE_AIRY_UNITS)

        self._build_ui()

        # Start the resource monitor immediately and keep it running
        self._monitor = _ResourceMonitor(parent=self)
        self._monitor.metrics_updated.connect(self._monitor_bar.update_metrics)
        self._monitor.start()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Left: controls (scrollable) ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(440)
        scroll.setMaximumWidth(500)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(6)
        scroll.setWidget(ctrl_widget)
        splitter.addWidget(scroll)

        # Title
        title = QLabel("CI Deconvolve")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        ctrl_layout.addWidget(title)

        def _set_field_tooltip(form_layout: QFormLayout, widget: QWidget, text: str) -> None:
            widget.setToolTip(text)
            label = form_layout.labelForField(widget)
            if label is not None:
                label.setToolTip(text)

        # --- Method ---
        method_group = QGroupBox("Method")
        ml = QFormLayout()
        method_group.setLayout(ml)

        self._method_combo = NoWheelComboBox()
        _method_items = ["ci_rl"]
        if self._dlref_available:
            _method_items.append("ci_rl_dl")
        _method_items += ["ci_rl_tv", "ci_sparse_hessian"]
        self._method_combo.addItems(_method_items)
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        ml.addRow("Method:", self._method_combo)

        self._le_niter = QLineEdit("80")
        self._le_niter.setToolTip(
            "Iterations per channel, comma-separated.\n"
            "E.g. '50' for all channels, or '50, 80' for ch1=50, ch2=80."
        )
        ml.addRow("Iterations:", self._le_niter)

        self._conv_combo = NoWheelComboBox()
        self._conv_combo.addItems(["auto", "fixed"])
        ml.addRow("Convergence:", self._conv_combo)

        self._sp_rel_thresh = NoWheelDoubleSpinBox()
        self._sp_rel_thresh.setRange(1e-8, 1.0)
        self._sp_rel_thresh.setDecimals(6)
        self._sp_rel_thresh.setSingleStep(0.0001)
        self._sp_rel_thresh.setValue(0.001)
        ml.addRow("Rel. threshold:", self._sp_rel_thresh)

        ctrl_layout.addWidget(method_group)

        advanced_section = CollapsibleSection("Advanced Parameters", expanded=False)
        advanced_layout = advanced_section.content_layout()

        # --- Method tuning ---
        method_adv_group = QGroupBox("Method Tuning")
        aml = QFormLayout()
        method_adv_group.setLayout(aml)

        self._sp_tv_lambda = NoWheelDoubleSpinBox()
        self._sp_tv_lambda.setRange(0.0, 1.0)
        self._sp_tv_lambda.setDecimals(6)
        self._sp_tv_lambda.setSingleStep(0.0001)
        self._sp_tv_lambda.setValue(0.0001)
        aml.addRow("TV lambda:", self._sp_tv_lambda)
        self._tv_lambda_label = aml.labelForField(self._sp_tv_lambda)  # type: ignore

        self._damping_combo = NoWheelComboBox()
        self._damping_combo.addItems(["none", "auto", "manual"])
        self._damping_combo.currentTextChanged.connect(self._on_damping_changed)
        aml.addRow("Damping:", self._damping_combo)
        self._damping_label = aml.labelForField(self._damping_combo)  # type: ignore

        self._sp_damping = NoWheelDoubleSpinBox()
        self._sp_damping.setRange(0.0, 100.0)
        self._sp_damping.setDecimals(2)
        self._sp_damping.setSingleStep(0.1)
        self._sp_damping.setValue(3.0)
        self._sp_damping.setEnabled(False)
        aml.addRow("Damping value:", self._sp_damping)
        self._damping_value_label = aml.labelForField(self._sp_damping)  # type: ignore

        self._two_d_mode_combo = NoWheelComboBox()
        self._two_d_mode_combo.addItems(["Auto", "Legacy 2D"])
        self._two_d_mode_combo.currentTextChanged.connect(self._refresh_two_d_wf_expert_state)
        aml.addRow("2D WF model:", self._two_d_mode_combo)
        self._two_d_mode_label = aml.labelForField(self._two_d_mode_combo)  # type: ignore

        self._sp_sparse_weight = NoWheelDoubleSpinBox()
        self._sp_sparse_weight.setRange(0.0, 1.0)
        self._sp_sparse_weight.setDecimals(4)
        self._sp_sparse_weight.setSingleStep(0.01)
        self._sp_sparse_weight.setValue(0.6)
        aml.addRow("Sparse weight:", self._sp_sparse_weight)
        self._sparse_weight_label = aml.labelForField(self._sp_sparse_weight)  # type: ignore

        self._sp_sparse_reg = NoWheelDoubleSpinBox()
        self._sp_sparse_reg.setRange(0.0, 1.0)
        self._sp_sparse_reg.setDecimals(4)
        self._sp_sparse_reg.setSingleStep(0.001)
        self._sp_sparse_reg.setValue(0.98)
        aml.addRow("Sparse reg:", self._sp_sparse_reg)
        self._sparse_reg_label = aml.labelForField(self._sp_sparse_reg)  # type: ignore

        self._bg_combo = NoWheelComboBox()
        self._bg_combo.addItems(["auto", "manual"])
        self._bg_combo.currentTextChanged.connect(self._on_bg_changed)
        aml.addRow("Background:", self._bg_combo)

        self._sp_bg_value = NoWheelDoubleSpinBox()
        self._sp_bg_value.setRange(0.0, 1e9)
        self._sp_bg_value.setDecimals(2)
        self._sp_bg_value.setValue(0.0)
        self._sp_bg_value.setEnabled(False)
        aml.addRow("BG value:", self._sp_bg_value)

        self._offset_combo = NoWheelComboBox()
        self._offset_combo.addItems(["auto", "none", "manual"])
        self._offset_combo.currentTextChanged.connect(self._on_offset_changed)
        aml.addRow("Offset:", self._offset_combo)

        self._sp_offset = NoWheelDoubleSpinBox()
        self._sp_offset.setRange(0.0, 1000.0)
        self._sp_offset.setDecimals(1)
        self._sp_offset.setSingleStep(1.0)
        self._sp_offset.setValue(5.0)
        self._sp_offset.setEnabled(False)
        aml.addRow("Offset value:", self._sp_offset)

        self._sp_prefilter = NoWheelDoubleSpinBox()
        self._sp_prefilter.setRange(0.0, 5.0)
        self._sp_prefilter.setDecimals(2)
        self._sp_prefilter.setSingleStep(0.1)
        self._sp_prefilter.setValue(0.0)
        aml.addRow("Prefilter sigma:", self._sp_prefilter)

        self._start_combo = NoWheelComboBox()
        self._start_combo.addItems([
            "auto",
            "flat",
            "percentile_flat",
            "observed",
            "observed_bgsub",
            "lowpass",
            "lowpass_bgsub",
            "hybrid",
        ])
        aml.addRow("Start:", self._start_combo)

        self._sp_check_every = NoWheelSpinBox()
        self._sp_check_every.setRange(1, 1000)
        self._sp_check_every.setValue(5)
        self._sp_check_every.setEnabled(False)
        aml.addRow("Check every:", self._sp_check_every)
        self._conv_combo.currentTextChanged.connect(self._on_conv_changed)

        self._device_combo = NoWheelComboBox()
        self._device_combo.addItems(["auto", "cuda", "cpu"])
        aml.addRow("Device:", self._device_combo)

        advanced_layout.addWidget(method_adv_group)

        # --- 2D widefield expert ---
        wf2d_group = QGroupBox("2D Widefield Expert")
        wf2d_layout = QFormLayout()
        wf2d_group.setLayout(wf2d_layout)
        self._two_d_wf_group = wf2d_group

        self._two_d_wf_aggr_combo = NoWheelComboBox()
        self._two_d_wf_aggr_combo.addItems(
            ["Very Conservative", "Conservative", "Balanced", "Strong", "Very Strong"]
        )
        self._two_d_wf_aggr_combo.setCurrentText("Balanced")
        wf2d_layout.addRow("2D WF aggressiveness:", self._two_d_wf_aggr_combo)

        self._sp_two_d_wf_bg_radius = NoWheelDoubleSpinBox()
        self._sp_two_d_wf_bg_radius.setRange(0.05, 10.0)
        self._sp_two_d_wf_bg_radius.setDecimals(2)
        self._sp_two_d_wf_bg_radius.setSingleStep(0.05)
        self._sp_two_d_wf_bg_radius.setValue(0.50)
        wf2d_layout.addRow("Background estimator radius (um):", self._sp_two_d_wf_bg_radius)

        self._sp_two_d_wf_bg_scale = NoWheelDoubleSpinBox()
        self._sp_two_d_wf_bg_scale.setRange(0.10, 3.00)
        self._sp_two_d_wf_bg_scale.setDecimals(2)
        self._sp_two_d_wf_bg_scale.setSingleStep(0.05)
        self._sp_two_d_wf_bg_scale.setValue(1.00)
        wf2d_layout.addRow("Auto background scale:", self._sp_two_d_wf_bg_scale)

        advanced_layout.addWidget(wf2d_group)

        # --- Optics / PSF ---
        optics_group = QGroupBox("Optics / PSF")
        ol = QFormLayout()
        optics_group.setLayout(ol)

        self._sp_na = NoWheelDoubleSpinBox()
        self._sp_na.setRange(0.1, 2.0)
        self._sp_na.setDecimals(3)
        self._sp_na.setSingleStep(0.05)
        self._sp_na.setValue(1.4)
        ol.addRow("NA:", self._sp_na)

        self._le_emission = QLineEdit("520")
        self._le_emission.setToolTip(
            "Emission wavelength(s) in nm, comma-separated per channel."
        )
        ol.addRow("Emission (nm):", self._le_emission)

        self._sp_px_xy = NoWheelDoubleSpinBox()
        self._sp_px_xy.setRange(1.0, 10000.0)
        self._sp_px_xy.setDecimals(3)
        self._sp_px_xy.setSingleStep(1.0)
        self._sp_px_xy.setValue(65.0)
        ol.addRow("Pixel XY (nm):", self._sp_px_xy)

        self._sp_px_z = NoWheelDoubleSpinBox()
        self._sp_px_z.setRange(1.0, 50000.0)
        self._sp_px_z.setDecimals(3)
        self._sp_px_z.setSingleStep(10.0)
        self._sp_px_z.setValue(200.0)
        ol.addRow("Pixel Z (nm):", self._sp_px_z)

        self._micro_combo = NoWheelComboBox()
        self._micro_combo.addItems(["widefield", "confocal"])
        self._micro_combo.setCurrentText("confocal")
        self._micro_combo.currentTextChanged.connect(self._on_micro_changed)
        ol.addRow("Microscope:", self._micro_combo)

        self._le_excitation = QLineEdit("488")
        self._le_excitation.setToolTip(
            "Excitation wavelength(s) in nm, comma-separated per channel.\n"
            "Used only for confocal PSF generation."
        )
        self._le_excitation.setEnabled(True)
        ol.addRow("Excitation (nm):", self._le_excitation)

        self._le_pinhole_airy = QLineEdit(str(_DEFAULT_PINHOLE_AIRY_UNITS))
        self._le_pinhole_airy.setToolTip(
            "Confocal pinhole diameter(s) in Airy disk units, comma-separated per channel. "
            "0 uses the legacy point-detector model."
        )
        self._le_pinhole_na = QLineEdit("N/A")
        self._le_pinhole_na.setEnabled(False)
        self._pinhole_stack = QStackedWidget()
        self._pinhole_stack.addWidget(self._le_pinhole_airy)
        self._pinhole_stack.addWidget(self._le_pinhole_na)
        ol.addRow("Pinhole (AU):", self._pinhole_stack)

        ctrl_layout.addWidget(optics_group)

        # --- Refractive Indices ---
        ri_group = QGroupBox("Refractive Indices")
        rl = QFormLayout()
        ri_group.setLayout(rl)

        self._sp_ri_imm = NoWheelDoubleSpinBox()
        self._sp_ri_imm.setRange(1.0, 2.0)
        self._sp_ri_imm.setDecimals(4)
        self._sp_ri_imm.setSingleStep(0.001)
        self._sp_ri_imm.setValue(1.515)
        rl.addRow("RI immersion:", self._sp_ri_imm)

        # Embedding / mounting medium combo
        _MEDIUM_RI = {
            "water (1.333)": 1.333,
            "PBS (1.334)": 1.334,
            "culture medium (1.337)": 1.337,
            "vectashield (1.45)": 1.45,
            "prolong gold (1.47)": 1.47,
            "glycerol (1.474)": 1.474,
            "oil (1.515)": 1.515,
            "prolong glass (1.52)": 1.52,
        }
        self._medium_ri_map = _MEDIUM_RI
        self._medium_combo = NoWheelComboBox()
        self._medium_combo.addItems(list(_MEDIUM_RI.keys()))
        self._medium_combo.setCurrentText("prolong gold (1.47)")
        self._medium_combo.currentTextChanged.connect(self._on_medium_changed)
        rl.addRow("Emb. medium:", self._medium_combo)

        self._sp_ri_sample = NoWheelDoubleSpinBox()
        self._sp_ri_sample.setRange(1.0, 2.0)
        self._sp_ri_sample.setDecimals(4)
        self._sp_ri_sample.setSingleStep(0.001)
        self._sp_ri_sample.setValue(1.47)
        rl.addRow("RI sample:", self._sp_ri_sample)



        ctrl_layout.addWidget(ri_group)

        # --- Fit PSF controls ---
        fit_psf_group = QGroupBox("Fit PSF")
        fit_psf_layout = QFormLayout()
        fit_psf_group.setLayout(fit_psf_layout)

        self._sp_fit_niter = NoWheelSpinBox()
        self._sp_fit_niter.setRange(5, 200)
        self._sp_fit_niter.setValue(40)
        self._sp_fit_niter.setToolTip(
            "Number of RL iterations per trial.\n"
            "More iterations give a more reliable RI-sample result but take longer.\n"
            "Fewer than ~25 iterations may bias the result toward compact (high-RI) PSFs "
            "because narrower PSFs converge faster in early iterations."
        )
        fit_psf_layout.addRow("Fit iterations:", self._sp_fit_niter)

        self._btn_fit_psf = QPushButton("Fit PSF\u2026")
        self._btn_fit_psf.setToolTip(
            "Search for the best sample RI with a rough scan followed by a fine scan.\n"
            "Uses only the channels currently selected in the viewer.\n\n"
            "Note: use \u226525 iterations so that RI-sample sensitivity is accurate; "
            "fewer iterations bias toward compact PSFs (no spherical aberration)."
        )
        self._btn_fit_psf.setEnabled(False)
        self._btn_fit_psf.clicked.connect(self._on_fit_psf)
        fit_psf_layout.addRow(self._btn_fit_psf)

        ctrl_layout.addWidget(fit_psf_group)
        fit_psf_group.setVisible(self._fitpsf_available)
        cov_group = QGroupBox("Coverslip / Depth")
        cl = QFormLayout()
        cov_group.setLayout(cl)

        self._sp_tg = NoWheelDoubleSpinBox()
        self._sp_tg.setRange(0.0, 1e4)
        self._sp_tg.setDecimals(3)
        self._sp_tg.setSingleStep(0.001)
        self._sp_tg.setValue(170.000)
        cl.addRow("Coverslip thickness (um):", self._sp_tg)

        self._sp_tg0 = NoWheelDoubleSpinBox()
        self._sp_tg0.setRange(0.0, 1e4)
        self._sp_tg0.setDecimals(3)
        self._sp_tg0.setSingleStep(0.001)
        self._sp_tg0.setValue(170.000)
        cl.addRow("Coverslip thickness (design) (um):", self._sp_tg0)

        self._sp_ti0 = NoWheelDoubleSpinBox()
        self._sp_ti0.setRange(0.0, 1e4)
        self._sp_ti0.setDecimals(3)
        self._sp_ti0.setSingleStep(0.001)
        self._sp_ti0.setValue(100.000)
        cl.addRow("Immersion thickness (design) (um):", self._sp_ti0)

        self._sp_zp = NoWheelDoubleSpinBox()
        self._sp_zp.setRange(0.0, 1e4)
        self._sp_zp.setDecimals(3)
        self._sp_zp.setSingleStep(0.001)
        self._sp_zp.setValue(0)
        cl.addRow("Particle depth (z_p) (um):", self._sp_zp)

        advanced_layout.addWidget(cov_group)

        # --- PSF advanced ---
        psf_group = QGroupBox("PSF Advanced")
        pl = QFormLayout()
        psf_group.setLayout(pl)

        self._cb_integrate = QCheckBox()
        self._cb_integrate.setChecked(True)
        pl.addRow("Pixel integration:", self._cb_integrate)

        self._sp_subpixels = NoWheelSpinBox()
        self._sp_subpixels.setRange(1, 9)
        self._sp_subpixels.setValue(3)
        pl.addRow("Sub-pixels:", self._sp_subpixels)

        self._sp_n_pupil = NoWheelSpinBox()
        self._sp_n_pupil.setRange(33, 513)
        self._sp_n_pupil.setSingleStep(2)
        self._sp_n_pupil.setValue(129)
        pl.addRow("Pupil samples:", self._sp_n_pupil)

        advanced_layout.addWidget(psf_group)

        # --- Iteration movie (hidden unless launched with --movie) ---
        self._movie_group = QGroupBox("Iteration Movie")
        movie_layout = QFormLayout()
        self._movie_group.setLayout(movie_layout)

        self._cb_movie = QCheckBox()
        self._cb_movie.setChecked(False)
        movie_layout.addRow("Save MP4:", self._cb_movie)

        movie_path_row = QWidget()
        movie_path_layout = QHBoxLayout(movie_path_row)
        movie_path_layout.setContentsMargins(0, 0, 0, 0)
        movie_path_layout.setSpacing(6)
        self._le_movie_path = QLineEdit()
        self._le_movie_path.setPlaceholderText("Choose an MP4 output path")
        movie_path_layout.addWidget(self._le_movie_path, stretch=1)
        self._btn_movie_browse = QPushButton("Browse…")
        self._btn_movie_browse.clicked.connect(self._on_browse_movie_path)
        movie_path_layout.addWidget(self._btn_movie_browse)
        movie_layout.addRow("Output:", movie_path_row)

        self._sp_movie_fps = NoWheelSpinBox()
        self._sp_movie_fps.setRange(1, 60)
        self._sp_movie_fps.setValue(10)
        movie_layout.addRow("Framerate:", self._sp_movie_fps)

        self._cb_movie_hold = QCheckBox()
        self._cb_movie_hold.setChecked(False)
        self._cb_movie_hold.setToolTip("Hold the original and final frames for one second each.")
        movie_layout.addRow("Hold endpoints:", self._cb_movie_hold)

        self._cb_movie_gif = QCheckBox()
        self._cb_movie_gif.setChecked(False)
        self._cb_movie_gif.setToolTip("After the MP4 is written, also create a half-size animated GIF next to it.")
        movie_layout.addRow("Also GIF 1/2 size:", self._cb_movie_gif)

        self._movie_layout_combo = NoWheelComboBox()
        self._movie_layout_combo.addItems(["Standard", "Split-screen", "Standard + split-screen"])
        self._movie_layout_combo.setCurrentText("Standard")
        self._movie_layout_combo.setToolTip(
            "Standard renders the iteration view; split-screen compares original on the left with the current iteration on the right."
        )
        movie_layout.addRow("Movie view:", self._movie_layout_combo)

        self._movie_inset_combo = NoWheelComboBox()
        self._movie_inset_combo.addItems([
            "None",
            "Difference vs original",
            "Difference vs final",
            "Ratio vs original",
            "Ratio vs final",
            "Raw / estimated (PSF*deconv)",
            "Raw − estimated (PSF*deconv)",
        ])
        self._movie_inset_combo.setCurrentText("None")
        self._movie_inset_combo.setToolTip(
            "Add a small top-right panel showing the mismatch between raw and PSF*deconvolved. "
            "Both 'Raw / estimated' and 'Raw − estimated' use an inferno glow colormap "
            "(black = perfect match, yellow/white = large error) with a colorbar legend. "
            "'Raw / estimated' auto-normalises the log2 ratio each frame (always high-contrast). "
            "'Raw − estimated' uses the absolute difference fixed to p99 of the raw signal — "
            "brightness genuinely decreases toward black as the algorithm converges."
        )
        movie_layout.addRow("Mini-panel:", self._movie_inset_combo)

        self._le_movie_title = QLineEdit()
        self._le_movie_title.setPlaceholderText("Optional title overlay")
        movie_layout.addRow("Top text:", self._le_movie_title)

        self._cb_movie_info = QCheckBox()
        self._cb_movie_info.setChecked(False)
        self._cb_movie_info.setToolTip(
            "Overlay method, iteration, convergence, and rendered-frame image quality metrics."
        )
        movie_layout.addRow("Info overlay:", self._cb_movie_info)

        self._cb_movie_log_convergence = QCheckBox()
        self._cb_movie_log_convergence.setChecked(False)
        self._cb_movie_log_convergence.setToolTip("Plot the convergence curve on a log10 scale.")
        movie_layout.addRow("Log convergence:", self._cb_movie_log_convergence)

        quality_label = QLabel("High (H.264 CRF 16)")
        movie_layout.addRow("Quality:", quality_label)
        self._movie_group.setVisible(self._movie_available)
        advanced_layout.addWidget(self._movie_group)

        # --- DL Refinement parameters (hidden unless launched with --dlref) ---
        self._dl_group = QGroupBox("DL Refinement Parameters")
        dl_group_layout = QFormLayout()
        self._dl_group.setLayout(dl_group_layout)

        dl_model_widget = QWidget()
        self._dl_model_widget = dl_model_widget
        dl_model_row = QHBoxLayout(dl_model_widget)
        dl_model_row.setContentsMargins(0, 0, 0, 0)
        self._le_dl_model = QLineEdit("")
        self._le_dl_model.setPlaceholderText("optional final_model.pt")
        self._btn_dl_model = QPushButton("Browse")
        self._btn_dl_model.clicked.connect(self._on_browse_dl_model)
        dl_model_row.addWidget(self._le_dl_model)
        dl_model_row.addWidget(self._btn_dl_model)
        dl_group_layout.addRow("DL model:", dl_model_widget)

        self._sp_dl_z_context = NoWheelSpinBox()
        self._sp_dl_z_context.setRange(0, 8)
        self._sp_dl_z_context.setValue(2)
        dl_group_layout.addRow("DL z-context:", self._sp_dl_z_context)

        self._sp_dl_batch_size = NoWheelSpinBox()
        self._sp_dl_batch_size.setRange(1, 128)
        self._sp_dl_batch_size.setValue(8)
        dl_group_layout.addRow("DL batch size:", self._sp_dl_batch_size)

        self._cb_dl_mixed_precision = QCheckBox()
        self._cb_dl_mixed_precision.setChecked(True)
        dl_group_layout.addRow("DL mixed precision:", self._cb_dl_mixed_precision)

        self._sp_dl_residual_strength = QDoubleSpinBox()
        self._sp_dl_residual_strength.setRange(0.0, 2.0)
        self._sp_dl_residual_strength.setDecimals(2)
        self._sp_dl_residual_strength.setSingleStep(0.05)
        self._sp_dl_residual_strength.setValue(1.0)
        self._sp_dl_residual_strength.setToolTip(
            "Inference-only multiplier for the learned residual. Use 0.25-0.5 when the DL refinement is too aggressive."
        )
        dl_group_layout.addRow("DL residual strength:", self._sp_dl_residual_strength)

        self._dl_group.setVisible(self._dlref_available)
        advanced_layout.addWidget(self._dl_group)

        advanced_layout.addStretch()
        ctrl_layout.addWidget(advanced_section)

        _set_field_tooltip(
            ml,
            self._method_combo,
            "Choose the deconvolution algorithm. `ci_rl` is the standard Richardson-Lucy "
            "workflow, `ci_rl_dl` adds an experimental trained 2.5D residual refinement, "
            "`ci_rl_tv` adds edge-preserving TV regularization, and "
            "`ci_sparse_hessian` favors sparse filament-like structure with a different prior.",
        )
        _set_field_tooltip(
            ml,
            self._le_niter,
            "Maximum iteration count per channel. Higher values can recover more detail but "
            "also amplify noise and halos. You can enter a comma-separated list to use "
            "different counts per channel.",
        )
        _set_field_tooltip(
            ml,
            self._conv_combo,
            "`auto` stops early when improvement becomes small, while `fixed` always runs the "
            "full iteration count. `auto` is usually the safer starting point for RL methods.",
        )
        _set_field_tooltip(
            ml,
            self._sp_rel_thresh,
            "Relative improvement threshold used by `auto` convergence. Smaller values run "
            "longer before stopping; larger values stop earlier.",
        )

        _set_field_tooltip(
            aml,
            self._sp_tv_lambda,
            "Strength of total-variation regularization for `ci_rl_tv`. Increase it to suppress "
            "noise and ringing; reduce it if fine structure starts looking over-smoothed.",
        )
        _set_field_tooltip(
            aml,
            self._damping_combo,
            "Noise-gated damping for RL-family methods. `auto` picks a practical default, "
            "`manual` lets you tune the value directly, and `none` disables damping.",
        )
        _set_field_tooltip(
            aml,
            self._sp_damping,
            "Manual damping strength. Higher values make updates more conservative in noisy, "
            "near-background regions; lower values make RL more aggressive.",
        )
        _set_field_tooltip(
            aml,
            self._two_d_mode_combo,
            "How 2D widefield images are handled. `Auto` uses the widefield-aware collapsed-PSF "
            "model, while `Legacy 2D` keeps the older simpler behavior.",
        )
        _set_field_tooltip(
            aml,
            self._sp_sparse_weight,
            "Main regularization strength for `ci_sparse_hessian`. Increase it to enforce "
            "stronger structure priors; reduce it if the result becomes too constrained.",
        )
        _set_field_tooltip(
            aml,
            self._sp_sparse_reg,
            "Regularization balance for `ci_sparse_hessian`. Values closer to 1.0 generally "
            "make the sparse prior stronger and the result smoother.",
        )
        _set_field_tooltip(
            aml,
            self._bg_combo,
            "Background treatment during deconvolution. `auto` estimates a sensible floor from "
            "the image, while `manual` uses the value you provide below.",
        )
        _set_field_tooltip(
            aml,
            self._sp_bg_value,
            "Manual background intensity. This should approximate the residual camera/background "
            "offset that should remain unsharpened.",
        )
        _set_field_tooltip(
            aml,
            self._offset_combo,
            "Constant added before RL iterations to avoid unstable updates near zero intensity. "
            "`auto` is usually safest; `none` disables it entirely.",
        )
        _set_field_tooltip(
            aml,
            self._sp_offset,
            "Manual pre-deconvolution offset value. Larger offsets make RL less aggressive in "
            "dark regions, but too much can flatten weak signal.",
        )
        _set_field_tooltip(
            aml,
            self._sp_prefilter,
            "Optional Gaussian prefilter sigma in pixels. Use a small value when the raw data is "
            "very noisy and RL starts to chase pixel-level noise.",
        )
        _set_field_tooltip(
            aml,
            self._start_combo,
            "Initial estimate for iterative deconvolution. `auto` chooses from image statistics "
            "and microscope type; `flat` is robust, `_bgsub` modes subtract background first, "
            "and `hybrid` blends background-subtracted observed and smoothed structure.",
        )
        _set_field_tooltip(
            aml,
            self._sp_check_every,
            "How often convergence is evaluated when `Convergence` is set to `auto`. Lower "
            "values check more frequently but add a little overhead.",
        )
        _set_field_tooltip(
            aml,
            self._device_combo,
            "Processing device. `auto` chooses CUDA when available, otherwise CPU. Select "
            "`cpu` if you want deterministic fallback or your GPU is too small.",
        )

        _set_field_tooltip(
            wf2d_layout,
            self._two_d_wf_aggr_combo,
            "Controls how aggressively the 3D widefield PSF is collapsed for 2D RL. "
            "`Very Conservative` keeps weighting close to the focal plane, while `Very Strong` "
            "includes more out-of-focus blur and usually gives a more aggressive correction.",
        )
        _set_field_tooltip(
            wf2d_layout,
            self._sp_two_d_wf_bg_radius,
            "Spatial radius used for the local background estimator in 2D widefield auto mode. "
            "Use a larger radius for broad background haze and a smaller one for tight local background.",
        )
        _set_field_tooltip(
            wf2d_layout,
            self._sp_two_d_wf_bg_scale,
            "Multiplier applied to the automatically estimated 2D widefield background. Increase "
            "it if the result is too aggressive in the background; decrease it if haze remains.",
        )

        _set_field_tooltip(
            ol,
            self._sp_na,
            "Objective numerical aperture. This is one of the most important PSF parameters: a "
            "higher NA usually means a tighter PSF and higher achievable resolution.",
        )
        _set_field_tooltip(
            ol,
            self._le_emission,
            "Emission wavelength for each channel in nanometers, comma-separated if needed. "
            "Used directly in PSF generation, so accurate values matter.",
        )
        _set_field_tooltip(
            ol,
            self._sp_px_xy,
            "Lateral pixel size in nanometers. This sets the image sampling in X/Y and is used "
            "to scale the PSF correctly.",
        )
        _set_field_tooltip(
            ol,
            self._sp_px_z,
            "Axial spacing between planes in nanometers. Important for 3D deconvolution and for "
            "matching the PSF to the real Z sampling of the data.",
        )
        _set_field_tooltip(
            ol,
            self._micro_combo,
            "Microscope model used for PSF generation. `widefield` and `confocal` use different "
            "optical assumptions, and confocal also uses the excitation wavelength.",
        )
        _set_field_tooltip(
            ol,
            self._le_excitation,
            "Excitation wavelength for each channel in nanometers. Used for confocal PSF "
            "generation; ignored for widefield data.",
        )

        _set_field_tooltip(
            rl,
            self._sp_ri_imm,
            "Refractive index of the immersion medium on the objective side, for example oil or water.",
        )
        _set_field_tooltip(
            rl,
            self._medium_combo,
            "Convenience preset for the sample or mounting medium. Choosing a medium updates the "
            "`RI sample` field to a typical refractive index.",
        )
        _set_field_tooltip(
            rl,
            self._sp_ri_sample,
            "Refractive index of the sample or mounting medium around the fluorophores. Mismatch "
            "between sample, coverslip, and immersion media can noticeably change the PSF.",
        )

        _set_field_tooltip(
            cl,
            self._sp_tg,
            "Actual coverslip thickness in micrometers. Use the real coverslip value when known, "
            "especially if you are working far from the objective design conditions.",
        )
        _set_field_tooltip(
            cl,
            self._sp_tg0,
            "Design coverslip thickness expected by the optical system or objective. A mismatch "
            "between actual and design thickness contributes to spherical aberration.",
        )
        _set_field_tooltip(
            cl,
            self._sp_ti0,
            "Design immersion-path thickness used by the optical model. Usually leave this at the "
            "objective default unless you have a specific calibration reason to change it.",
        )
        _set_field_tooltip(
            cl,
            self._sp_zp,
            "Emitter depth below the coverslip in micrometers. Increase this when the structure of "
            "interest is deeper in the sample and depth-induced aberration should be modeled.",
        )

        _set_field_tooltip(
            pl,
            self._cb_integrate,
            "Integrate the PSF over the finite pixel area instead of sampling only at pixel "
            "centers. Usually improves realism a bit, at the cost of more PSF computation time.",
        )
        _set_field_tooltip(
            pl,
            self._sp_subpixels,
            "Subdivisions per pixel used when pixel integration is enabled. Higher values give a "
            "more accurate integrated PSF but increase computation cost.",
        )
        _set_field_tooltip(
            pl,
            self._sp_n_pupil,
            "Sampling density of the pupil function used during PSF generation. Higher values can "
            "improve PSF accuracy but are slower and use more memory.",
        )

        ctrl_layout.addStretch()

        # ---- Right: image viewer ----
        viewer = QWidget()
        vl = QVBoxLayout(viewer)
        vl.setContentsMargins(0, 0, 0, 0)
        self._viewer = DualViewerWidget()
        self._viewer.timepointChanged.connect(self._on_viewer_time_changed)
        self._viewer.logRequested.connect(self._open_log_dialog)
        vl.addWidget(self._viewer, stretch=1)
        splitter.addWidget(viewer)

        # --- Top toolbar: Open / Run / Save ---
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 4)

        btn_open = QPushButton("Open\u2026")
        btn_open.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        btn_open.clicked.connect(self._on_open)
        bottom.addWidget(btn_open)

        btn_open_zarr = QPushButton("Open Zarr\u2026")
        btn_open_zarr.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        btn_open_zarr.clicked.connect(self._on_open_zarr)
        bottom.addWidget(btn_open_zarr)

        btn_open_leica = QPushButton("Open Leica\u2026")
        btn_open_leica.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        btn_open_leica.clicked.connect(self._on_open_leica)
        bottom.addWidget(btn_open_leica)

        btn_open_omero = QPushButton("Open OMERO\u2026")
        btn_open_omero.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        btn_open_omero.clicked.connect(self._on_open_omero)
        bottom.addWidget(btn_open_omero)

        btn_batch = QPushButton("Batch\u2026")
        btn_batch.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        btn_batch.clicked.connect(self._on_batch_deconvolver)
        bottom.addWidget(btn_batch)

        middle_bar = QWidget()
        middle_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        middle_layout = QHBoxLayout(middle_bar)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(8)

        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(False)
        self._file_label.setToolTip("No file loaded")
        self._file_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        middle_layout.addWidget(self._file_label, stretch=1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        self._progress.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._progress.setMinimumWidth(180)
        middle_layout.addWidget(self._progress, stretch=2)

        bottom.addWidget(middle_bar, stretch=1)

        self._btn_run = QPushButton("Run Deconvolution")
        self._btn_run.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        bottom.addWidget(self._btn_run)

        self._btn_save = QPushButton("Save\u2026")
        self._btn_save.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        bottom.addWidget(self._btn_save)

        self._btn_save_view = QPushButton("Save View\u2026")
        self._btn_save_view.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._btn_save_view.setEnabled(False)
        self._btn_save_view.clicked.connect(self._on_save_view)
        bottom.addWidget(self._btn_save_view)

        self._btn_save_series = QPushButton("Save T-Series\u2026")
        self._btn_save_series.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._btn_save_series.setEnabled(False)
        self._btn_save_series.setVisible(False)
        self._btn_save_series.clicked.connect(self._on_save_t_series)
        bottom.addWidget(self._btn_save_series)

        # --- Settings buttons ---
        self._btn_restore = QPushButton("Restore")
        self._btn_restore.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._btn_restore.setToolTip("Restore parameter values from the previous run")
        self._btn_restore.setEnabled(LAST_SETTINGS_PATH.exists())
        self._btn_restore.clicked.connect(self._on_restore_settings)
        bottom.addWidget(self._btn_restore)

        btn_save_settings = QPushButton("Save Settings\u2026")
        btn_save_settings.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        btn_save_settings.setToolTip("Save current parameter values to a JSON file")
        btn_save_settings.clicked.connect(self._on_save_settings)
        bottom.addWidget(btn_save_settings)

        btn_load_settings = QPushButton("Load Settings\u2026")
        btn_load_settings.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        btn_load_settings.setToolTip("Load parameter values from a JSON file")
        btn_load_settings.clicked.connect(self._on_load_settings)
        bottom.addWidget(btn_load_settings)

        root.insertLayout(0, bottom)
        root.addWidget(splitter, stretch=1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # Resource monitor bar (permanent widget on the right side)
        self._monitor_bar = ResourceMonitorBar()
        self._status.addPermanentWidget(self._monitor_bar)

        # Splitter proportions
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Initial method state
        self._on_method_changed(self._method_combo.currentText())

    # -----------------------------------------------------------------------
    # Log window
    # -----------------------------------------------------------------------

    def _open_log_dialog(self) -> None:
        if self._log_dialog is None:
            self._log_dialog = _LogDialog(self)
            self._log_dialog.finished.connect(lambda _code: setattr(self, "_log_dialog", None))
            self._log_dialog.metricsToggled.connect(self._set_compute_image_metrics)
            self._log_dialog.chartsToggled.connect(self._set_live_charts)
            self._log_dialog.set_text("\n".join(self._log_lines))
            self._log_dialog.set_compute_metrics(self._compute_image_metrics)
            self._log_dialog.set_charts_enabled(self._live_charts_enabled)
            self._log_dialog.set_conv_log_scale(self._live_log_scale)
            self._log_dialog.set_running(self._log_running)
        self._log_dialog.show()
        self._log_dialog.raise_()
        self._log_dialog.activateWindow()

    def _reset_log(self, title: str) -> None:
        self._log_lines = []
        self._log_running = False
        self._last_final_iteration_payload = None
        self._last_live_update_s = 0.0
        if self._log_dialog is not None:
            self._log_dialog.set_running(False)
            self._log_dialog.set_text("")
            self._log_dialog.clear_charts()
        self._log("=" * 70)
        self._log(title)
        self._log("=" * 70)
        for line in _runtime_environment_lines():
            self._log(line)

    def _log(self, text: str) -> None:
        line = str(text)
        self._log_lines.append(line)
        print(line, flush=True)
        if self._log_dialog is not None:
            self._log_dialog.append_line(line)

    def _log_from_logging(self, text: str) -> None:
        line = str(text)
        self._log_lines.append(line)
        if self._log_dialog is not None:
            self._log_dialog.append_line(line)

    def _log_many(self, lines: list[str]) -> None:
        for line in lines:
            self._log(line)

    def _set_log_running(self, running: bool) -> None:
        self._log_running = bool(running)
        if self._log_dialog is not None:
            self._log_dialog.set_running(self._log_running)

    def _set_compute_image_metrics(self, enabled: bool) -> None:
        self._compute_image_metrics = bool(enabled)
        self._log(f"Image metrics: {'enabled' if self._compute_image_metrics else 'disabled'}")

    def _set_live_charts(self, enabled: bool) -> None:
        self._live_charts_enabled = bool(enabled)

    def _on_worker_progress(self, msg: str) -> None:
        self._status.showMessage(msg)
        self._log(msg)

    # -----------------------------------------------------------------------
    # Slots — control panel
    # -----------------------------------------------------------------------

    def _on_method_changed(self, text: str):
        is_rl_family = text in ("ci_rl", "ci_rl_tv", "ci_rl_dl")
        is_tv = text == "ci_rl_tv"
        is_sparse = text == "ci_sparse_hessian"
        self._sp_tv_lambda.setEnabled(is_tv)
        if not is_tv:
            self._sp_tv_lambda.setValue(0.0)
        else:
            if self._sp_tv_lambda.value() == 0.0:
                self._sp_tv_lambda.setValue(0.0001)
        for label, widget in (
            (self._tv_lambda_label, self._sp_tv_lambda),
            (self._damping_label, self._damping_combo),
            (self._damping_value_label, self._sp_damping),
            (self._two_d_mode_label, self._two_d_mode_combo),
            (self._sparse_weight_label, self._sp_sparse_weight),
            (self._sparse_reg_label, self._sp_sparse_reg),
        ):
            if label is not None:
                label.setVisible(False)
            widget.setVisible(False)

        if self._tv_lambda_label is not None:
            self._tv_lambda_label.setVisible(is_tv)
        self._sp_tv_lambda.setVisible(is_tv)

        if self._damping_label is not None:
            self._damping_label.setVisible(is_rl_family)
        self._damping_combo.setVisible(is_rl_family)
        if self._damping_value_label is not None:
            self._damping_value_label.setVisible(is_rl_family)
        self._sp_damping.setVisible(is_rl_family)
        if self._two_d_mode_label is not None:
            self._two_d_mode_label.setVisible(is_rl_family)
        self._two_d_mode_combo.setVisible(is_rl_family)
        self._two_d_wf_group.setVisible(is_rl_family)

        if self._sparse_weight_label is not None:
            self._sparse_weight_label.setVisible(is_sparse)
        self._sp_sparse_weight.setVisible(is_sparse)
        if self._sparse_reg_label is not None:
            self._sparse_reg_label.setVisible(is_sparse)
        self._sp_sparse_reg.setVisible(is_sparse)

        self._on_damping_changed(self._damping_combo.currentText())
        self._refresh_two_d_wf_expert_state()

    def _on_bg_changed(self, text: str):
        self._sp_bg_value.setEnabled(text == "manual")

    def _on_offset_changed(self, text: str):
        self._sp_offset.setEnabled(text == "manual")

    def _on_damping_changed(self, text: str):
        rl_family = self._method_combo.currentText() in ("ci_rl", "ci_rl_tv", "ci_rl_dl")
        self._sp_damping.setEnabled(rl_family and text == "manual")

    def _on_conv_changed(self, text: str):
        auto = text == "auto"
        self._sp_rel_thresh.setEnabled(auto)
        self._sp_check_every.setEnabled(auto)

    def _on_micro_changed(self, text: str):
        if text == "confocal":
            self._le_excitation.setEnabled(True)
            self._le_excitation.setText(self._excitation_saved)
            self._le_pinhole_airy.setText(self._pinhole_airy_saved)
            self._pinhole_stack.setCurrentWidget(self._le_pinhole_airy)
            self._le_niter.setText("50")
        else:
            current = self._le_excitation.text()
            if current != "N/A":
                self._excitation_saved = current
            self._le_excitation.setText("N/A")
            self._le_excitation.setEnabled(False)
            self._pinhole_airy_saved = self._le_pinhole_airy.text()
            self._pinhole_stack.setCurrentWidget(self._le_pinhole_na)
            self._le_niter.setText("80")
        self._refresh_two_d_wf_expert_state()

    def _refresh_two_d_wf_expert_state(self, _text: str = ""):
        rl_family = self._method_combo.currentText() in ("ci_rl", "ci_rl_tv", "ci_rl_dl")
        widefield = self._micro_combo.currentText() == "widefield"
        auto_mode = self._two_d_mode_combo.currentText() == "Auto"
        enabled = rl_family and widefield and auto_mode
        self._two_d_wf_group.setEnabled(enabled)

    def _on_medium_changed(self, text: str):
        """Set RI sample spinbox from embedding medium combo selection."""
        ri = self._medium_ri_map.get(text)
        if ri is not None:
            self._sp_ri_sample.setValue(ri)

    def _on_proj_changed(self, text: str):
        del text

    def _on_browse_dl_model(self):
        start_dir = str(Path(self._le_dl_model.text()).parent) if self._le_dl_model.text().strip() else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open ci_rl_dl Model",
            start_dir,
            "PyTorch checkpoints (*.pt *.pth);;All Files (*)",
        )
        if path:
            self._le_dl_model.setText(path)
            self._apply_dl_model_metadata(Path(path))

    def _apply_dl_model_metadata(self, model_path: Path):
        meta_path = model_path.with_suffix(".json")
        if not meta_path.exists():
            self._log(f"No ci_rl_dl metadata JSON found next to model: {meta_path.name}")
            return
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"Could not read ci_rl_dl metadata JSON: {exc}")
            return

        recommended = metadata.get("recommended_inference") or {}
        rl_kwargs = recommended.get("rl_kwargs") or {}
        dl_kwargs = recommended.get("dl_kwargs") or {}

        niter = recommended.get("iterations", rl_kwargs.get("niter"))
        if niter is not None:
            self._le_niter.setText(str(int(niter)))
        z_context = recommended.get("dl_z_context", dl_kwargs.get("z_radius"))
        if z_context is not None:
            self._sp_dl_z_context.setValue(int(z_context))
        batch_size = recommended.get("dl_batch_size", dl_kwargs.get("batch_size"))
        if batch_size is not None:
            self._sp_dl_batch_size.setValue(int(batch_size))
        mixed_precision = recommended.get("dl_mixed_precision", dl_kwargs.get("mixed_precision"))
        if mixed_precision is not None:
            self._cb_dl_mixed_precision.setChecked(bool(mixed_precision))
        start = rl_kwargs.get("start")
        if start:
            self._start_combo.setCurrentText(str(start))
        convergence = rl_kwargs.get("convergence")
        if convergence:
            self._conv_combo.setCurrentText(str(convergence))
        two_d_mode = rl_kwargs.get("two_d_mode")
        if str(two_d_mode).lower() == "legacy_2d":
            self._two_d_mode_combo.setCurrentText("Legacy 2D")
        elif str(two_d_mode).lower() == "auto":
            self._two_d_mode_combo.setCurrentText("Auto")

        best = metadata.get("best_epoch") or {}
        best_text = ""
        if best:
            best_text = f", best epoch={int(best.get('epoch', 0))}, val={float(best.get('val_loss', 0.0)):.5g}"
        self._log(f"Loaded ci_rl_dl model metadata from {meta_path.name}{best_text}")

    def _begin_progress(self, total: int, text: str) -> None:
        self._progress.setRange(0, max(int(total), 1))
        self._progress.setValue(0)
        self._progress.setFormat("%v / %m")
        self._progress.setVisible(True)
        self._status.showMessage(text)
        QApplication.processEvents()

    def _begin_busy_progress(self, text: str) -> None:
        self._progress.setRange(0, 0)
        self._progress.setVisible(True)
        self._status.showMessage(text)
        QApplication.processEvents()

    def _advance_progress(self, value: int, text: Optional[str] = None) -> None:
        if self._progress.maximum() > 0:
            self._progress.setValue(max(0, min(int(value), self._progress.maximum())))
        if text is not None:
            self._status.showMessage(text)
        QApplication.processEvents()

    def _end_progress(self) -> None:
        self._progress.setVisible(False)

    def _restore_last_browse_dirs(self) -> None:
        """Restore lightweight browse roots that should apply at startup."""
        try:
            with open(LAST_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self._last_leica_dir = _accessible_directory_or_home(
            data.get("last_leica_dir", self._last_leica_dir)
        )

    def _set_last_leica_dir(self, path_like: Any, *, persist: bool = False) -> None:
        self._last_leica_dir = _accessible_directory_or_home(path_like)
        if persist:
            self._save_last_settings()

    @staticmethod
    def _leica_dialog_current_root(dialog) -> Optional[Path]:
        root = getattr(dialog, "_current_root", None)
        if root:
            return Path(root)
        current_file = getattr(dialog, "_current_file", None)
        if current_file:
            current_path = Path(current_file)
            return current_path.parent if current_path.is_file() else current_path
        return None

    # -----------------------------------------------------------------------
    # File open
    # -----------------------------------------------------------------------

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            self._last_open_dir,
            "Images (*.ome.tiff *.ome.tif *.tiff *.tif *.nd2 *.czi);;All Files (*)",
        )
        if path:
            self._last_open_dir = str(Path(path).parent)
            self._do_load(path)

    def _on_open_zarr(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open OME-Zarr Folder", self._last_zarr_dir,
        )
        if path:
            self._last_zarr_dir = str(Path(path).parent)
            self._do_load(path)

    def _on_batch_deconvolver(self):
        if self._batch_dialog is None:
            self._batch_dialog = _BatchDeconvolverDialog(self, parent=self)
            self._batch_dialog.finished.connect(lambda _=None: setattr(self, "_batch_dialog", None))
        self._batch_dialog.show()
        self._batch_dialog.raise_()
        self._batch_dialog.activateWindow()

    def _on_open_leica(self):
        try:
            from leica_browser_qt import LeicaBrowserDialog
        except ImportError:
            QMessageBox.critical(
                self,
                "Leica Browser Missing",
                "leica-browser-qt is not installed.\n\n"
                "Install version 0.2.0 or newer to open Leica LIF, XLEF, and LOF data.",
            )
            return

        start_dir = _accessible_directory_or_home(self._last_leica_dir)
        self._last_leica_dir = start_dir
        dialog = LeicaBrowserDialog(
            roots=[start_dir],
            selection_mode="single",
            parent=self,
        )
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        browsed_root = self._leica_dialog_current_root(dialog)
        if browsed_root is not None:
            self._set_last_leica_dir(browsed_root, persist=True)
        if not accepted:
            return
        context = dialog.selected_context()
        if context is None:
            return
        container = Path(context.container_path)
        if browsed_root is None:
            self._set_last_leica_dir(container.parent if container.is_file() else container, persist=True)
        self._reset_log(f"CIDeconvolve GUI log — opening Leica image {context.name}")
        self._log(f"Opening Leica source: {context.container_path} :: {context.internal_path}")
        t_open = time.time()
        try:
            self._begin_busy_progress(f"Opening {context.name} …")
            source = _build_leica_source(context)
            self._apply_image_source(
                source,
                context.name,
                source_path=Path(context.container_path),
                source_factory=lambda _context=context: _build_leica_source(_context),
            )
            self._log(f"Open complete in {_format_duration(time.time() - t_open)}")
        except Exception as exc:
            self._log(f"Leica load failed: {exc}")
            QMessageBox.critical(self, "Leica Load Error", str(exc))
            self._status.showMessage("Leica load failed", 5000)
        finally:
            self._end_progress()

    def _do_load(self, path: str):
        self._reset_log(f"CIDeconvolve GUI log — opening {Path(path).name}")
        self._log(f"Opening source: {path}")
        t_open = time.time()
        try:
            self._begin_busy_progress(f"Opening {Path(path).name} …")
            source = _build_file_source(path)
            if source.metadata.get("hcs_plate_field"):
                self._log(
                    "HCS plate detected; using first field "
                    f"{source.metadata['hcs_plate_field']}"
                )
            self._apply_image_source(
                source,
                Path(path).name,
                source_path=Path(path),
                source_factory=lambda _path=str(path): _build_file_source(_path),
            )
            self._log(f"Open complete in {_format_duration(time.time() - t_open)}")
        except Exception as exc:
            self._log(f"Load failed: {exc}")
            QMessageBox.critical(self, "Load Error", str(exc))
            self._status.showMessage("Load failed", 5000)
        finally:
            self._end_progress()

    def _apply_image_source(
        self,
        source: _BaseTimepointSource,
        display_name: str,
        source_path: Optional[Path] = None,
        source_factory: Optional[Callable[[], _BaseTimepointSource]] = None,
    ):
        """Apply a lazy timepoint source to the UI (shared by file and OMERO open)."""
        self._input_source = source
        self._input_source_factory = source_factory
        self._input_channels = []
        self._loaded_timepoint = None
        self._metadata = dict(source.metadata)
        self._preview_outputs_by_t.clear()
        self._input_path = source_path

        # Populate UI from metadata
        meta = self._metadata
        from_file = meta.get("_from_file", set())

        def _bg(found: bool) -> str:
            """Stylesheet snippet: green if from metadata, orange if default."""
            if found:
                return "background-color: #c8e6c9; color: black;"   # soft green
            return "background-color: #ffe0b2; color: black;"       # soft orange

        if meta.get("na"):
            self._sp_na.setValue(float(meta["na"]))
        self._sp_na.setStyleSheet(_bg("na" in from_file))
        px_x = meta.get("pixel_size_x")
        if px_x:
            self._sp_px_xy.setValue(float(px_x) * 1000.0)  # µm → nm
        self._sp_px_xy.setStyleSheet(_bg("pixel_size_x" in from_file))
        px_z = meta.get("pixel_size_z")
        if px_z:
            self._sp_px_z.setValue(float(px_z) * 1000.0)
        self._sp_px_z.setStyleSheet(_bg("pixel_size_z" in from_file))
        ri = meta.get("refractive_index")
        if ri:
            self._sp_ri_imm.setValue(float(ri))
        self._sp_ri_imm.setStyleSheet(_bg("refractive_index" in from_file))
        micro = meta.get("microscope_type")
        if micro:
            idx = self._micro_combo.findText(micro)
            if idx >= 0:
                self._micro_combo.setCurrentIndex(idx)
            # Auto-set iterations based on microscope type
            if micro == "confocal":
                self._le_niter.setText("50")
            else:
                self._le_niter.setText("80")
        self._micro_combo.setStyleSheet(_bg("microscope_type" in from_file))

        # Per-channel wavelengths
        ch_info = meta.get("channels", [])
        if ch_info:
            self._le_emission.setText(
                _format_channel_values(ch_info, "emission_wavelength", 520.0)
            )
            ex_text = _format_channel_values(ch_info, "excitation_wavelength", 488.0)
            self._excitation_saved = ex_text
            if self._micro_combo.currentText() == "confocal":
                self._le_excitation.setText(ex_text)
                self._le_excitation.setEnabled(True)
            # (widefield: field stays N/A and disabled)
        self._le_emission.setStyleSheet(
            _bg("emission_wavelength" in from_file))
        self._le_excitation.setStyleSheet(
            _bg("excitation_wavelength" in from_file))
        if ch_info:
            pinhole_text = _format_channel_values(
                ch_info, "pinhole_airy_units", _DEFAULT_PINHOLE_AIRY_UNITS, digits=2
            )
            self._pinhole_airy_saved = pinhole_text
            if self._micro_combo.currentText() == "confocal":
                self._le_pinhole_airy.setText(pinhole_text)
        self._le_pinhole_airy.setStyleSheet(
            _bg("pinhole_airy_units" in from_file))
        self._le_pinhole_na.setStyleSheet(
            _bg("pinhole_airy_units" in from_file))

        sample_ri = meta.get("sample_refractive_index")
        if sample_ri:
            self._sp_ri_sample.setValue(float(sample_ri))
            self._sp_ri_sample.setStyleSheet(_bg("sample_refractive_index" in from_file))
        else:
            self._sp_ri_sample.setStyleSheet(
                "background-color: #ffe0e0; color: black;")  # pastel red

        self._viewer.set_input_data([], self._metadata)
        self._load_timepoint_into_viewer(int(self._metadata.get("default_t", 0)), force=True)
        self._log_many(_image_detail_lines(display_name, source_path, self._metadata, self._input_channels))
        size_t = self._metadata.get("size_t", 1)
        size_z = self._metadata.get("size_z", 1)
        size_y = self._metadata.get("size_y", "?")
        size_x = self._metadata.get("size_x", "?")
        source_is_pyramidal = bool(getattr(source, "is_pyramidal", False))
        if source_is_pyramidal:
            provider = getattr(source, "tile_provider", None)
            full_size = provider.full_size() if provider is not None else (size_y, size_x)
            n_levels = getattr(provider, "n_levels", "?")
            self._log(
                "Large OMERO pyramid detected: using streamed pyramid tiles for viewing "
                f"({full_size[0]}x{full_size[1]}, levels={n_levels})."
            )
            self._log(
                "Run Deconvolution will stream full-resolution tiles directly to OME-Zarr."
            )

        n_ch = int(self._metadata.get("size_c", 0))
        self._file_label.setText(display_name)
        self._file_label.setToolTip(
            f"{display_name}\n{n_ch} ch, T={size_t}, Z={size_z}, YX={size_y}×{size_x}"
        )
        self._btn_run.setEnabled(True)
        self._btn_fit_psf.setEnabled(not source_is_pyramidal)
        self._btn_save_series.setEnabled(self._viewer.has_time_axis() and not source_is_pyramidal)
        self._btn_save_series.setVisible(self._viewer.has_time_axis())
        self._btn_save.setEnabled(False)
        self._btn_save_view.setEnabled(False)
        self._sync_preview_buttons()
        if source_is_pyramidal:
            self._btn_fit_psf.setEnabled(False)
        self._viewer.refresh_view()
        self._set_default_movie_output_path(display_name, source_path)
        self._status.showMessage(f"Loaded {display_name}", 5000)

    def _default_movie_output_path(self, display_name: str, source_path: Optional[Path]) -> str:
        if source_path is not None:
            stem = source_path.stem
        else:
            stem = Path(str(display_name)).stem
        stem = _safe_filename_stem(stem)
        return str(_default_downloads_dir() / f"{stem}_iterations.mp4")

    def _set_default_movie_output_path(self, display_name: str, source_path: Optional[Path]) -> None:
        if not self._movie_available:
            return
        current = self._le_movie_path.text().strip()
        old_default = self._movie_default_path
        new_default = self._default_movie_output_path(display_name, source_path)
        if not current or current == old_default:
            self._le_movie_path.setText(new_default)
        self._movie_default_path = new_default

    # -----------------------------------------------------------------------
    # Open from OMERO
    # -----------------------------------------------------------------------

    def _on_open_omero(self):
        try:
            from omero_browser_qt import (
                LoginDialog,
                OmeroBrowserDialog,
                OmeroGateway,
            )
        except ImportError:
            QMessageBox.warning(
                self,
                "OMERO not available",
                "omero-browser-qt is not installed.\n\n"
                "Install with:\n  pip install \"omero-browser-qt[viewer]==0.2.5\"",
            )
            return

        if self._omero_gw is None:
            try:
                self._omero_gw = OmeroGateway()
            except RuntimeError:
                # PyQt6 singleton init order: QObject.__init__() must be
                # called before any attribute access.  Work around by
                # bootstrapping the QObject base on the raw singleton.
                from PyQt6.QtCore import QObject as _QObj
                inst = OmeroGateway._instance
                _QObj.__init__(inst)
                OmeroGateway.__init__(inst)
                self._omero_gw = inst
        gw = self._omero_gw

        if gw.is_connected() and not self._omero_session_is_reusable():
            gw.disconnect()
            self._omero_session_deadline = 0.0

        if not self._omero_session_is_reusable():
            dlg = LoginDialog(self, gateway=gw)
            self._configure_omero_login_dialog(dlg)
            if dlg.exec() != LoginDialog.DialogCode.Accepted:
                return
            self._refresh_omero_session_deadline()
        else:
            self._refresh_omero_session_deadline()

        browser = OmeroBrowserDialog(self, gateway=gw, multiselect=False)
        if browser.exec() != OmeroBrowserDialog.DialogCode.Accepted:
            return

        self._refresh_omero_session_deadline()

        images = browser.get_selected_images()
        if not images:
            return

        image = images[0]
        name = image.getName()
        self._reset_log(f"CIDeconvolve GUI log — opening OMERO image {name}")
        self._log(f"Opening OMERO source: {name}")
        t_open = time.time()
        try:
            self._begin_busy_progress(f"Opening {name} from OMERO …")
            source = _build_omero_source(image)
            self._apply_image_source(
                source,
                f"OMERO: {name}",
                source_factory=lambda _image=image: _build_omero_source(_image),
            )
            self._log(f"Open complete in {_format_duration(time.time() - t_open)}")
        except Exception as exc:
            self._log(f"OMERO load failed: {exc}")
            QMessageBox.critical(self, "OMERO Error", str(exc))
            self._status.showMessage("OMERO load failed", 5000)
        finally:
            self._end_progress()

    # -----------------------------------------------------------------------
    # Run deconvolution
    # -----------------------------------------------------------------------

    def _collect_params(self) -> dict:
        device_text = self._device_combo.currentText()
        device = None if device_text == "auto" else device_text

        bg_text = self._bg_combo.currentText()
        background: str | float = "auto"
        if bg_text == "manual":
            background = self._sp_bg_value.value()

        offset_text = self._offset_combo.currentText()
        offset: str | float = "auto"
        if offset_text == "none":
            offset = 0.0
        elif offset_text == "manual":
            offset = self._sp_offset.value()

        em_list = _parse_float_list(self._le_emission.text())
        ex_list = _parse_float_list(self._le_excitation.text())
        pinhole_list = _parse_float_list(self._le_pinhole_airy.text())
        if not pinhole_list:
            pinhole_list = [_DEFAULT_PINHOLE_AIRY_UNITS]

        niter_list = []
        for s in self._le_niter.text().split(","):
            s = s.strip()
            if s:
                try:
                    niter_list.append(max(1, int(s)))
                except ValueError:
                    pass
        if not niter_list:
            niter_list = [50]

        damping_text = self._damping_combo.currentText()
        damping: str | float = "auto"
        if damping_text == "none":
            damping = 0.0
        elif damping_text == "manual":
            damping = self._sp_damping.value()

        movie_enabled = (
            self._movie_available
            and hasattr(self, "_cb_movie")
            and self._cb_movie.isChecked()
        )
        movie = {
            "enabled": bool(movie_enabled),
        }
        if movie_enabled:
            movie = {
                "enabled": True,
                "path": self._le_movie_path.text().strip(),
                "fps": self._sp_movie_fps.value(),
                "hold_endpoints": self._cb_movie_hold.isChecked(),
                "create_gif": self._cb_movie_gif.isChecked(),
                "layout_mode": self._movie_layout_combo.currentText(),
                "difference_inset": self._movie_inset_combo.currentText(),
                "title_text": self._le_movie_title.text().strip(),
                "show_info_metrics": self._cb_movie_info.isChecked(),
                "convergence_log_scale": self._cb_movie_log_convergence.isChecked(),
                "render_state": self._viewer.movie_render_state(),
            }

        return {
            "method": self._method_combo.currentText(),
            "compute_metrics": self._compute_image_metrics,
            "movie": movie,
            "niter_list": niter_list,
            "tv_lambda": self._sp_tv_lambda.value(),
            "damping": damping,
            "two_d_mode": "auto" if self._two_d_mode_combo.currentText() == "Auto" else "legacy_2d",
            "two_d_wf_aggressiveness": self._two_d_wf_aggr_combo.currentText().strip().lower(),
            "two_d_wf_bg_radius_um": self._sp_two_d_wf_bg_radius.value(),
            "two_d_wf_bg_scale": self._sp_two_d_wf_bg_scale.value(),
            "dl_model_path": self._le_dl_model.text().strip(),
            "dl_z_context": self._sp_dl_z_context.value(),
            "dl_batch_size": self._sp_dl_batch_size.value(),
            "dl_mixed_precision": self._cb_dl_mixed_precision.isChecked(),
            "dl_residual_strength": self._sp_dl_residual_strength.value(),
            "offset": offset,
            "prefilter_sigma": self._sp_prefilter.value(),
            "start": self._start_combo.currentText(),
            "sparse_hessian_weight": self._sp_sparse_weight.value(),
            "sparse_hessian_reg": self._sp_sparse_reg.value(),
            "background": background,
            "convergence": self._conv_combo.currentText(),
            "rel_threshold": self._sp_rel_thresh.value(),
            "check_every": self._sp_check_every.value(),
            "device": device,
            "na": self._sp_na.value(),
            "emission_wavelengths": em_list,
            "excitation_wavelengths": ex_list,
            "pinhole_airy_units": pinhole_list,
            "pixel_size_xy_nm": self._sp_px_xy.value(),
            "pixel_size_z_nm": self._sp_px_z.value(),
            "ri_immersion": self._sp_ri_imm.value(),
            "ri_sample": self._sp_ri_sample.value(),
            "t_g": self._sp_tg.value() * 1000.0,
            "t_g0": self._sp_tg0.value() * 1000.0,
            "t_i0": self._sp_ti0.value() * 1000.0,
            "z_p": self._sp_zp.value() * 1000.0,
            "microscope_type": self._micro_combo.currentText(),
            "integrate_pixels": self._cb_integrate.isChecked(),
            "n_subpixels": self._sp_subpixels.value(),
            "n_pupil": self._sp_n_pupil.value(),
        }

    def _is_streamed_omero_pyramid_source(self) -> bool:
        return (
            isinstance(self._input_source, _OmeroTimepointSource)
            and bool(getattr(self._input_source, "is_pyramidal", False))
        )

    def _start_streaming_omero_deconvolution(self) -> None:
        if not isinstance(self._input_source, _OmeroTimepointSource):
            return
        params = self._collect_params()
        streaming_metadata = dict(self._metadata)
        streaming_metadata["pixel_size_x"] = float(params["pixel_size_xy_nm"]) / 1000.0
        streaming_metadata["pixel_size_y"] = float(params["pixel_size_xy_nm"]) / 1000.0
        streaming_metadata["pixel_size_z"] = float(params["pixel_size_z_nm"]) / 1000.0
        streaming_metadata["microscope_type"] = params.get("microscope_type")
        streaming_metadata["na"] = float(params["na"])
        streaming_metadata["refractive_index"] = float(params["ri_immersion"])
        streaming_metadata["sample_refractive_index"] = float(params["ri_sample"])
        streaming_metadata["cideconvolve_processing"] = {
            "method": params.get("method"),
            "iterations": list(params.get("niter_list") or []),
            "background": params.get("background"),
            "offset": params.get("offset"),
            "damping": params.get("damping"),
            "tv_lambda": params.get("tv_lambda"),
            "sparse_hessian_weight": params.get("sparse_hessian_weight"),
            "sparse_hessian_reg": params.get("sparse_hessian_reg"),
            "prefilter_sigma": params.get("prefilter_sigma"),
            "convergence": params.get("convergence"),
            "rel_threshold": params.get("rel_threshold"),
            "check_every": params.get("check_every"),
            "two_d_mode": params.get("two_d_mode"),
            "two_d_wf_aggressiveness": params.get("two_d_wf_aggressiveness"),
        }
        channels_meta = [
            dict(ch) if isinstance(ch, dict) else {}
            for ch in streaming_metadata.get("channels", [])
        ]
        render_state = self._viewer.movie_render_state()
        channel_colors = list(render_state.get("channel_colors") or [])
        active_channels = set(int(v) for v in render_state.get("active_channels") or [])
        size_c = max(int(streaming_metadata.get("size_c", len(channel_colors))), len(channel_colors))
        while len(channels_meta) < size_c:
            channels_meta.append({})
        names = list(streaming_metadata.get("channel_names") or [])
        for i in range(size_c):
            if i < len(channel_colors):
                channels_meta[i]["color"] = tuple(int(v) for v in channel_colors[i][:3])
            channels_meta[i]["active"] = i in active_channels if active_channels else bool(channels_meta[i].get("active", True))
            if "name" not in channels_meta[i] and i < len(names):
                channels_meta[i]["name"] = names[i]
        streaming_metadata["channels"] = channels_meta
        if params.get("method") == "ci_rl_dl":
            QMessageBox.warning(
                self,
                "Streaming not available",
                "Streaming ci_rl_dl is not enabled yet. Choose ci_rl, ci_rl_tv, or ci_sparse_hessian.",
            )
            return
        movie_params = params.get("movie") or {}
        if movie_params.get("enabled"):
            QMessageBox.warning(
                self,
                "Movie Export",
                "Iteration movies are disabled for full-resolution streamed OMERO jobs.",
            )
            return

        display_name = self._file_label.text().strip() or "omero_image"
        method = params.get("method", "ci_rl")
        niter_text = self._le_niter.text().strip().replace(", ", "-").replace(",", "-")
        suggested = Path(self._last_save_dir) / f"{_safe_filename_stem(display_name)}_{method}_{niter_text}i.ome.zarr"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Streamed OME-Zarr",
            str(suggested),
            "OME-Zarr (*.ome.zarr);;Zarr (*.zarr)",
        )
        if not path:
            return
        if not (path.lower().endswith(".ome.zarr") or path.lower().endswith(".zarr")):
            path += ".ome.zarr"
        output_path = Path(path)
        if output_path.exists() and not output_path.is_dir():
            QMessageBox.warning(self, "Output Error", "OME-Zarr output must be a directory path.")
            return

        self._last_save_dir = str(output_path.parent)
        self._save_last_settings()
        self._btn_run.setText("Stop")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #e53935; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self._btn_save.setEnabled(False)
        self._btn_save_view.setEnabled(False)
        self._btn_save_series.setEnabled(False)
        self._begin_busy_progress("Streaming OMERO deconvolution ...")
        self._monitor_bar.set_active(True)
        self._set_log_running(True)
        self._log("")
        self._log("=" * 70)
        self._log(f"Starting streamed OMERO deconvolution to {output_path}")
        self._log("=" * 70)
        self._worker = _StreamingOmeroWorker(
            self._input_source,
            streaming_metadata,
            params,
            str(output_path),
            parent=self,
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_deconv_done)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_run(self):
        # --- Stop mode: cancel running worker ---
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            _release_cuda_memory(synchronize=True)
            self._btn_run.setEnabled(False)
            self._log("Stop requested by user.")
            self._status.showMessage("Stopping …")
            return

        if self._save_worker is not None and self._save_worker.isRunning():
            QMessageBox.information(
                self,
                "Save in progress",
                "A full T-series export is currently running. Please wait for it to finish first.",
            )
            return

        if not self._input_channels:
            return
        if self._is_streamed_omero_pyramid_source():
            self._start_streaming_omero_deconvolution()
            return

        self._btn_run.setText("Stop")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #e53935; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self._btn_save.setEnabled(False)
        self._btn_save_view.setEnabled(False)
        self._btn_save_series.setEnabled(False)
        self._begin_busy_progress("Running deconvolution …")

        # Signal that deconvolution is active (dot indicator)
        self._monitor_bar.set_active(True)

        # Auto-save settings before each run
        self._save_last_settings()

        current_t = self._viewer.current_timepoint()
        params = self._collect_params()
        movie_params = params.get("movie") or {}
        if movie_params.get("enabled"):
            movie_path = str(movie_params.get("path") or "").strip()
            if not movie_path:
                self._end_progress()
                self._monitor_bar.set_active(False)
                self._btn_run.setText("Run Deconvolution")
                self._btn_run.setStyleSheet(
                    "QPushButton { background-color: #4CAF50; color: white; "
                    "font-weight: bold; padding: 8px; }"
                )
                self._btn_run.setEnabled(True)
                self._btn_save_series.setEnabled(bool(self._input_channels) and self._viewer.has_time_axis())
                QMessageBox.warning(self, "Movie Export", "Choose an MP4 output path before running.")
                return
            try:
                import imageio.v2  # noqa: F401
                import imageio_ffmpeg  # noqa: F401
            except ImportError:
                self._end_progress()
                self._monitor_bar.set_active(False)
                self._btn_run.setText("Run Deconvolution")
                self._btn_run.setStyleSheet(
                    "QPushButton { background-color: #4CAF50; color: white; "
                    "font-weight: bold; padding: 8px; }"
                )
                self._btn_run.setEnabled(True)
                self._btn_save_series.setEnabled(bool(self._input_channels) and self._viewer.has_time_axis())
                QMessageBox.warning(
                    self,
                    "Movie Export",
                    "Movie export requires imageio and imageio-ffmpeg.\n\n"
                    "Install the GUI requirements again, or run:\n"
                    "  pip install \"imageio[ffmpeg]\"",
                )
                return
        # Clear charts and live-update state for a fresh run
        self._last_final_iteration_payload = None
        self._last_live_update_s = 0.0
        if self._log_dialog is not None:
            self._log_dialog.clear_charts()
        self._set_log_running(True)
        self._log("")
        self._log("=" * 70)
        self._log(f"Starting deconvolution preview for T={current_t + 1}")
        self._log("=" * 70)
        self._worker = _DeconvolveWorker(
            self._input_channels, self._metadata, params, current_t, parent=self
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_deconv_done)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.liveUpdate.connect(self._on_live_iteration_update)
        movie_params = params.get("movie") or {}
        self._live_log_scale = bool(movie_params.get("convergence_log_scale", False))
        self._worker.start()

    def _on_deconv_done(self, result):
        self._monitor_bar.set_active(False)
        self._set_log_running(False)
        _release_cuda_memory(synchronize=True)

        self._end_progress()
        self._btn_run.setText("Run Deconvolution")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self._btn_run.setEnabled(True)
        self._btn_save_series.setEnabled(
            bool(self._input_channels)
            and self._viewer.has_time_axis()
            and not self._is_streamed_omero_pyramid_source()
        )

        if isinstance(result, Exception):
            self._worker = None
            msg = str(result)
            if "Stopped by user" in msg:
                self._log("Deconvolution stopped by user.")
                self._status.showMessage("Deconvolution stopped", 5000)
            else:
                self._log(f"Deconvolution failed: {msg}")
                QMessageBox.critical(self, "Deconvolution Error", msg)
                self._status.showMessage("Deconvolution failed", 5000)
            return

        try:
            if isinstance(result, dict) and result.get("streaming_output"):
                summary = result.get("summary") or {}
                output_path = str(result.get("streaming_output"))
                provenance = str(result.get("provenance") or "")
                self._log(
                    "Streaming deconvolution complete: "
                    f"{summary.get('tiles_completed', '?')}/{summary.get('tiles_total', '?')} tiles written."
                )
                self._log(f"OME-Zarr output: {output_path}")
                if provenance:
                    self._log(f"Provenance: {provenance}")
                self._status.showMessage(f"Streaming deconvolution complete → {Path(output_path).name}", 8000)
                return

            timepoint = int(result["timepoint"])
            self._preview_outputs_by_t[timepoint] = result["channels"]
            self._viewer.set_preview_result(timepoint, result["channels"])
            self._sync_preview_buttons()
            self._log("Deconvolution complete.")
            self._status.showMessage(f"Deconvolution complete for T={timepoint + 1}", 5000)
            # Push final residual to live chart panel if enabled
            if self._log_dialog is not None and self._log_dialog.is_charts_enabled():
                payload = self._last_final_iteration_payload
                if payload is not None:
                    estimated = payload.get("estimated")
                    ch_idx = int(payload.get("channel_index", 0))
                    if (
                        estimated is not None
                        and self._input_channels
                        and ch_idx < len(self._input_channels)
                    ):
                        raw_arr = np.asarray(self._input_channels[ch_idx], dtype=np.float32)
                        est_arr = np.asarray(estimated, dtype=np.float32)

                        def _central_z(arr: np.ndarray) -> np.ndarray:
                            if arr.ndim == 2:
                                return arr
                            if arr.ndim == 3:
                                return arr[arr.shape[0] // 2]
                            if arr.ndim == 4:
                                return arr[0, arr.shape[1] // 2]
                            return arr.reshape(arr.shape[-2], arr.shape[-1])

                        self._log_dialog.push_final_residual(
                            _central_z(raw_arr), _central_z(est_arr)
                        )
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Viewer Error", str(exc))
        finally:
            self._worker = None

    # -----------------------------------------------------------------------
    # Live chart updates from worker thread
    # -----------------------------------------------------------------------

    def _on_live_iteration_update(self, payload: dict) -> None:
        """Throttled handler for live iteration updates; runs on the main thread."""
        if self._log_dialog is None or not self._log_dialog.is_charts_enabled():
            return
        is_final = bool(payload.get("is_final", False))
        if is_final:
            self._last_final_iteration_payload = payload
        now = time.monotonic()
        if is_final or (now - self._last_live_update_s) >= 0.25:
            self._last_live_update_s = now
            self._log_dialog.push_iteration(payload)

    # -----------------------------------------------------------------------
    # PSF fitting
    # -----------------------------------------------------------------------

    def _on_fit_psf(self):
        if not self._input_channels:
            self._log("No image loaded — cannot fit PSF.")
            return
        if self._is_streamed_omero_pyramid_source():
            QMessageBox.information(
                self,
                "Large OMERO image",
                "PSF fitting is disabled for OMERO pyramid overview data. "
                "Use a small crop/ROI or a non-pyramidal source for fitting.",
            )
            return
        if self._fit_psf_worker is not None and self._fit_psf_worker.isRunning():
            self._fit_psf_worker.requestInterruption()
            self._btn_fit_psf.setEnabled(False)
            self._btn_fit_psf.setText("Stopping…")
            self._log("PSF fit stop requested by user.")
            self._status.showMessage("Stopping PSF fit …", 0)
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Deconvolution running",
                "Please wait for the current deconvolution to finish before fitting PSF.",
            )
            return

        # Determine which channels to use — RI fitting requires exactly ONE channel.
        active_indices = self._viewer._active_channel_indices() if hasattr(self._viewer, "_active_channel_indices") else []
        if not active_indices:
            active_indices = list(range(len(self._input_channels)))
        if len(active_indices) > 1:
            QMessageBox.warning(
                self,
                "Select One Channel",
                "RI fitting is performed on a single channel.\n\n"
                "Please enable exactly one channel in the viewer, then try again.",
            )
            return
        channels = [self._input_channels[i] for i in active_indices if i < len(self._input_channels)]
        if not channels:
            self._log("No channels available for PSF fitting.")
            return

        params = self._collect_params()

        # Build per-channel emission/excitation lists for the selected channels
        em_all = params.get("emission_wavelengths", [520]) or [520]
        ex_all = params.get("excitation_wavelengths", [488]) or [488]
        ph_all = params.get("pinhole_airy_units", [1.0]) or [1.0]
        em_sel = [em_all[i] if i < len(em_all) else em_all[-1] for i in active_indices]
        ex_sel = [ex_all[i] if i < len(ex_all) else ex_all[-1] for i in active_indices]
        ph_sel = [ph_all[i] if i < len(ph_all) else ph_all[-1] for i in active_indices]

        fit_params = dict(params)
        fit_params["emission_wavelengths"] = em_sel
        fit_params["excitation_wavelengths"] = ex_sel
        fit_params["pinhole_airy_units"] = ph_sel

        is_2d = channels[0].ndim == 2 if channels else False

        guessed_z_p = None
        if not is_2d and float(params.get("z_p", 0.0)) <= 0.0:
            guessed_z_p = _guess_fit_zp_um(channels, params.get("pixel_size_z_nm", 0.0))
            if guessed_z_p is not None:
                guessed_z_p_um, _, _ = guessed_z_p
                fit_params["z_p"] = guessed_z_p_um * 1000.0
                self._sp_zp.setValue(guessed_z_p_um)

        self._btn_fit_psf.setEnabled(True)
        self._btn_fit_psf.setText("Stop Fit PSF")
        self._btn_fit_psf.setStyleSheet(
            "QPushButton { background-color: #e53935; color: white; "
            "font-weight: bold; }"
        )
        self._btn_run.setEnabled(False)
        self._sp_fit_niter.setEnabled(False)
        self._fit_psf_started_at = time.time()
        niter_fit = self._sp_fit_niter.value()
        self._log("")
        self._log("PSF Fitting")
        n_ch = len(channels)
        microscope = params.get("microscope_type", "widefield")
        self._log(f"  Channels    : {n_ch} (indices {active_indices})")
        self._log(f"  Microscope  : {microscope}")
        self._log(f"  RL iters/trial: {niter_fit}")
        self._log(f"  RI sample   : {params['ri_sample']:.4f}")
        if microscope == "confocal":
            self._log(f"  Pinholes    : {ph_sel} (held fixed during fitting)")
        self._log("  RI scan     : rough full-range scan + fine local refinement")
        # Warn if image is 2D: RI mismatch mainly affects axial PSF (spherical aberration),
        # so the RI axis of the heatmap will be near-flat for single Z-plane data.
        if is_2d:
            self._log("  Note: 2D image — RI sample has minimal effect on lateral PSF (spherical aberration is axial).")
        if guessed_z_p is not None:
            guessed_z_p_um, nz_guess, stack_depth_um = guessed_z_p
            self._log(
                f"  z_p         : auto {guessed_z_p_um:.3f} um from stack depth "
                f"({nz_guess} planes, {stack_depth_um:.3f} um total)"
            )
        elif not is_2d and float(params.get("z_p", 0.0)) <= 0.0:
            self._log("  Warning: Particle depth (z_p) is 0 um, and stack depth could not be inferred for automatic RI fitting.")
        if niter_fit < 25:
            self._log(
                f"  Warning: {niter_fit} iterations may be too few to discriminate RI sample — "
                "narrow PSFs converge faster in early iterations, biasing toward RI=1.515. "
                "Recommend \u226525 iterations."
            )
        self._log("  Searching\u2026")

        self._fit_psf_worker = _FitPsfWorker(channels, fit_params, niter_fit=niter_fit, parent=self)
        self._fit_psf_worker.progress.connect(self._on_fit_psf_progress)
        self._fit_psf_worker.finished.connect(self._on_fit_psf_done)
        self._fit_psf_worker.start()

    def _on_fit_psf_progress(self, idx: int, total: int, params: dict, i_div: float):
        if total > 0:
            pct = int(idx * 100 / total)
            ri = params.get("ri_sample", "?")
            eta_text = ""
            if self._fit_psf_started_at is not None and idx > 0:
                elapsed = max(time.time() - self._fit_psf_started_at, 0.0)
                seconds_per_trial = elapsed / idx
                eta_seconds = max(total - idx, 0) * seconds_per_trial
                eta_text = f"  ETA {_format_duration(eta_seconds)}"
            self._status.showMessage(
                f"Fitting PSF… {pct}%{eta_text}  (RI={ri:.4f}, residual={i_div:.5g})"
                if isinstance(ri, float) else
                f"Fitting PSF… {pct}%{eta_text}  residual={i_div:.5g}",
                0,
            )

    def _on_fit_psf_done(self, result):
        self._btn_fit_psf.setText("Fit PSF\u2026")
        self._btn_fit_psf.setStyleSheet("")
        self._btn_fit_psf.setEnabled(bool(self._input_channels))
        self._btn_run.setEnabled(bool(self._input_channels))
        self._sp_fit_niter.setEnabled(True)
        self._fit_psf_started_at = None

        if isinstance(result, Exception):
            self._fit_psf_worker = None
            if "Stopped by user" in str(result):
                self._log("PSF fitting stopped by user.")
                self._status.showMessage("PSF fitting stopped", 5000)
                return
            self._log(f"PSF fitting failed: {result}")
            QMessageBox.critical(self, "PSF Fit Error", str(result))
            self._status.showMessage("PSF fitting failed", 5000)
            return

        try:
            best = result["best_params"]
            grid_best = result.get("grid_best_params", best)
            improvement = result.get("improvement_pct", 0.0)
            score_label = result.get("score_label", "PSF fit score")
            best_ri = float(best.get("ri_sample", 1.33))
            grid_best_ri = float(grid_best.get("ri_sample", best_ri))

            self._log("")
            self._log(f"  Best score        : {result['best_i_div']:.5g}")
            baseline = result.get("baseline_i_div", float("inf"))
            self._log(f"  Baseline score    : {baseline:.5g}" if baseline != float("inf") else "  Baseline score    : (not available)")
            self._log(f"  Score model       : {score_label}")

            if improvement <= 0.0:
                msg = (
                    f"PSF fit: no improvement found. "
                    f"Original parameters kept. "
                    f"(grid best RI={grid_best_ri:.4f})"
                )
                self._log(msg)
                self._status.showMessage(msg, 8000)
            else:
                # Apply best params to UI
                self._sp_ri_sample.setValue(best_ri)
                msg = (
                    f"PSF fit complete \u2014 {improvement:.0f}% improvement "
                    f"(RI={best_ri:.4f})"
                )
                self._log(msg)
                self._status.showMessage(msg, 8000)

            # Always render heatmap so user can see the full search landscape
            self._render_psf_fit_heatmap(result)

        except Exception as exc:
            traceback.print_exc()
            self._log(f"PSF fit result display error: {exc}")
        finally:
            self._fit_psf_worker = None

    def _render_psf_fit_heatmap(self, result: dict):
        """Render a PSF fitting heatmap and display it in the decon viewer pane."""
        try:
            import matplotlib.figure as mfig
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            search_log = result.get("search_log", [])
            grid = result.get("grid", {})
            pinhole_vals = grid.get("pinhole", [])
            ri_vals = grid.get("ri_sample", [])
            is_confocal = len(pinhole_vals) > 1

            if not search_log:
                return

            if is_confocal:
                n_ri = len(ri_vals)
                n_ph = len(pinhole_vals)
                matrix = np.full((n_ri, n_ph), np.nan)
                ph_idx = {round(v, 6): i for i, v in enumerate(pinhole_vals)}
                ri_idx = {round(v, 6): i for i, v in enumerate(ri_vals)}
                for entry in search_log:
                    ph = round(float(entry["params"].get("pinhole_airy_units", -1)), 6)
                    ri = round(float(entry["params"].get("ri_sample", -1)), 6)
                    pi = ph_idx.get(ph)
                    ri_i = ri_idx.get(ri)
                    if pi is not None and ri_i is not None:
                        matrix[ri_i, pi] = entry["i_div"]

                fig = mfig.Figure(figsize=(max(5.0, n_ph * 0.8), max(3.5, n_ri * 0.8)), dpi=100)
                FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)
                im = ax.imshow(
                    matrix,
                    aspect="auto",
                    cmap="inferno_r",
                    origin="lower",
                    interpolation="nearest",
                )
                ax.set_xticks(range(n_ph))
                ax.set_xticklabels([f"{v:.2f}" for v in pinhole_vals], fontsize=8)
                ax.set_yticks(range(n_ri))
                ax.set_yticklabels([f"{v:.4f}" for v in ri_vals], fontsize=8)
                ax.set_xlabel("Pinhole (AU)", fontsize=9)
                ax.set_ylabel("RI sample", fontsize=9)
                ax.set_title("PSF Fit — composite score (lower = better)", fontsize=9)

                # Mark best
                best = result.get("best_params", {})
                best_ph = round(float(best.get("pinhole_airy_units", -1)), 6)
                best_ri = round(float(best.get("ri_sample", -1)), 6)
                bpi = ph_idx.get(best_ph)
                bri = ri_idx.get(best_ri)
                if bpi is not None and bri is not None:
                    ax.plot(bpi, bri, "w*", markersize=14, zorder=5)

                fig.colorbar(im, ax=ax, label="Composite score")
                fig.tight_layout()

            else:
                # Widefield: 1D bar chart over RI
                ri_div = [(e["params"].get("ri_sample", 0), e["i_div"]) for e in search_log]
                ri_div.sort(key=lambda x: x[0])
                xs = [v[0] for v in ri_div]
                ys = [v[1] for v in ri_div]

                fig = mfig.Figure(figsize=(5.5, 3.5), dpi=100)
                FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)
                ax.bar(range(len(xs)), ys, color="#ff6600", edgecolor="white", alpha=0.85)
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels([f"{v:.4f}" for v in xs], fontsize=8, rotation=45, ha="right")
                ax.set_xlabel("RI sample", fontsize=9)
                ax.set_ylabel("Composite score", fontsize=9)
                ax.set_title("PSF Fit — composite score by RI sample (lower = better)", fontsize=9)

                # Mark best
                best_ri = float(result.get("best_params", {}).get("ri_sample", -1))
                for i, x in enumerate(xs):
                    if abs(x - best_ri) < 1e-5:
                        ax.bar(i, ys[i], color="#ffdd00", edgecolor="white", alpha=1.0)
                        ax.annotate("best", (i, ys[i]), textcoords="offset points",
                                    xytext=(0, 4), ha="center", fontsize=8, color="white",
                                    fontweight="bold")
                fig.tight_layout()

            # Render to RGBA buffer via the Agg canvas
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            w, h = fig.canvas.get_width_height()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            rgb = arr[:, :, :3].copy()
            fig.clf()

            # Convert to QPixmap
            h_px, w_px, _ = rgb.shape
            qimg = QImage(rgb.data, w_px, h_px, w_px * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            self._viewer._output_pane.set_pixmap(pixmap)
            self._viewer._output_pane.show_content()

        except Exception as exc:
            self._log(f"Heatmap render error: {exc}")
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def _on_save(self):
        current_t = self._viewer.current_timepoint()
        preview_channels = self._preview_outputs_by_t.get(current_t)
        if not preview_channels:
            return

        stem = self._input_path.stem if self._input_path else "deconvolved"
        method = self._method_combo.currentText()
        niter_text = self._le_niter.text().strip().replace(", ", "-").replace(",", "-")
        suggested = f"{stem}_{method}_{niter_text}i_T{current_t:03d}.ome.tiff"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Deconvolved Preview",
            str(Path(self._last_save_dir) / suggested),
            "OME-TIFF (*.ome.tiff);;TIFF (*.tiff *.tif)",
        )
        if not path:
            return
        self._last_save_dir = str(Path(path).parent)

        try:
            t_save = time.time()
            self._log(f"Saving preview to {path}")
            stack = np.stack(preview_channels, axis=0)
            data = stack[np.newaxis, ...].astype(np.float32)
            _write_ome_tiff(data, path, self._metadata)
            size = Path(path).stat().st_size / (1024 * 1024) if Path(path).exists() else 0.0
            self._log(f"Saved preview in {_format_duration(time.time() - t_save)} ({_format_bytes(size)})")
            self._status.showMessage(f"Saved → {Path(path).name}", 5000)
        except Exception as exc:
            self._log(f"Save failed: {exc}")
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_save_view(self):
        current_t = self._viewer.current_timepoint()
        if current_t not in self._preview_outputs_by_t:
            QMessageBox.information(
                self,
                "No deconvolved view",
                "Run deconvolution for the current timepoint before saving both viewer panes.",
            )
            return

        stem = _safe_filename_stem(self._input_path.stem if self._input_path else self._file_label.text())
        if self._viewer.has_time_axis():
            stem = f"{stem}_T{current_t:03d}"
        suggested = f"{stem}.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Original View PNG",
            str(Path(self._last_save_dir) / suggested),
            "PNG files (*.png);;All files (*)",
        )
        if not path:
            return

        original_path = Path(path)
        if original_path.suffix.lower() != ".png":
            original_path = original_path.with_suffix(".png")
        method_suffix = _safe_filename_stem(self._method_combo.currentText()) or "decon"
        decon_path = original_path.with_name(f"{original_path.stem}_{method_suffix}.png")
        self._last_save_dir = str(original_path.parent)

        try:
            QApplication.processEvents()
            images = self._viewer.current_view_images()
            original = images.get("original")
            deconvolved = images.get("deconvolved")
            if original is None or original.isNull():
                raise RuntimeError("Could not render the current original viewer pane.")
            if deconvolved is None or deconvolved.isNull():
                raise RuntimeError("Could not render the current deconvolved viewer pane.")
            if not original.save(str(original_path), "PNG"):
                raise RuntimeError(f"Could not write {original_path}")
            if not deconvolved.save(str(decon_path), "PNG"):
                raise RuntimeError(f"Could not write {decon_path}")
            self._log(f"Saved view PNGs: {original_path} and {decon_path}")
            self._status.showMessage(f"Saved view PNGs → {original_path.name}, {decon_path.name}", 5000)
        except Exception as exc:
            self._log(f"Save view failed: {exc}")
            QMessageBox.critical(self, "Save View Error", str(exc))

    def _on_save_t_series(self):
        if not self._input_channels:
            return
        if self._is_streamed_omero_pyramid_source():
            QMessageBox.information(
                self,
                "Large OMERO image",
                "Use Run Deconvolution to stream this OMERO pyramid to OME-Zarr.",
            )
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Preview running",
                "Stop the current preview deconvolution before starting a full T-series export.",
            )
            return
        if self._save_worker is not None and self._save_worker.isRunning():
            return

        stem = self._input_path.stem if self._input_path else "deconvolved"
        method = self._method_combo.currentText()
        niter_text = self._le_niter.text().strip().replace(", ", "-").replace(",", "-")
        suggested = f"{stem}_{method}_{niter_text}i_TSERIES.ome.tiff"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Deconvolved T-Series",
            str(Path(self._last_save_dir) / suggested),
            "OME-TIFF (*.ome.tiff);;TIFF (*.tiff *.tif)",
        )
        if not path:
            return
        self._last_save_dir = str(Path(path).parent)

        self._save_last_settings()
        self._begin_busy_progress("Saving full T-series …")
        self._btn_run.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_save_view.setEnabled(False)
        self._btn_save_series.setEnabled(False)
        self._monitor_bar.set_active(True)
        self._set_log_running(True)
        self._log("")
        self._log("=" * 70)
        self._log(f"Starting full T-series export to {path}")
        self._log("=" * 70)

        if self._input_source_factory is None:
            QMessageBox.warning(self, "Save Error", "No image source is available for T-series export.")
            self._monitor_bar.set_active(False)
            self._set_log_running(False)
            self._end_progress()
            self._btn_run.setEnabled(bool(self._input_channels))
            self._sync_preview_buttons()
            self._btn_save_series.setEnabled(bool(self._input_channels) and self._viewer.has_time_axis())
            return

        params = self._collect_params()
        try:
            save_source = self._input_source_factory()
        except Exception as exc:
            self._monitor_bar.set_active(False)
            self._set_log_running(False)
            self._end_progress()
            self._btn_run.setEnabled(bool(self._input_channels))
            self._sync_preview_buttons()
            self._btn_save_series.setEnabled(bool(self._input_channels) and self._viewer.has_time_axis())
            QMessageBox.critical(self, "Save Error", str(exc))
            return

        self._save_worker = _SaveTSeriesWorker(
            save_source,
            self._metadata,
            params,
            path,
            parent=self,
        )
        self._save_worker.progress.connect(self._on_worker_progress)
        self._save_worker.finished.connect(self._on_save_t_series_done)
        self._save_worker.finished.connect(self._save_worker.deleteLater)
        self._save_worker.start()

    def _on_save_t_series_done(self, result):
        self._monitor_bar.set_active(False)
        self._set_log_running(False)
        self._end_progress()
        self._btn_run.setEnabled(bool(self._input_channels))
        self._sync_preview_buttons()
        self._btn_save_series.setEnabled(bool(self._input_channels) and self._viewer.has_time_axis())
        self._save_worker = None

        if isinstance(result, Exception):
            msg = str(result)
            if "Stopped by user" in msg:
                self._log("T-series export stopped by user.")
                self._status.showMessage("T-series export stopped", 5000)
            else:
                self._log(f"T-series export failed: {msg}")
                QMessageBox.critical(self, "Save Error", msg)
                self._status.showMessage("T-series export failed", 5000)
            return

        path = Path(result["path"])
        size = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        self._log(f"T-series export saved: {path} ({_format_bytes(size)})")
        self._status.showMessage(f"Saved → {Path(result['path']).name}", 5000)

    # -----------------------------------------------------------------------
    # Settings Save / Load / Restore
    # -----------------------------------------------------------------------

    def _settings_to_dict(self) -> dict:
        """Collect all UI parameter values into a JSON-serializable dict."""
        return {
            "method": self._method_combo.currentText(),
            "iterations": self._le_niter.text(),
            "tv_lambda": self._sp_tv_lambda.value(),
            "damping": self._damping_combo.currentText(),
            "damping_value": self._sp_damping.value(),
            "two_d_mode": self._two_d_mode_combo.currentText(),
            "two_d_wf_aggressiveness": self._two_d_wf_aggr_combo.currentText(),
            "two_d_wf_bg_radius_um": self._sp_two_d_wf_bg_radius.value(),
            "two_d_wf_bg_scale": self._sp_two_d_wf_bg_scale.value(),
            "dl_model_path": self._le_dl_model.text(),
            "dl_z_context": self._sp_dl_z_context.value(),
            "dl_batch_size": self._sp_dl_batch_size.value(),
            "dl_mixed_precision": self._cb_dl_mixed_precision.isChecked(),
            "dl_residual_strength": self._sp_dl_residual_strength.value(),
            "background": self._bg_combo.currentText(),
            "background_value": self._sp_bg_value.value(),
            "offset": self._offset_combo.currentText(),
            "offset_value": self._sp_offset.value(),
            "prefilter_sigma": self._sp_prefilter.value(),
            "start": self._start_combo.currentText(),
            "sparse_hessian_weight": self._sp_sparse_weight.value(),
            "sparse_hessian_reg": self._sp_sparse_reg.value(),
            "convergence": self._conv_combo.currentText(),
            "rel_threshold": self._sp_rel_thresh.value(),
            "check_every": self._sp_check_every.value(),
            "device": self._device_combo.currentText(),
            "na": self._sp_na.value(),
            "emission_wavelengths": self._le_emission.text(),
            "excitation_wavelengths": self._excitation_saved if not self._le_excitation.isEnabled() else self._le_excitation.text(),
            "pinhole_airy_units": self._pinhole_airy_saved if self._pinhole_stack.currentWidget() is self._le_pinhole_na else self._le_pinhole_airy.text(),
            "pixel_size_xy_nm": self._sp_px_xy.value(),
            "pixel_size_z_nm": self._sp_px_z.value(),
            "ri_immersion": self._sp_ri_imm.value(),
            "ri_sample": self._sp_ri_sample.value(),
            "embedding_medium": self._medium_combo.currentText(),
            "t_g": self._sp_tg.value() * 1000.0,
            "t_g0": self._sp_tg0.value() * 1000.0,
            "t_i0": self._sp_ti0.value() * 1000.0,
            "z_p": self._sp_zp.value() * 1000.0,
            "microscope_type": self._micro_combo.currentText(),
            "integrate_pixels": self._cb_integrate.isChecked(),
            "n_subpixels": self._sp_subpixels.value(),
            "n_pupil": self._sp_n_pupil.value(),
            "pct_lo": self._viewer.lo_percentile(),
            "pct_hi": self._viewer.hi_percentile(),
            "movie_enabled": self._cb_movie.isChecked() if self._movie_available else False,
            "movie_path": self._le_movie_path.text() if self._movie_available else "",
            "movie_fps": self._sp_movie_fps.value() if self._movie_available else 10,
            "movie_hold_endpoints": self._cb_movie_hold.isChecked() if self._movie_available else False,
            "movie_create_gif": self._cb_movie_gif.isChecked() if self._movie_available else False,
            "movie_layout_mode": self._movie_layout_combo.currentText() if self._movie_available else "Standard",
            "movie_difference_inset": self._movie_inset_combo.currentText() if self._movie_available else "None",
            "movie_title_text": self._le_movie_title.text() if self._movie_available else "",
            "movie_info_overlay": self._cb_movie_info.isChecked() if self._movie_available else False,
            "movie_log_convergence": self._cb_movie_log_convergence.isChecked() if self._movie_available else False,
            "last_leica_dir": self._last_leica_dir,
        }

    def _apply_settings(self, data: dict):
        """Populate UI widgets from a settings dict."""
        def _combo(combo: QComboBox, key: str):
            val = data.get(key)
            if val is not None:
                idx = combo.findText(str(val))
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        def _spin(spin, key: str):
            val = data.get(key)
            if val is not None:
                spin.setValue(type(spin.value())(val))

        def _line(le: QLineEdit, key: str):
            val = data.get(key)
            if val is not None:
                le.setText(str(val))

        _combo(self._method_combo, "method")
        _line(self._le_niter, "iterations")
        _spin(self._sp_tv_lambda, "tv_lambda")
        _combo(self._damping_combo, "damping")
        _spin(self._sp_damping, "damping_value")
        two_d_mode = data.get("two_d_mode")
        if two_d_mode is not None:
            if str(two_d_mode).strip().lower() == "auto":
                self._two_d_mode_combo.setCurrentText("Auto")
            elif str(two_d_mode).strip().lower() in {"legacy_2d", "legacy 2d"}:
                self._two_d_mode_combo.setCurrentText("Legacy 2D")
        aggr = data.get("two_d_wf_aggressiveness")
        if aggr is not None:
            lookup = {
                "very conservative": "Very Conservative",
                "conservative": "Conservative",
                "balanced": "Balanced",
                "strong": "Strong",
                "very strong": "Very Strong",
            }
            self._two_d_wf_aggr_combo.setCurrentText(lookup.get(str(aggr).strip().lower(), str(aggr)))
        _spin(self._sp_two_d_wf_bg_radius, "two_d_wf_bg_radius_um")
        _spin(self._sp_two_d_wf_bg_scale, "two_d_wf_bg_scale")
        _line(self._le_dl_model, "dl_model_path")
        _spin(self._sp_dl_z_context, "dl_z_context")
        _spin(self._sp_dl_batch_size, "dl_batch_size")
        _spin(self._sp_dl_residual_strength, "dl_residual_strength")
        if "dl_mixed_precision" in data:
            self._cb_dl_mixed_precision.setChecked(bool(data.get("dl_mixed_precision")))
        _combo(self._bg_combo, "background")
        _spin(self._sp_bg_value, "background_value")
        _combo(self._offset_combo, "offset")
        _spin(self._sp_offset, "offset_value")
        _spin(self._sp_prefilter, "prefilter_sigma")
        _combo(self._start_combo, "start")
        _spin(self._sp_sparse_weight, "sparse_hessian_weight")
        _spin(self._sp_sparse_reg, "sparse_hessian_reg")
        _combo(self._conv_combo, "convergence")
        _spin(self._sp_rel_thresh, "rel_threshold")
        _spin(self._sp_check_every, "check_every")
        _combo(self._device_combo, "device")
        _spin(self._sp_na, "na")
        _line(self._le_emission, "emission_wavelengths")
        ex_val = data.get("excitation_wavelengths")
        if ex_val is not None:
            self._excitation_saved = str(ex_val)
            if self._le_excitation.isEnabled():
                self._le_excitation.setText(self._excitation_saved)
        pinhole_val = data.get("pinhole_airy_units")
        if pinhole_val is not None:
            if isinstance(pinhole_val, list):
                pinhole_text = _format_pinhole_values([float(v) for v in pinhole_val])
            else:
                # May be a single-value string ("1.37") or multi-channel string ("1.37, 1.19")
                parts = [p.strip() for p in str(pinhole_val).split(",") if p.strip()]
                try:
                    pinhole_text = _format_pinhole_values([float(p) for p in parts])
                except ValueError:
                    pinhole_text = str(pinhole_val)
            self._pinhole_airy_saved = pinhole_text
            if self._pinhole_stack.currentWidget() is self._le_pinhole_airy:
                self._le_pinhole_airy.setText(pinhole_text)
        _spin(self._sp_px_xy, "pixel_size_xy_nm")
        _spin(self._sp_px_z, "pixel_size_z_nm")
        _spin(self._sp_ri_imm, "ri_immersion")
        _spin(self._sp_ri_sample, "ri_sample")
        # Try to match combo to the loaded RI value; value is leading
        ri_val = data.get("ri_sample")
        if ri_val is not None:
            matched = False
            for name, ri in self._medium_ri_map.items():
                if abs(ri - float(ri_val)) < 1e-4:
                    self._medium_combo.blockSignals(True)
                    self._medium_combo.setCurrentText(name)
                    self._medium_combo.blockSignals(False)
                    matched = True
                    break
            if not matched:
                # No exact match — just leave combo as-is
                emb = data.get("embedding_medium")
                if emb is not None:
                    idx = self._medium_combo.findText(str(emb))
                    if idx >= 0:
                        self._medium_combo.blockSignals(True)
                        self._medium_combo.setCurrentIndex(idx)
                        self._medium_combo.blockSignals(False)
        t_g = data.get("t_g")
        if t_g is not None:
            self._sp_tg.setValue(float(t_g) / 1000.0)
        t_g0 = data.get("t_g0")
        if t_g0 is not None:
            self._sp_tg0.setValue(float(t_g0) / 1000.0)
        t_i0 = data.get("t_i0")
        if t_i0 is not None:
            self._sp_ti0.setValue(float(t_i0) / 1000.0)
        z_p = data.get("z_p")
        if z_p is not None:
            self._sp_zp.setValue(float(z_p) / 1000.0)
        _combo(self._micro_combo, "microscope_type")
        _spin(self._sp_subpixels, "n_subpixels")
        _spin(self._sp_n_pupil, "n_pupil")
        if data.get("pct_lo") is not None:
            self._viewer.set_lo_percentile(float(data["pct_lo"]))
        if data.get("pct_hi") is not None:
            self._viewer.set_hi_percentile(float(data["pct_hi"]))
        if self._movie_available:
            if data.get("movie_enabled") is not None:
                self._cb_movie.setChecked(bool(data["movie_enabled"]))
            _line(self._le_movie_path, "movie_path")
            _spin(self._sp_movie_fps, "movie_fps")
            if data.get("movie_hold_endpoints") is not None:
                self._cb_movie_hold.setChecked(bool(data["movie_hold_endpoints"]))
            if data.get("movie_create_gif") is not None:
                self._cb_movie_gif.setChecked(bool(data["movie_create_gif"]))
            _combo(self._movie_layout_combo, "movie_layout_mode")
            _combo(self._movie_inset_combo, "movie_difference_inset")
            _line(self._le_movie_title, "movie_title_text")
            if data.get("movie_info_overlay") is not None:
                self._cb_movie_info.setChecked(bool(data["movie_info_overlay"]))
            if data.get("movie_log_convergence") is not None:
                self._cb_movie_log_convergence.setChecked(bool(data["movie_log_convergence"]))

        # integrate_pixels checkbox
        val = data.get("integrate_pixels")
        if val is not None:
            self._cb_integrate.setChecked(bool(val))
        if data.get("last_leica_dir") is not None:
            self._last_leica_dir = _accessible_directory_or_home(data.get("last_leica_dir"))

    def _save_last_settings(self):
        """Auto-save settings to .last_settings.json."""
        try:
            with open(LAST_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(self._settings_to_dict(), f, indent=2)
            self._btn_restore.setEnabled(True)
        except OSError:
            pass

    def _on_restore_settings(self):
        """Restore settings from .last_settings.json."""
        try:
            with open(LAST_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self._apply_settings(data)
        self._viewer.refresh_view()
        self._status.showMessage("Settings restored", 3000)

    def _on_save_settings(self):
        """Save current settings to a user-chosen JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", str(Path(self._last_settings_dir) / "settings.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        self._last_settings_dir = str(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._settings_to_dict(), f, indent=2)
        self._status.showMessage(f"Settings saved → {Path(path).name}", 3000)

    def _on_load_settings(self):
        """Load settings from a user-chosen JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", self._last_settings_dir,
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        self._last_settings_dir = str(Path(path).parent)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Load Error", str(exc))
            return
        self._apply_settings(data)
        self._viewer.refresh_view()
        self._status.showMessage(f"Settings loaded from {Path(path).name}", 3000)

    def _configure_omero_login_dialog(self, dlg):
        """Hide any remember-me checkbox and force 10-minute reuse."""
        for cb in dlg.findChildren(QCheckBox):
            text = cb.text().strip().lower()
            if "remember" in text:
                cb.setChecked(True)
                cb.hide()

    def _refresh_omero_session_deadline(self):
        """Extend OMERO session reuse window from the current moment."""
        self._omero_session_deadline = time.monotonic() + OMERO_SESSION_REUSE_S

    def _omero_session_is_reusable(self) -> bool:
        """Return True when the current OMERO connection is still reusable."""
        if self._omero_gw is None or not self._omero_gw.is_connected():
            return False
        return time.monotonic() < self._omero_session_deadline

    # -----------------------------------------------------------------------
    # Viewer
    # -----------------------------------------------------------------------
    def _load_timepoint_into_viewer(self, timepoint: int, *, force: bool = False) -> None:
        if self._input_source is None:
            return
        target_t = max(0, min(int(timepoint), max(int(self._metadata.get("size_t", 1)) - 1, 0)))
        if not force and self._loaded_timepoint == target_t and self._input_channels:
            return

        size_z = max(int(self._metadata.get("size_z", 1)), 1)
        size_c = max(int(self._metadata.get("size_c", 1)), 1)
        is_omero_pyramid = (
            isinstance(self._input_source, _OmeroTimepointSource)
            and bool(getattr(self._input_source, "is_pyramidal", False))
        )
        if is_omero_pyramid:
            total = 1
        else:
            total = size_c * size_z if isinstance(self._input_source, _OmeroTimepointSource) and size_z > 1 else size_c
        if total > 1:
            self._begin_progress(total, f"Loading T={target_t + 1}…")
        else:
            self._begin_busy_progress(f"Loading T={target_t + 1}…")

        try:
            t_load = time.time()
            channels = self._input_source.load_timepoint(
                target_t,
                progress_cb=lambda done, total_steps, text: (
                    self._advance_progress(done, text),
                    self._log(text),
                ),
            )
            self._input_channels = channels
            self._loaded_timepoint = target_t
            self._viewer.set_input_timepoint_data(target_t, channels)
            if is_omero_pyramid:
                provider = getattr(self._input_source, "tile_provider", None)
                if provider is not None:
                    self._viewer.set_tiled_input_provider(provider)
            self._log(f"Loaded timepoint T={target_t + 1} in {_format_duration(time.time() - t_load)}")
        finally:
            self._end_progress()

    def _sync_preview_buttons(self) -> None:
        current_t = self._viewer.current_timepoint()
        has_preview = current_t in self._preview_outputs_by_t
        self._btn_save.setEnabled(has_preview)
        self._btn_save_view.setEnabled(has_preview)

    def _on_viewer_time_changed(self, timepoint: int) -> None:
        previous_t = self._loaded_timepoint
        try:
            self._load_timepoint_into_viewer(int(timepoint))
        except Exception as exc:
            if previous_t is not None and previous_t != int(timepoint):
                self._viewer.set_timepoint(previous_t)
            QMessageBox.critical(self, "Load Error", str(exc))
            self._status.showMessage("Timepoint load failed", 5000)
        self._sync_preview_buttons()

    def _update_viewer(self):
        self._viewer.refresh_view()

    def _on_browse_movie_path(self) -> None:
        stem = self._input_path.stem if self._input_path else "deconvolution_iterations"
        current = self._le_movie_path.text().strip()
        suggested = current or str(_default_downloads_dir() / f"{_safe_filename_stem(stem)}_iterations.mp4")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Iteration Movie",
            suggested,
            "MP4 movie (*.mp4);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        self._last_save_dir = str(Path(path).parent)
        self._le_movie_path.setText(path)

    def closeEvent(self, event):
        """Ensure background threads are stopped before the window closes."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)
            self._worker = None
        if self._save_worker is not None and self._save_worker.isRunning():
            self._save_worker.requestInterruption()
            self._save_worker.wait(2000)
            self._save_worker = None
        if self._monitor is not None:
            self._monitor.request_stop()
            self._monitor = None
        try:
            logging.getLogger().removeHandler(self._log_handler)
        except Exception:
            pass
        if self._omero_gw is not None:
            try:
                self._omero_gw.disconnect()
            except Exception:
                pass
            self._omero_gw = None
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _excepthook(exc_type, exc_value, exc_tb):
    """Show uncaught exceptions in a dialog instead of silently crashing."""
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    log.critical("Uncaught exception:\n%s", msg)
    print(msg, file=sys.stderr, flush=True)
    QMessageBox.critical(None, "Fatal Error", msg)


def main():
    sys.excepthook = _excepthook

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--movie",
        action="store_true",
        help="Expose advanced MP4 iteration movie export controls.",
    )
    parser.add_argument(
        "--fitpsf",
        action="store_true",
        help="Expose the experimental Fit PSF panel for refractive-index fitting.",
    )
    parser.add_argument(
        "--dlref",
        action="store_true",
        help="Expose the ci_rl_dl method and DL refinement parameter panel.",
    )
    args, qt_args = parser.parse_known_args(sys.argv[1:])

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("CI Deconvolve")
    app_icon = _load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    window = DeconvolveCIWindow(movie_available=args.movie, fitpsf_available=args.fitpsf, dlref_available=args.dlref)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
