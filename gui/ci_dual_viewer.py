"""Dual-pane XYZT / 3D viewer for the CI deconvolution GUI.

Adapted from the omero-browser-qt 0.2.2 viewer concepts, but packaged as an
embeddable widget with fixed Original / Deconvolved panes and shared controls.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from vispy import scene as vispy_scene
    from vispy.color import BaseColormap
    from vispy.visuals.transforms import STTransform

    _HAS_VISPY = True
except ImportError:
    _HAS_VISPY = False


_FALLBACK_PALETTE = [
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
]

_RGB_CHANNEL_NAMES = ("R", "G", "B")
_RGB_CHANNEL_COLORS = (
    (220, 68, 68),
    (56, 184, 104),
    (72, 136, 255),
)

_PROJECTION_MODES = ["Slice", "MIP", "SUM"]
_VIEW_SELECTOR_MODES = ["Both", "Original", "Deconvolved"]
_TWO_D_MODE = "2D"
_THREE_D_MODE = "3D"
_VOLUME_METHODS = [
    "mip",
    "attenuated_mip",
    "minip",
    "translucent",
    "average",
    "iso",
    "additive",
]
_VOLUME_METHOD_LABELS = {
    "mip": "MIP",
    "attenuated_mip": "Attenuated MIP",
    "minip": "MinIP",
    "translucent": "Translucent",
    "average": "Average",
    "iso": "Isosurface",
    "additive": "Additive",
}
_VOLUME_METHOD_UI = {
    "mip": {"label": "Gain:", "range": (1, 200), "default": 100, "role": "gain"},
    "attenuated_mip": {"label": "Atten.:", "range": (1, 300), "default": 100, "role": "attenuation"},
    "minip": {"label": "Cutoff:", "range": (0, 100), "default": 100, "role": "minip_cutoff"},
    "translucent": {"label": "Gain:", "range": (1, 500), "default": 200, "role": "gain"},
    "average": {"label": "Gain:", "range": (1, 600), "default": 180, "role": "gain"},
    "iso": {"label": "Threshold:", "range": (0, 100), "default": 22, "role": "threshold"},
    "additive": {"label": "Gain:", "range": (1, 200), "default": 28, "role": "gain"},
}
_INTERPOLATION_TOGGLE_METHODS = set(_VOLUME_METHODS) - {"iso"}
_MAX_HIST_SAMPLES = 1_000_000
_MAX_VIEW_PIXELS = 2_000_000


def _toolbar_icon(kind: str, color: str = "#9cc8ff", size: int = 18) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color), 1.8)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    s = float(size)
    if kind == "eye":
        painter.drawEllipse(QRectF(2.5, 5.2, s - 5.0, s - 10.4))
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(s / 2.0, s / 2.0), 2.5, 2.5)
    elif kind == "projection":
        painter.drawRect(QRectF(3.0, 3.0, s - 8.0, s - 8.0))
        painter.drawRect(QRectF(6.5, 6.5, s - 8.0, s - 8.0))
    elif kind == "display":
        for y, x in ((4.5, 4.0), (9.0, 2.5), (13.5, 5.5)):
            painter.drawLine(QPointF(2.0, y), QPointF(s - 2.0, y))
            painter.setBrush(QColor(color))
            painter.drawEllipse(QPointF(x + 7.0, y), 2.0, 2.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.end()
    return pixmap


@dataclass(slots=True)
class _ScaleBarSpec:
    image_pixels: float
    screen_pixels: float
    label: str
    physical_um: float


def _sample_flat_for_hist(arr: np.ndarray, max_samples: int = _MAX_HIST_SAMPLES) -> np.ndarray:
    """Return a cheap representative 1-D sample for histogram/percentile work."""
    flat = np.asarray(arr).ravel()
    if flat.size <= max_samples:
        return flat
    step = max(1, int(np.ceil(flat.size / max_samples)))
    return flat[::step]


def _finite_sample(arr: np.ndarray, max_samples: int = _MAX_HIST_SAMPLES) -> np.ndarray:
    sample = _sample_flat_for_hist(arr, max_samples)
    if sample.size == 0:
        return sample
    return sample[np.isfinite(sample)]


def _display_stride(shape: tuple[int, int], max_pixels: int = _MAX_VIEW_PIXELS) -> int:
    pixels = int(shape[0]) * int(shape[1])
    if pixels <= max_pixels:
        return 1
    return max(1, int(np.ceil(np.sqrt(pixels / max_pixels))))


def _format_physical_length(physical_um: float) -> str:
    if physical_um >= 1000:
        mm = physical_um / 1000.0
        if mm >= 10:
            return f"{mm:.0f} mm"
        if mm >= 1:
            return f"{mm:.1f} mm"
        return f"{mm:.2f} mm"
    if physical_um >= 10:
        return f"{physical_um:.0f} um"
    if physical_um >= 1:
        return f"{physical_um:.1f} um"
    return f"{physical_um:.2f} um"


def _compute_scale_bar(
    um_per_image_pixel: Optional[float],
    screen_pixels_per_image_pixel: float,
    *,
    target_screen_px: float = 120.0,
    min_screen_px: float = 70.0,
    max_screen_px: float = 180.0,
) -> Optional[_ScaleBarSpec]:
    if (
        um_per_image_pixel is None
        or um_per_image_pixel <= 0
        or screen_pixels_per_image_pixel <= 0
    ):
        return None

    raw_um = target_screen_px * um_per_image_pixel / screen_pixels_per_image_pixel
    if raw_um <= 0:
        return None

    magnitude = 10 ** math.floor(math.log10(raw_um))
    for factor in (1, 2, 5, 10):
        physical_um = factor * magnitude
        screen_px = physical_um / um_per_image_pixel * screen_pixels_per_image_pixel
        if min_screen_px <= screen_px <= max_screen_px:
            break
    else:
        physical_um = raw_um
        screen_px = target_screen_px

    return _ScaleBarSpec(
        image_pixels=physical_um / um_per_image_pixel,
        screen_pixels=screen_px,
        label=_format_physical_length(physical_um),
        physical_um=physical_um,
    )


def _emission_to_rgb(wavelength_nm: Optional[float]) -> tuple[int, int, int]:
    if wavelength_nm is None:
        return (255, 255, 255)
    wl = float(wavelength_nm)
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


def _channels_look_like_rgb(channels: list[dict]) -> bool:
    if len(channels) != 3:
        return False
    names = [str(ch.get("name", "")).strip().lower() for ch in channels]
    if names in (["r", "g", "b"], ["red", "green", "blue"]):
        return True
    colors = [tuple(ch.get("color", (0, 0, 0))) for ch in channels]
    return set(colors) == set(_RGB_CHANNEL_COLORS)


def _channels_look_fluorescence_like(channels: list[dict]) -> bool:
    if not channels or _channels_look_like_rgb(channels):
        return False
    fluor_markers = (
        "dapi", "fitc", "gfp", "yfp", "cfp", "rfp", "mcherry", "tdtomato",
        "tritc", "cy3", "cy5", "alexa", "hoechst", "far red",
    )
    for ch in channels:
        emission = ch.get("emission_wavelength")
        if emission is not None:
            try:
                if float(emission) > 0:
                    return True
            except (TypeError, ValueError):
                pass
        name = str(ch.get("name", "")).strip().lower()
        if any(marker in name for marker in fluor_markers):
            return True
    return False


def _resolve_channel_colors(channels: list[dict]) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int] | None] = []
    for ch in channels:
        color = ch.get("color")
        if isinstance(color, str):
            text = color.strip().lstrip("#")
            if len(text) == 6:
                try:
                    colors.append(tuple(int(text[i:i + 2], 16) for i in (0, 2, 4)))
                    continue
                except ValueError:
                    pass
        if isinstance(color, (list, tuple)) and len(color) >= 3 and tuple(color[:3]) != (255, 255, 255):
            colors.append(tuple(int(v) for v in color[:3]))
            continue
        emission = ch.get("emission_wavelength")
        colors.append(_emission_to_rgb(emission))
    if len(colors) > 1 and len(set(colors)) == 1:
        return [_FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)] for i in range(len(colors))]
    return [c if c is not None else _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)] for i, c in enumerate(colors)]


def _project_stack(stack: np.ndarray, mode: str, z_index: int) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"Expected ZYX stack, got {stack.shape}")
    if mode == "Slice":
        z = max(0, min(int(z_index), stack.shape[0] - 1))
        return stack[z]
    if mode == "MIP":
        return stack.max(axis=0)
    if mode == "SUM":
        return stack.sum(axis=0).astype(np.float64)
    raise ValueError(f"Unsupported projection mode: {mode}")


def _percentile_from_hist(bin_edges: np.ndarray, counts: np.ndarray, pct: float) -> float:
    """Return a percentile value from a precomputed histogram (fast, O(n_bins))."""
    total = int(counts.sum())
    if total == 0:
        return float(bin_edges[0])
    target = pct / 100.0 * total
    cumsum = np.cumsum(counts)
    idx = int(np.searchsorted(cumsum, target))
    idx = min(idx, len(bin_edges) - 2)
    prev_cum = int(cumsum[idx - 1]) if idx > 0 else 0
    bin_frac = (target - prev_cum) / max(float(counts[idx]), 1.0)
    return float(bin_edges[idx] + bin_frac * (bin_edges[idx + 1] - bin_edges[idx]))


def _composite_to_rgb(
    slices: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]],
) -> np.ndarray:
    if not slices:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    height, width = slices[0][0].shape
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    for arr, (cr, cg, cb), (lo, hi, gamma) in slices:
        if hi <= lo:
            hi = lo + 1.0
        norm = (arr.astype(np.float32, copy=False) - np.float32(lo)) / np.float32(hi - lo)
        np.clip(norm, 0.0, 1.0, out=norm)
        gamma_safe = max(float(gamma), 1e-3)
        if abs(gamma_safe - 1.0) > 1e-6:
            np.power(norm, 1.0 / gamma_safe, out=norm)
        canvas[..., 0] += norm * (cr / 255.0)
        canvas[..., 1] += norm * (cg / 255.0)
        canvas[..., 2] += norm * (cb / 255.0)
    np.clip(canvas, 0.0, 1.0, out=canvas)
    return np.ascontiguousarray((canvas * 255).astype(np.uint8))


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    if rgb.size == 0:
        return QImage()
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    qimg = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return qimg.copy()


def _composite_to_pixmap(
    slices: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]],
) -> QPixmap:
    if not slices:
        return QPixmap()
    stride = _display_stride(slices[0][0].shape)
    if stride > 1:
        slices = [(arr[::stride, ::stride], color, contrast) for arr, color, contrast in slices]
    return QPixmap.fromImage(_rgb_to_qimage(_composite_to_rgb(slices)))


class ZoomableImageView(QGraphicsView):
    """Simple pannable / zoomable QGraphicsView with optional linking."""

    cursorMoved = pyqtSignal(float, float, bool)
    rightSplitDragged = pyqtSignal(float)
    panDragStarted = pyqtSignal()
    panDragFinished = pyqtSignal()

    _ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._custom_item = None
        self._linked: list["ZoomableImageView"] = []
        self._syncing = False
        self._smooth_zoom = False
        self._scale_bar_enabled = True
        self._scale_bar_um_per_pixel: Optional[float] = None
        self._navigator_enabled = True
        self._right_dragging_split = False
        self._navigator_dragging = False
        self._navigator_rect = QRectF()
        self._right_dragging_pan = False
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumSize(260, 260)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setStyleSheet(
            "QGraphicsView { background: #17191c; border: 1px solid #343a40; border-radius: 8px; }"
        )

    def link_to(self, other: "ZoomableImageView") -> None:
        if other not in self._linked:
            self._linked.append(other)
        if self not in other._linked:
            other._linked.append(self)

    def set_smooth_zoom(self, enabled: bool) -> None:
        self._smooth_zoom = bool(enabled)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._smooth_zoom)
        self._apply_pixmap_transformation_mode()
        self.viewport().update()

    def set_scale_bar_enabled(self, enabled: bool) -> None:
        self._scale_bar_enabled = bool(enabled)
        self.viewport().update()

    def set_scale_bar_um_per_pixel(self, value: Optional[float]) -> None:
        try:
            value_f = float(value) if value is not None else None
        except (TypeError, ValueError):
            value_f = None
        self._scale_bar_um_per_pixel = value_f if value_f is not None and value_f > 0 else None
        self.viewport().update()

    def set_navigator_enabled(self, enabled: bool) -> None:
        self._navigator_enabled = bool(enabled)
        self.viewport().update()

    def set_interaction_cursor(self, shape: Qt.CursorShape) -> None:
        self.setCursor(shape)
        self.viewport().setCursor(shape)

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._scene.clear()
        self._pix_item = None
        self._custom_item = None
        if pixmap is not None and not pixmap.isNull():
            self._pix_item = self._scene.addPixmap(pixmap)
            self._apply_pixmap_transformation_mode()
            self._scene.setSceneRect(QRectF(pixmap.rect()))

    def set_graphics_item(self, item) -> None:
        """Show a custom graphics item, for example a progressive tiled pyramid."""
        if item is self._custom_item:
            if item is not None:
                self._scene.setSceneRect(item.boundingRect())
            return
        self._scene.clear()
        self._pix_item = None
        self._custom_item = item
        if item is not None:
            self._scene.addItem(item)
            self._scene.setSceneRect(item.boundingRect())

    def _apply_pixmap_transformation_mode(self) -> None:
        if self._pix_item is None:
            return
        mode = (
            Qt.TransformationMode.SmoothTransformation
            if self._smooth_zoom
            else Qt.TransformationMode.FastTransformation
        )
        self._pix_item.setTransformationMode(mode)

    def clear(self) -> None:
        self._scene.clear()
        self._pix_item = None
        self._custom_item = None

    def fit_in_view(self) -> None:
        if self._pix_item is not None:
            self.resetTransform()
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        elif not self._scene.sceneRect().isEmpty():
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def copy_view_state_from(self, other: "ZoomableImageView") -> None:
        self.setTransform(other.transform())
        self.horizontalScrollBar().setValue(other.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(other.verticalScrollBar().value())

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        factor = self._ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / self._ZOOM_FACTOR
        self.scale(factor, factor)
        self._sync_transform()

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802
        super().scrollContentsBy(dx, dy)
        if not self._syncing:
            self._sync_transform()

    def _sync_transform(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        transform = self.transform()
        x_value = self.horizontalScrollBar().value()
        y_value = self.verticalScrollBar().value()
        for other in self._linked:
            other._syncing = True
            other.setTransform(transform)
            other.horizontalScrollBar().setValue(x_value)
            other.verticalScrollBar().setValue(y_value)
            other.viewport().update()
            other._syncing = False
        self._syncing = False

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        self._draw_minimap()
        self._draw_scale_bar()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._pix_item is not None:
            self._emit_split_ratio(event.position().x())
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton and self._navigator_contains(event.position()):
            self._navigator_dragging = True
            self._pan_from_navigator(event.position())
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton and self._pix_item is not None:
            self._right_dragging_pan = True
            self.set_interaction_cursor(Qt.CursorShape.ClosedHandCursor)
            self.panDragStarted.emit()
            mapped = QMouseEvent(
                event.type(),
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            super().mousePressEvent(mapped)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._navigator_dragging:
            self._pan_from_navigator(event.position())
            event.accept()
            return
        if event.buttons() & Qt.MouseButton.LeftButton and self._pix_item is not None:
            self._emit_split_ratio(event.position().x())
            event.accept()
            return
        if self._right_dragging_pan:
            mapped = QMouseEvent(
                event.type(),
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            super().mouseMoveEvent(mapped)
            event.accept()
            return
        scene_pos = self.mapToScene(event.pos())
        rect = self._scene.sceneRect()
        inside = rect.contains(scene_pos)
        self.cursorMoved.emit(float(scene_pos.x()), float(scene_pos.y()), bool(inside))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._navigator_dragging:
            self._navigator_dragging = False
            self._pan_from_navigator(event.position())
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton and self._right_dragging_split:
            self._right_dragging_split = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton and self._right_dragging_pan:
            self._right_dragging_pan = False
            mapped = QMouseEvent(
                event.type(),
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                event.modifiers(),
            )
            super().mouseReleaseEvent(mapped)
            self.panDragFinished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.cursorMoved.emit(0.0, 0.0, False)
        super().leaveEvent(event)

    def _draw_minimap(self) -> None:
        if not self._navigator_enabled:
            return
        if self._pix_item is None or self._pix_item.pixmap().isNull():
            return
        if self.transform().m11() <= 1.25 and self.transform().m22() <= 1.25:
            self._navigator_rect = QRectF()
            return
        pixmap = self._pix_item.pixmap()
        max_w = min(150, max(80, self.viewport().width() // 5))
        max_h = min(110, max(60, self.viewport().height() // 5))
        overview = pixmap.scaled(
            max_w,
            max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if overview.isNull():
            return
        margin = 14
        x = self.viewport().width() - overview.width() - margin
        y = margin
        self._navigator_rect = QRectF(float(x), float(y), float(overview.width()), float(overview.height()))
        sx = overview.width() / max(float(pixmap.width()), 1.0)
        sy = overview.height() / max(float(pixmap.height()), 1.0)
        visible = self.mapToScene(self.viewport().rect()).boundingRect().intersected(self._scene.sceneRect())
        visible_rect = QRectF(
            x + visible.x() * sx,
            y + visible.y() * sy,
            visible.width() * sx,
            visible.height() * sy,
        )

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(2, 6, 23, 180))
        painter.drawRoundedRect(x - 5, y - 5, overview.width() + 10, overview.height() + 10, 7, 7)
        painter.drawPixmap(x, y, overview)
        painter.setPen(QPen(QColor("#f8fafc"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(visible_rect)
        painter.end()

    def _emit_split_ratio(self, viewport_x: float) -> None:
        width = max(float(self.viewport().width()), 1.0)
        ratio = max(0.05, min(0.95, float(viewport_x) / width))
        self.rightSplitDragged.emit(ratio)

    def _navigator_contains(self, pos: QPointF) -> bool:
        return (
            self._navigator_enabled
            and self._pix_item is not None
            and not self._navigator_rect.isNull()
            and self._navigator_rect.contains(pos)
        )

    def _pan_from_navigator(self, pos: QPointF) -> None:
        if self._pix_item is None or self._navigator_rect.isNull():
            return
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        rx = (float(pos.x()) - self._navigator_rect.x()) / max(self._navigator_rect.width(), 1.0)
        ry = (float(pos.y()) - self._navigator_rect.y()) / max(self._navigator_rect.height(), 1.0)
        scene_x = rect.left() + max(0.0, min(1.0, rx)) * rect.width()
        scene_y = rect.top() + max(0.0, min(1.0, ry)) * rect.height()
        self.centerOn(scene_x, scene_y)
        self._sync_transform()
        self.viewport().update()

    def _draw_scale_bar(self) -> None:
        if not self._scale_bar_enabled:
            return
        spec = _compute_scale_bar(self._scale_bar_um_per_pixel, self.transform().m11())
        if spec is None:
            return

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        margin = 18
        bar_x = margin
        bar_y = self.viewport().height() - margin
        text_rect = painter.fontMetrics().boundingRect(spec.label)
        bg_width = int(max(spec.screen_pixels + 16, text_rect.width() + 20))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(2, 6, 23, 190))
        painter.drawRoundedRect(bar_x - 8, bar_y - 34, bg_width, 38, 8, 8)

        painter.setPen(QPen(QColor("#f8fafc"), 2))
        painter.drawLine(int(bar_x), int(bar_y - 10), int(bar_x + spec.screen_pixels), int(bar_y - 10))
        painter.drawLine(int(bar_x), int(bar_y - 14), int(bar_x), int(bar_y - 6))
        painter.drawLine(
            int(bar_x + spec.screen_pixels),
            int(bar_y - 14),
            int(bar_x + spec.screen_pixels),
            int(bar_y - 6),
        )
        painter.drawText(
            int(bar_x),
            int(bar_y - 30),
            max(int(spec.screen_pixels), text_rect.width() + 4),
            16,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            spec.label,
        )
        painter.end()


@dataclass
class _PaneMessage:
    text: str


class _PaneWidget(QWidget):
    cameraStateChanged = pyqtSignal(object)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._syncing_camera = False
        self._volumes: list[object] = []
        self._vol_camera_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
        self._camera_emit_timer = QTimer(self)
        self._camera_emit_timer.setSingleShot(True)
        self._camera_emit_timer.setInterval(30)
        self._camera_emit_timer.timeout.connect(self._emit_camera_state)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 700; color: #f3f4f6;")
        layout.addWidget(title_label)

        self._display_stack = QStackedWidget()
        layout.addWidget(self._display_stack, stretch=1)

        self._mode_stack = QStackedWidget()
        self._display_stack.addWidget(self._mode_stack)

        self.view2d = ZoomableImageView()
        self._mode_stack.addWidget(self.view2d)

        if _HAS_VISPY:
            self._vispy_canvas = vispy_scene.SceneCanvas(keys="interactive", show=False, bgcolor="#000000")
            self._vispy_view = self._vispy_canvas.central_widget.add_view()
            self._vispy_view.camera = vispy_scene.ArcballCamera(fov=60, distance=None)
            self._connect_camera_events(self._vispy_view.camera)
            self._mode_stack.addWidget(self._vispy_canvas.native)
        else:
            self._vispy_canvas = None
            self._vispy_view = None
            missing = QLabel("3D viewer unavailable.\nInstall vispy and PyOpenGL.")
            missing.setAlignment(Qt.AlignmentFlag.AlignCenter)
            missing.setStyleSheet("color: #c0c4c8; background: #111315; border: 1px solid #343a40; border-radius: 8px;")
            self._mode_stack.addWidget(missing)

        self._placeholder = QLabel("")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            "color: #d0d3d7; background: #111315; border: 1px dashed #43484d; border-radius: 8px; padding: 18px;"
        )
        self._display_stack.addWidget(self._placeholder)
        self._display_stack.setCurrentIndex(0)

    def set_mode(self, mode: str) -> None:
        if mode == _THREE_D_MODE:
            self._mode_stack.setCurrentIndex(1)
        else:
            self._mode_stack.setCurrentIndex(0)

    def show_placeholder(self, text: str) -> None:
        self._placeholder.setText(text)
        self._display_stack.setCurrentIndex(1)
        self.view2d.clear()
        self.clear_3d()

    def show_content(self) -> None:
        self._display_stack.setCurrentIndex(0)

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self.show_content()
        self._mode_stack.setCurrentIndex(0)
        self.view2d.set_pixmap(pixmap)

    def set_graphics_item(self, item) -> None:
        self.show_content()
        self._mode_stack.setCurrentIndex(0)
        self.view2d.set_graphics_item(item)

    def fit_2d(self) -> None:
        self.view2d.fit_in_view()

    def copy_2d_view_from(self, other: "_PaneWidget") -> None:
        self.view2d.copy_view_state_from(other.view2d)

    def set_smooth_zoom(self, enabled: bool) -> None:
        self.view2d.set_smooth_zoom(enabled)

    def set_scale_bar(self, enabled: bool, um_per_pixel: Optional[float]) -> None:
        self.view2d.set_scale_bar_enabled(enabled)
        self.view2d.set_scale_bar_um_per_pixel(um_per_pixel)

    def set_navigator_enabled(self, enabled: bool) -> None:
        self.view2d.set_navigator_enabled(enabled)

    def grab_3d_image(self) -> QImage:
        if not _HAS_VISPY or self._vispy_canvas is None:
            return QImage()
        try:
            self._vispy_canvas.update()
            rgba = np.ascontiguousarray(self._vispy_canvas.render(), dtype=np.uint8)
        except Exception:
            return QImage()
        if rgba.ndim != 3 or rgba.shape[2] < 3:
            return QImage()
        if rgba.shape[2] >= 4:
            height, width = rgba.shape[:2]
            qimg = QImage(rgba.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
            return qimg.copy()
        return _rgb_to_qimage(rgba[..., :3])

    def clear_3d(self) -> None:
        if not _HAS_VISPY or self._vispy_view is None:
            return
        for volume in self._volumes:
            try:
                volume.parent = None
            except Exception:
                pass
        self._volumes.clear()
        self._vol_camera_ranges = None
        self._vispy_canvas.update()

    def load_3d(
        self,
        stacks: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]],
        *,
        method: str,
        slider_val: float,
        interpolation: str,
        downsample: int,
        pixel_size_x: Optional[float],
        pixel_size_z: Optional[float],
        preserve_camera_state: object = None,
    ) -> None:
        self.show_content()
        self._mode_stack.setCurrentIndex(1)
        if not _HAS_VISPY or self._vispy_view is None:
            return
        self.clear_3d()
        z_scale = 1.0
        if pixel_size_x and pixel_size_z and pixel_size_x > 0:
            z_scale = float(pixel_size_z) / float(pixel_size_x)

        reference_shape: tuple[int, int, int] | None = None
        for stack, color, contrast in stacks:
            if reference_shape is None:
                reference_shape = tuple(int(v) for v in stack.shape)
            work = stack[::downsample, ::downsample, ::downsample] if downsample > 1 else stack
            lo, hi, gamma = contrast
            gain = slider_val if _VOLUME_METHOD_UI[method]["role"] == "gain" else 1.0
            data = self._prepare_volume_data(work, lo, hi, gamma, method, gain)
            cmap = _ChannelColormap(color, translucent_boost=(method == "translucent"))
            volume = vispy_scene.visuals.Volume(
                data,
                parent=self._vispy_view.scene,
                method=method,
                threshold=slider_val if _VOLUME_METHOD_UI[method]["role"] == "threshold" else 0.0,
                attenuation=slider_val if _VOLUME_METHOD_UI[method]["role"] == "attenuation" else 1.0,
                mip_cutoff=slider_val if _VOLUME_METHOD_UI[method]["role"] == "mip_cutoff" else None,
                minip_cutoff=slider_val if _VOLUME_METHOD_UI[method]["role"] == "minip_cutoff" else None,
                cmap=cmap,
                interpolation=interpolation,
            )
            volume.transform = STTransform(scale=(downsample, downsample, z_scale * downsample))
            volume.set_gl_state("additive", depth_test=False)
            self._volumes.append(volume)

        if reference_shape is not None:
            oz, oy, ox = reference_shape
            self._vol_camera_ranges = (
                (0.0, float(max(ox - 1, 0))),
                (0.0, float(max(oy - 1, 0))),
                (0.0, float(max(oz - 1, 0)) * z_scale),
            )
        if preserve_camera_state is not None:
            self.reset_camera()
            self.apply_camera_state(preserve_camera_state)
        else:
            self.reset_camera()

    def apply_camera_state(self, state: object) -> None:
        if not _HAS_VISPY or self._vispy_view is None or state is None:
            return
        camera = self._vispy_view.camera
        if not hasattr(camera, "set_state"):
            return
        self._syncing_camera = True
        try:
            camera.set_state(state)
            self._vispy_canvas.update()
        finally:
            self._syncing_camera = False

    def camera_state(self) -> object:
        if not _HAS_VISPY or self._vispy_view is None:
            return None
        camera = self._vispy_view.camera
        if hasattr(camera, "get_state"):
            return dict(camera.get_state())
        return None

    def reset_camera(self) -> None:
        if not _HAS_VISPY or self._vispy_view is None:
            return
        camera = vispy_scene.ArcballCamera(fov=60, distance=None)
        self._vispy_view.camera = camera
        self._connect_camera_events(camera)
        if self._vol_camera_ranges is not None:
            x_range, y_range, z_range = self._vol_camera_ranges
            camera.set_range(x=x_range, y=y_range, z=z_range, margin=0.05)
            camera.center = (
                0.5 * (x_range[0] + x_range[1]),
                0.5 * (y_range[0] + y_range[1]),
                0.5 * (z_range[0] + z_range[1]),
            )
            if hasattr(camera, "set_default_state"):
                camera.set_default_state()
        self._vispy_canvas.update()

    def _connect_camera_events(self, camera) -> None:
        """Hook camera-change notifications in a vispy-version-tolerant way."""
        camera_events = getattr(camera, "events", None)
        if camera_events is not None:
            changed = getattr(camera_events, "changed", None)
            if changed is not None and hasattr(changed, "connect"):
                changed.connect(self._on_camera_changed)
                return

        if self._vispy_canvas is None:
            return
        canvas_events = getattr(self._vispy_canvas, "events", None)
        if canvas_events is None:
            return
        for name in ("mouse_move", "mouse_wheel", "mouse_release", "resize"):
            emitter = getattr(canvas_events, name, None)
            if emitter is not None and hasattr(emitter, "connect"):
                emitter.connect(self._schedule_camera_emit)

    def _schedule_camera_emit(self, _event=None) -> None:
        if self._syncing_camera:
            return
        self._camera_emit_timer.start()

    def _emit_camera_state(self) -> None:
        state = self.camera_state()
        if state is not None:
            self.cameraStateChanged.emit(state)

    def _prepare_volume_data(
        self,
        stack: np.ndarray,
        lo: float,
        hi: float,
        gamma: float,
        method: str,
        gain: float,
    ) -> np.ndarray:
        volume = stack.astype(np.float32)
        denom = hi - lo if hi > lo else 1.0
        volume = (volume - lo) / denom
        np.clip(volume, 0.0, 1.0, out=volume)
        gamma_safe = max(float(gamma), 1e-3)
        if abs(gamma_safe - 1.0) > 1e-6:
            np.power(volume, 1.0 / gamma_safe, out=volume)
        if method == "translucent":
            np.power(volume, 0.5, out=volume)
            volume *= gain
        elif method == "average":
            np.power(volume, 0.7, out=volume)
            volume = 1.0 - np.exp(-(volume * gain * 1.8))
        else:
            volume *= gain
        np.clip(volume, 0.0, 1.0, out=volume)
        return volume

    def _on_camera_changed(self, _event=None) -> None:
        if self._syncing_camera:
            return
        self._schedule_camera_emit()


class _HistogramWidget(QWidget):
    markerDragged = pyqtSignal(str, str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original_bin_edges = np.array([0.0, 1.0], dtype=np.float64)
        self._deconvolved_bin_edges = np.array([0.0, 1.0], dtype=np.float64)
        self._original_counts = np.zeros(1, dtype=np.float64)
        self._deconvolved_counts = np.zeros(1, dtype=np.float64)
        self._channel_color = (255, 255, 255)
        self._original_range = (0.0, 1.0)
        self._deconvolved_range = (0.0, 1.0)
        self._has_original = False
        self._has_deconvolved = False
        self._log_scale = True
        self._drag_target: tuple[str, str] | None = None
        self._plot_rects: dict[str, QRectF] = {}
        self.setMinimumHeight(240)

    def set_histogram(
        self,
        original_bin_edges: np.ndarray,
        deconvolved_bin_edges: np.ndarray,
        original_counts: np.ndarray,
        deconvolved_counts: np.ndarray,
        channel_color: tuple[int, int, int],
        original_range: tuple[float, float],
        deconvolved_range: tuple[float, float],
        has_original: bool,
        has_deconvolved: bool,
        *,
        log_scale: bool,
    ) -> None:
        self._original_bin_edges = np.asarray(original_bin_edges, dtype=np.float64)
        self._deconvolved_bin_edges = np.asarray(deconvolved_bin_edges, dtype=np.float64)
        self._original_counts = np.asarray(original_counts, dtype=np.float64)
        self._deconvolved_counts = np.asarray(deconvolved_counts, dtype=np.float64)
        self._channel_color = tuple(int(v) for v in channel_color)
        self._original_range = (float(original_range[0]), float(original_range[1]))
        self._deconvolved_range = (float(deconvolved_range[0]), float(deconvolved_range[1]))
        self._has_original = bool(has_original)
        self._has_deconvolved = bool(has_deconvolved)
        self._log_scale = bool(log_scale)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#1a1d20"))

        outer_rect = self.rect().adjusted(12, 12, -12, -28)
        if outer_rect.width() <= 0 or outer_rect.height() <= 0:
            return

        any_original = self._has_original and self._original_counts.size > 0 and np.any(self._original_counts > 0)
        any_deconvolved = (
            self._has_deconvolved
            and self._deconvolved_counts.size > 0
            and np.any(self._deconvolved_counts > 0)
        )
        if (
            self._original_bin_edges.size < 2
            and self._deconvolved_bin_edges.size < 2
        ) or (not any_original and not any_deconvolved):
            painter.setPen(QColor("#aeb4ba"))
            painter.drawText(outer_rect, Qt.AlignmentFlag.AlignCenter, "No histogram data")
            return

        gap = 12
        half_height = max((outer_rect.height() - gap) / 2.0, 32.0)
        original_rect = QRectF(
            outer_rect.left(),
            outer_rect.top(),
            outer_rect.width(),
            half_height,
        )
        deconvolved_rect = QRectF(
            outer_rect.left(),
            original_rect.bottom() + gap,
            outer_rect.width(),
            half_height,
        )
        self._plot_rects = {
            "original": original_rect,
            "deconvolved": deconvolved_rect,
        }

        if any_original:
            y_original = np.log1p(self._original_counts) if self._log_scale else self._original_counts
            max_count_original = max(1.0, float(np.max(y_original)))
        else:
            y_original = np.zeros_like(self._original_counts, dtype=np.float64)
            max_count_original = 1.0
        if any_deconvolved:
            y_deconvolved = (
                np.log1p(self._deconvolved_counts)
                if self._log_scale
                else self._deconvolved_counts
            )
            max_count_deconvolved = max(1.0, float(np.max(y_deconvolved)))
        else:
            y_deconvolved = np.zeros_like(self._deconvolved_counts, dtype=np.float64)
            max_count_deconvolved = 1.0

        def _x_for_value(value: float, rect: QRectF, bin_edges: np.ndarray) -> int:
            axis_min = float(bin_edges[0])
            axis_max = float(bin_edges[-1]) if float(bin_edges[-1]) > axis_min else axis_min + 1.0
            pct = (float(value) - axis_min) / max(axis_max - axis_min, 1e-12)
            pct = min(max(pct, 0.0), 1.0)
            return int(rect.left() + pct * rect.width())

        def _draw_histogram(
            rect: QRectF,
            title: str,
            bin_edges: np.ndarray,
            values: np.ndarray,
            color: QColor,
            width_px: int,
            value_range: tuple[float, float],
            max_count: float,
        ) -> None:
            painter.fillRect(rect.adjusted(1, 1, -1, -1), QColor("#0b0c0e"))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#343a40")))
            painter.drawRect(rect)
            painter.setPen(QColor("#d7dce1"))
            painter.drawText(
                rect.adjusted(6, 4, -6, -4),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                title,
            )
            pen = QPen(color, max(1, width_px - 1))
            painter.setPen(pen)
            baseline = int(rect.bottom()) - 1
            width = max(int(rect.width()), 1)
            top = int(rect.top()) + 20
            height = max(baseline - top, 1)
            n_bins = max(len(bin_edges) - 1, 1)
            column_values = np.zeros(width, dtype=np.float64)
            if len(values):
                x_columns = ((np.arange(n_bins, dtype=np.float64) + 0.5) * width / n_bins).astype(np.int32)
                np.clip(x_columns, 0, width - 1, out=x_columns)
                np.maximum.at(column_values, x_columns, np.asarray(values[:n_bins], dtype=np.float64))
            for col, value in enumerate(column_values):
                if value <= 0:
                    continue
                x = int(rect.left()) + col
                y = baseline - int((float(value) / max_count) * height)
                y = max(y, top)
                painter.drawLine(x, baseline, x, y)
            marker_pen = QPen(color, 2)
            painter.setPen(marker_pen)
            for edge_value in value_range:
                x = _x_for_value(edge_value, rect, bin_edges)
                painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(x - 4, baseline - 4, 8, 8)
                painter.setPen(marker_pen)
            axis_min = float(bin_edges[0])
            axis_max = float(bin_edges[-1]) if float(bin_edges[-1]) > axis_min else axis_min + 1.0
            axis_label_rect = rect.adjusted(4, 0, -4, 0)
            painter.setPen(QColor("#d7dce1"))
            painter.drawText(
                axis_label_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                f"{axis_min:.1f}",
            )
            painter.drawText(
                axis_label_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                f"{axis_max:.1f}",
            )

        r, g, b = self._channel_color
        if any_original:
            _draw_histogram(
                original_rect,
                "Original",
                self._original_bin_edges,
                y_original,
                QColor(r, g, b, 220),
                2,
                self._original_range,
                max_count_original,
            )
        else:
            painter.fillRect(original_rect.adjusted(1, 1, -1, -1), QColor("#0b0c0e"))
            painter.setPen(QPen(QColor("#343a40")))
            painter.drawRect(original_rect)
            painter.setPen(QColor("#70757c"))
            painter.drawText(original_rect, Qt.AlignmentFlag.AlignCenter, "Original: no data")
        if any_deconvolved:
            _draw_histogram(
                deconvolved_rect,
                "Deconvolved",
                self._deconvolved_bin_edges,
                y_deconvolved,
                QColor(r, g, b, 220),
                2,
                self._deconvolved_range,
                max_count_deconvolved,
            )
        else:
            painter.fillRect(deconvolved_rect.adjusted(1, 1, -1, -1), QColor("#0b0c0e"))
            painter.setPen(QPen(QColor("#343a40")))
            painter.drawRect(deconvolved_rect)
            painter.setPen(QColor("#70757c"))
            painter.drawText(deconvolved_rect, Qt.AlignmentFlag.AlignCenter, "Deconvolved: no data")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        target = self._marker_at_position(event.position().x(), event.position().y())
        if target is not None:
            self._drag_target = target
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_target is not None:
            pane, edge = self._drag_target
            value = self._value_for_position(pane, event.position().x())
            self.markerDragged.emit(pane, edge, value)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_target = None
        super().mouseReleaseEvent(event)

    def _marker_at_position(self, x_pos: float, y_pos: float) -> tuple[str, str] | None:
        for pane, rect in self._plot_rects.items():
            if not rect.contains(x_pos, y_pos):
                continue
            value_range = self._original_range if pane == "original" else self._deconvolved_range
            for edge, value in (("min", value_range[0]), ("max", value_range[1])):
                marker_x = self._value_for_marker_x(pane, value)
                if abs(marker_x - x_pos) <= 8:
                    return pane, edge
        return None

    def _value_for_marker_x(self, pane: str, value: float) -> float:
        rect = self._plot_rects.get(pane)
        if rect is None:
            return 0.0
        bin_edges = self._original_bin_edges if pane == "original" else self._deconvolved_bin_edges
        axis_min = float(bin_edges[0])
        axis_max = float(bin_edges[-1]) if float(bin_edges[-1]) > axis_min else axis_min + 1.0
        pct = (float(value) - axis_min) / max(axis_max - axis_min, 1e-12)
        pct = min(max(pct, 0.0), 1.0)
        return rect.left() + pct * rect.width()

    def _value_for_position(self, pane: str, x_pos: float) -> float:
        rect = self._plot_rects.get(pane)
        if rect is None:
            return 0.0
        bin_edges = self._original_bin_edges if pane == "original" else self._deconvolved_bin_edges
        axis_min = float(bin_edges[0])
        axis_max = float(bin_edges[-1]) if float(bin_edges[-1]) > axis_min else axis_min + 1.0
        pct = (float(x_pos) - rect.left()) / max(rect.width(), 1e-12)
        pct = min(max(pct, 0.0), 1.0)
        return axis_min + pct * (axis_max - axis_min)


class _ChannelButton(QPushButton):
    """Channel toggle button: left click toggles, right click picks colour."""

    colorPickRequested = pyqtSignal(int)   # emits the channel index

    def __init__(self, label: str, channel_index: int, parent=None):
        super().__init__(label, parent)
        self._channel_index = channel_index

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            if self.rect().contains(event.position().toPoint()):
                self.setDown(False)
                self.colorPickRequested.emit(self._channel_index)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self.setDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _ChannelTable(QTableWidget):
    """Channel table where right-click requests a colour picker."""

    colorPickRequested = pyqtSignal(int)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            row = self.rowAt(event.position().toPoint().y())
            if row >= 0:
                self.selectRow(row)
                self.colorPickRequested.emit(row)
            event.accept()
            return
        super().mousePressEvent(event)


class _AdvancedScalingWindow(QWidget):
    closed = pyqtSignal()
    channelVisibilityChanged = pyqtSignal(int, bool)
    channelColorChanged = pyqtSignal(int, object)
    channelLevelsChanged = pyqtSignal(int, object)
    autoRequested = pyqtSignal(int)
    resetRequested = pyqtSignal(int)

    _SLIDER_STEPS = 1000

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Advanced Scaling")
        self.resize(420, 680)

        self._updating = False
        self._channel_names: list[str] = []
        self._channel_colors: list[tuple[int, int, int]] = []
        self._channel_visible: list[bool] = []
        self._channel_levels: list[dict[str, object]] = []
        self._channel_maxima: list[dict[str, float]] = []
        self._histograms: list[dict[str, object]] = []
        self._current_range_max: dict[str, float] = {"original": 1.0, "deconvolved": 1.0}
        self._pending_levels_index: Optional[int] = None
        self._levels_emit_timer = QTimer(self)
        self._levels_emit_timer.setSingleShot(True)
        self._levels_emit_timer.setInterval(120)
        self._levels_emit_timer.timeout.connect(self._emit_pending_levels)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self._table = _ChannelTable(0, 2)
        self._table.setHorizontalHeaderLabels(["Channel", "Show"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.colorPickRequested.connect(self._on_channel_color_requested)
        root.addWidget(self._table, stretch=1)

        self._log_hist_check = QCheckBox("Log histogram")
        self._log_hist_check.setChecked(True)
        self._log_hist_check.toggled.connect(self._refresh_histogram)
        root.addWidget(self._log_hist_check)

        legend = QHBoxLayout()
        self._original_legend = QLabel("Original")
        self._original_legend.setStyleSheet("font-weight: 700;")
        legend.addWidget(self._original_legend)
        self._deconvolved_legend = QLabel("Deconvolved")
        self._deconvolved_legend.setStyleSheet("font-weight: 700; color: #f3f4f6;")
        legend.addWidget(self._deconvolved_legend)
        legend.addStretch()
        root.addLayout(legend)

        self._histogram = _HistogramWidget()
        self._histogram.markerDragged.connect(self._on_histogram_marker_dragged)
        root.addWidget(self._histogram)

        self._selected_label = QLabel("No channel selected")
        self._selected_label.setStyleSheet("font-weight: 700;")
        root.addWidget(self._selected_label)

        root.addLayout(
            self._build_value_row(
                "Original min:",
                "_original_min_slider",
                "_original_min_spin",
                self._on_original_min_slider,
                self._on_original_min_spin,
            )
        )
        root.addLayout(
            self._build_value_row(
                "Original max:",
                "_original_max_slider",
                "_original_max_spin",
                self._on_original_max_slider,
                self._on_original_max_spin,
            )
        )
        root.addLayout(
            self._build_value_row(
                "Deconv min:",
                "_deconv_min_slider",
                "_deconv_min_spin",
                self._on_deconv_min_slider,
                self._on_deconv_min_spin,
            )
        )
        root.addLayout(
            self._build_value_row(
                "Deconv max:",
                "_deconv_max_slider",
                "_deconv_max_spin",
                self._on_deconv_max_slider,
                self._on_deconv_max_spin,
            )
        )
        root.addLayout(self._build_gamma_row())

        buttons = QHBoxLayout()
        self._auto_button = QPushButton("Auto")
        self._auto_button.clicked.connect(self._on_auto_clicked)
        buttons.addWidget(self._auto_button)
        self._reset_button = QPushButton("Reset")
        self._reset_button.clicked.connect(self._on_reset_clicked)
        buttons.addWidget(self._reset_button)
        buttons.addStretch()
        self._close_button = QPushButton("Close")
        self._close_button.clicked.connect(self.close)
        buttons.addWidget(self._close_button)
        root.addLayout(buttons)

    def _build_value_row(self, label_text: str, slider_name: str, spin_name: str, slider_slot, spin_slot) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(8)
        label = QLabel(label_text)
        label.setMinimumWidth(88)
        layout.addWidget(label)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, self._SLIDER_STEPS)
        slider.valueChanged.connect(slider_slot)
        setattr(self, slider_name, slider)
        layout.addWidget(slider, stretch=1)
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setSingleStep(1.0)
        spin.setRange(0.0, 1.0)
        spin.valueChanged.connect(spin_slot)
        setattr(self, spin_name, spin)
        layout.addWidget(spin)
        return layout

    def _build_gamma_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(8)
        label = QLabel("Viewer gamma:")
        label.setMinimumWidth(88)
        layout.addWidget(label)
        self._gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self._gamma_slider.setRange(10, 500)
        self._gamma_slider.valueChanged.connect(self._on_gamma_slider)
        layout.addWidget(self._gamma_slider, stretch=1)
        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setDecimals(2)
        self._gamma_spin.setRange(0.10, 5.00)
        self._gamma_spin.setSingleStep(0.05)
        self._gamma_spin.valueChanged.connect(self._on_gamma_spin)
        layout.addWidget(self._gamma_spin)
        return layout

    def set_channel_data(
        self,
        names: list[str],
        colors: list[tuple[int, int, int]],
        visible: list[bool],
        levels: list[dict[str, object]],
        maxima: list[dict[str, float]],
        histograms: list[dict[str, object]],
    ) -> None:
        selected_row = self.current_channel()
        if selected_row < 0:
            selected_row = 0

        self._channel_names = list(names)
        self._channel_colors = list(colors)
        self._channel_visible = list(visible)
        self._channel_levels = [dict(level) for level in levels]
        self._channel_maxima = [dict(v) for v in maxima]
        self._histograms = [dict(v) for v in histograms]

        self._updating = True
        self._table.blockSignals(True)
        self._table.setRowCount(len(names))
        for row, name in enumerate(names):
            name_item = QTableWidgetItem(name)
            name_item.setIcon(self._color_icon(colors[row]))
            name_item.setData(Qt.ItemDataRole.ForegroundRole, QColor(*colors[row]))
            self._table.setItem(row, 0, name_item)

            show_item = self._table.item(row, 1)
            if show_item is None:
                show_item = QTableWidgetItem("")
                show_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                self._table.setItem(row, 1, show_item)
            show_item.setCheckState(
                Qt.CheckState.Checked if visible[row] else Qt.CheckState.Unchecked
            )
        self._table.blockSignals(False)
        self._updating = False

        if names:
            selected_row = max(0, min(selected_row, len(names) - 1))
            self._table.selectRow(selected_row)
        self._refresh_selected_channel()

    def set_channel_visibility(self, index: int, visible: bool) -> None:
        if index < 0 or index >= self._table.rowCount():
            return
        item = self._table.item(index, 1)
        if item is None:
            return
        self._updating = True
        item.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
        self._updating = False
        if index < len(self._channel_visible):
            self._channel_visible[index] = bool(visible)

    def set_channel_color(self, index: int, color: tuple[int, int, int]) -> None:
        if index < 0 or index >= self._table.rowCount():
            return
        rgb = tuple(int(v) for v in color)
        if index < len(self._channel_colors):
            self._channel_colors[index] = rgb
        item = self._table.item(index, 0)
        if item is not None:
            item.setIcon(self._color_icon(rgb))
            item.setData(Qt.ItemDataRole.ForegroundRole, QColor(*rgb))
        self._refresh_histogram()

    def current_channel(self) -> int:
        selection_model = self._table.selectionModel()
        indexes = selection_model.selectedRows() if selection_model else []
        if not indexes:
            return -1
        return int(indexes[0].row())

    def closeEvent(self, event) -> None:  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)

    def _color_icon(self, color: tuple[int, int, int]) -> QIcon:
        pix = QPixmap(12, 12)
        pix.fill(QColor(*color))
        return QIcon(pix)

    def _value_to_slider(self, value: float) -> int:
        return self._value_to_slider_for_max(value, 1.0)

    def _value_to_slider_for_max(self, value: float, range_max: float) -> int:
        if range_max <= 0:
            return 0
        return int(round((float(value) / range_max) * self._SLIDER_STEPS))

    def _slider_to_value(self, slider_value: int) -> float:
        return self._slider_to_value_for_max(slider_value, 1.0)

    def _slider_to_value_for_max(self, slider_value: int, range_max: float) -> float:
        if range_max <= 0:
            return 0.0
        return float(slider_value) / self._SLIDER_STEPS * range_max

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._log_hist_check,
            self._gamma_slider,
            self._gamma_spin,
            self._auto_button,
            self._reset_button,
        ):
            widget.setEnabled(enabled)

    def _refresh_selected_channel(self) -> None:
        idx = self.current_channel()
        if idx < 0 or idx >= len(self._channel_levels):
            self._selected_label.setText("No channel selected")
            self._set_controls_enabled(False)
            self._set_range_controls_enabled("original", False)
            self._set_range_controls_enabled("deconvolved", False)
            return

        self._set_controls_enabled(True)
        self._selected_label.setText(self._channel_names[idx])
        maxima = self._channel_maxima[idx] if idx < len(self._channel_maxima) else {}
        original_range_max = max(float(maxima.get("original", 0.0)), 1.0)
        deconvolved_range_max = max(float(maxima.get("deconvolved", 0.0)), 1.0)
        self._current_range_max = {
            "original": original_range_max,
            "deconvolved": deconvolved_range_max,
        }

        levels = self._normalize_levels(idx)
        self._updating = True
        self._apply_range_controls("original", levels["original"], original_range_max)
        self._apply_range_controls("deconvolved", levels["deconvolved"], deconvolved_range_max)
        self._gamma_spin.setValue(float(levels["gamma"]))
        self._gamma_slider.setValue(int(round(float(levels["gamma"]) * 100)))
        self._updating = False

        histogram = self._histograms[idx] if idx < len(self._histograms) else {}
        has_original = bool(histogram.get("has_original", False))
        has_deconvolved = bool(histogram.get("has_deconvolved", False))
        self._set_range_controls_enabled("original", has_original)
        self._set_range_controls_enabled("deconvolved", has_deconvolved)
        self._refresh_histogram()

    def _refresh_histogram(self) -> None:
        idx = self.current_channel()
        if idx < 0 or idx >= len(self._histograms):
            return
        histogram = self._histograms[idx]
        levels = self._normalize_levels(idx)
        color = self._channel_colors[idx] if idx < len(self._channel_colors) else (255, 255, 255)
        legend_style = f"font-weight: 700; color: rgb({color[0]},{color[1]},{color[2]});"
        self._original_legend.setStyleSheet(
            legend_style
            if bool(histogram.get("has_original", False))
            else "font-weight: 700; color: #70757c;"
        )
        self._deconvolved_legend.setStyleSheet(
            legend_style
            if bool(histogram.get("has_deconvolved", False))
            else "font-weight: 700; color: #70757c;"
        )
        self._histogram.set_histogram(
            np.asarray(histogram.get("original_bin_edges", np.array([0.0, 1.0], dtype=np.float64))),
            np.asarray(histogram.get("deconvolved_bin_edges", np.array([0.0, 1.0], dtype=np.float64))),
            np.asarray(histogram.get("original", np.zeros(1, dtype=np.float64))),
            np.asarray(histogram.get("deconvolved", np.zeros(1, dtype=np.float64))),
            color,
            (
                float(levels["original"]["min"]),
                float(levels["original"]["max"]),
            ),
            (
                float(levels["deconvolved"]["min"]),
                float(levels["deconvolved"]["max"]),
            ),
            has_original=bool(histogram.get("has_original", False)),
            has_deconvolved=bool(histogram.get("has_deconvolved", False)),
            log_scale=self._log_hist_check.isChecked(),
        )

    def _apply_selected_levels(
        self,
        *,
        pane: Optional[str] = None,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        gamma: Optional[float] = None,
        emit: bool = True,
    ) -> None:
        idx = self.current_channel()
        if idx < 0 or idx >= len(self._channel_levels):
            return
        levels = self._normalize_levels(idx)
        original = dict(levels["original"])
        deconvolved = dict(levels["deconvolved"])
        gamma_val = float(levels["gamma"] if gamma is None else gamma)

        if pane == "original":
            lo_val = float(original["min"] if lo is None else lo)
            hi_val = float(original["max"] if hi is None else hi)
            pane_max = self._current_range_max["original"]
            lo_val = min(max(lo_val, 0.0), pane_max)
            hi_val = min(max(hi_val, lo_val), pane_max)
            original = {"min": lo_val, "max": hi_val}
        elif pane == "deconvolved":
            lo_val = float(deconvolved["min"] if lo is None else lo)
            hi_val = float(deconvolved["max"] if hi is None else hi)
            pane_max = self._current_range_max["deconvolved"]
            lo_val = min(max(lo_val, 0.0), pane_max)
            hi_val = min(max(hi_val, lo_val), pane_max)
            deconvolved = {"min": lo_val, "max": hi_val}

        gamma_val = min(max(gamma_val, 0.10), 5.00)
        self._channel_levels[idx] = {
            "original": original,
            "deconvolved": deconvolved,
            "gamma": gamma_val,
        }

        self._updating = True
        self._apply_range_controls("original", original, self._current_range_max["original"])
        self._apply_range_controls("deconvolved", deconvolved, self._current_range_max["deconvolved"])
        self._gamma_spin.setValue(gamma_val)
        self._gamma_slider.setValue(int(round(gamma_val * 100)))
        self._updating = False
        self._refresh_histogram()
        if emit:
            self._pending_levels_index = idx
            self._levels_emit_timer.start()

    def _emit_pending_levels(self) -> None:
        idx = self._pending_levels_index
        self._pending_levels_index = None
        if idx is None or idx < 0 or idx >= len(self._channel_levels):
            return
        levels = self._normalize_levels(idx)
        self.channelLevelsChanged.emit(
            idx,
            {
                "original": dict(levels["original"]),
                "deconvolved": dict(levels["deconvolved"]),
                "gamma": float(levels["gamma"]),
            },
        )

    def _normalize_levels(self, index: int) -> dict[str, object]:
        maxima = self._channel_maxima[index] if index < len(self._channel_maxima) else {}
        original_max = max(float(maxima.get("original", 0.0)), 0.0)
        deconvolved_max = max(float(maxima.get("deconvolved", 0.0)), 0.0)

        levels = dict(self._channel_levels[index]) if index < len(self._channel_levels) else {}
        original = dict(levels.get("original") or {"min": 0.0, "max": original_max})
        deconvolved = dict(levels.get("deconvolved") or {"min": 0.0, "max": deconvolved_max})
        gamma = min(max(float(levels.get("gamma", 1.0)), 0.10), 5.00)

        original["min"] = min(max(float(original.get("min", 0.0)), 0.0), original_max)
        original["max"] = min(max(float(original.get("max", original_max)), original["min"]), original_max)
        deconvolved["min"] = min(max(float(deconvolved.get("min", 0.0)), 0.0), deconvolved_max)
        deconvolved["max"] = min(max(float(deconvolved.get("max", deconvolved_max)), deconvolved["min"]), deconvolved_max)

        normalized = {
            "original": {"min": float(original["min"]), "max": float(original["max"])},
            "deconvolved": {"min": float(deconvolved["min"]), "max": float(deconvolved["max"])},
            "gamma": gamma,
        }
        if index < len(self._channel_levels):
            self._channel_levels[index] = normalized
        return normalized

    def _apply_range_controls(self, pane: str, values: dict[str, float], range_max: float) -> None:
        min_slider = getattr(self, f"_{'original' if pane == 'original' else 'deconv'}_min_slider")
        max_slider = getattr(self, f"_{'original' if pane == 'original' else 'deconv'}_max_slider")
        min_spin = getattr(self, f"_{'original' if pane == 'original' else 'deconv'}_min_spin")
        max_spin = getattr(self, f"_{'original' if pane == 'original' else 'deconv'}_max_spin")
        min_spin.setRange(0.0, range_max)
        max_spin.setRange(0.0, range_max)
        min_spin.setValue(float(values["min"]))
        max_spin.setValue(float(values["max"]))
        min_slider.setValue(self._value_to_slider_for_max(float(values["min"]), range_max))
        max_slider.setValue(self._value_to_slider_for_max(float(values["max"]), range_max))

    def _set_range_controls_enabled(self, pane: str, enabled: bool) -> None:
        prefix = "_original" if pane == "original" else "_deconv"
        for suffix in ("_min_slider", "_min_spin", "_max_slider", "_max_spin"):
            getattr(self, f"{prefix}{suffix}").setEnabled(enabled)

    def _on_selection_changed(self) -> None:
        if self._updating:
            return
        self._refresh_selected_channel()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != 1:
            return
        visible = item.checkState() == Qt.CheckState.Checked
        row = item.row()
        if row < len(self._channel_visible):
            self._channel_visible[row] = visible
        self.channelVisibilityChanged.emit(row, visible)

    def _on_channel_color_requested(self, row: int) -> None:
        if row < 0 or row >= len(self._channel_colors):
            return
        color = QColorDialog.getColor(QColor(*self._channel_colors[row]), self, "Select channel color")
        if not color.isValid():
            return
        self.channelColorChanged.emit(row, (color.red(), color.green(), color.blue()))

    def _on_original_min_slider(self, value: int) -> None:
        if self._updating:
            return
        self._apply_selected_levels(
            pane="original",
            lo=self._slider_to_value_for_max(value, self._current_range_max["original"]),
        )

    def _on_original_min_spin(self, value: float) -> None:
        if self._updating:
            return
        self._apply_selected_levels(pane="original", lo=float(value))

    def _on_original_max_slider(self, value: int) -> None:
        if self._updating:
            return
        self._apply_selected_levels(
            pane="original",
            hi=self._slider_to_value_for_max(value, self._current_range_max["original"]),
        )

    def _on_original_max_spin(self, value: float) -> None:
        if self._updating:
            return
        self._apply_selected_levels(pane="original", hi=float(value))

    def _on_deconv_min_slider(self, value: int) -> None:
        if self._updating:
            return
        self._apply_selected_levels(
            pane="deconvolved",
            lo=self._slider_to_value_for_max(value, self._current_range_max["deconvolved"]),
        )

    def _on_deconv_min_spin(self, value: float) -> None:
        if self._updating:
            return
        self._apply_selected_levels(pane="deconvolved", lo=float(value))

    def _on_deconv_max_slider(self, value: int) -> None:
        if self._updating:
            return
        self._apply_selected_levels(
            pane="deconvolved",
            hi=self._slider_to_value_for_max(value, self._current_range_max["deconvolved"]),
        )

    def _on_deconv_max_spin(self, value: float) -> None:
        if self._updating:
            return
        self._apply_selected_levels(pane="deconvolved", hi=float(value))

    def _on_gamma_slider(self, value: int) -> None:
        if self._updating:
            return
        self._apply_selected_levels(gamma=float(value) / 100.0)

    def _on_gamma_spin(self, value: float) -> None:
        if self._updating:
            return
        self._apply_selected_levels(gamma=float(value))

    def _on_auto_clicked(self) -> None:
        idx = self.current_channel()
        if idx >= 0:
            self.autoRequested.emit(idx)

    def _on_reset_clicked(self) -> None:
        idx = self.current_channel()
        if idx >= 0:
            self.resetRequested.emit(idx)

    def _on_histogram_marker_dragged(self, pane: str, edge: str, value: float) -> None:
        if pane == "original":
            self._apply_selected_levels(pane="original", lo=value if edge == "min" else None, hi=value if edge == "max" else None)
        elif pane == "deconvolved":
            self._apply_selected_levels(pane="deconvolved", lo=value if edge == "min" else None, hi=value if edge == "max" else None)


class DualViewerWidget(QWidget):
    timepointChanged = pyqtSignal(int)
    logRequested = pyqtSignal()
    cursorInfoChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metadata: dict = {}
        self._input_channels: list[np.ndarray] = []
        self._loaded_input_timepoint: Optional[int] = None
        self._tiled_provider = None
        self._tiled_item = None
        self._preview_by_t: dict[int, list[np.ndarray]] = {}
        self._channel_buttons: list[QPushButton] = []
        self._channel_colors: list[tuple[int, int, int]] = []
        self._channel_scaling: list[dict[str, float]] = []
        self._channel_scaling_projection: Optional[str] = None
        self._advanced_scaling_window: Optional[_AdvancedScalingWindow] = None
        self._advanced_scaling_active = False
        self._available_volume_methods = list(_VOLUME_METHODS)
        self._active_volume_method = _VOLUME_METHODS[0]
        self._volume_method_values = {
            method: spec["default"] for method, spec in _VOLUME_METHOD_UI.items()
        }
        self._syncing_camera = False
        self._reset_3d_on_next_render = False
        self._fit_on_next_render = False
        self._smooth_zoom_enabled = False
        self._scale_bar_enabled = True
        self._navigator_enabled = True
        self._blink_show_deconvolved = False
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(550)
        self._blink_timer.timeout.connect(self._on_blink_timer)
        # Histogram cache: id(numpy_array) -> (bin_edges, counts).
        # Cleared when new image data is loaded so stale entries don't accumulate.
        self._hist_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._max_cache: dict[int, float] = {}
        # SUM projection cache: id(stack) -> precomputed 2-D sum plane.
        # Held here so the array stays alive and its id() is stable for
        # _hist_cache keying (SUM values are n_z× larger than stack values
        # and need their own histogram range).
        self._sum_cache: dict[int, np.ndarray] = {}
        # Debounce timer for 2-D contrast changes: only redraw 300 ms after the
        # last spin-box change, avoiding expensive re-renders on every keystroke.
        self._contrast_2d_timer = QTimer(self)
        self._contrast_2d_timer.setSingleShot(True)
        self._contrast_2d_timer.setInterval(300)
        self._contrast_2d_timer.timeout.connect(self._refresh_view)
        self._refresh_3d_timer = QTimer(self)
        self._refresh_3d_timer.setSingleShot(True)
        self._refresh_3d_timer.setInterval(150)
        self._refresh_3d_timer.timeout.connect(self._refresh_view)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._build_toolbar(root)
        self._build_panes(root)
        self._refresh_view()

    def _build_toolbar(self, root: QVBoxLayout) -> None:
        self._channel_bar = QHBoxLayout()
        self._channel_bar.addWidget(QLabel("Channels:"))
        self._channel_bar.addStretch()
        root.addLayout(self._channel_bar)

        top = QHBoxLayout()
        top.setSpacing(10)

        def _group(icon_kind: str, tooltip: str) -> QWidget:
            group = QFrame()
            group.setFrameShape(QFrame.Shape.StyledPanel)
            group.setStyleSheet(
                "QFrame { border: 1px solid #343a40; border-radius: 4px; } "
                "QLabel { border: none; color: #bfc5cc; font-weight: 700; } "
                "QComboBox, QPushButton { border: 1px solid #3d4248; }"
            )
            layout = QHBoxLayout(group)
            layout.setContentsMargins(6, 3, 6, 3)
            layout.setSpacing(6)
            icon = QLabel()
            icon.setPixmap(_toolbar_icon(icon_kind))
            icon.setToolTip(tooltip)
            layout.addWidget(icon)
            return group

        view_section = _group("eye", "View mode")
        top.addWidget(view_section)
        view_section_layout = view_section.layout()
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([_TWO_D_MODE, _THREE_D_MODE])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._mode_combo.hide()

        show_group = QWidget()
        show_layout = QHBoxLayout(show_group)
        show_layout.setContentsMargins(0, 0, 0, 0)
        show_layout.setSpacing(6)
        self._view_selector = QComboBox()
        self._view_selector.addItems([
            "2D Both",
            "2D Original",
            "2D Deconvolved",
            "2D Linked split",
            "2D Blink",
            "2D Difference",
            "2D Ratio",
            "3D Both",
            "3D Original",
            "3D Deconvolved",
        ])
        self._view_selector.currentTextChanged.connect(self._on_view_mode_changed)
        self._view_selector.setMinimumContentsLength(16)
        self._view_selector.setStyleSheet("QComboBox { padding-left: 5px; padding-right: 8px; }")
        show_layout.addWidget(self._view_selector)
        self._compare_combo = self._view_selector
        view_section_layout.addWidget(show_group)

        view_group = QWidget()
        view_layout = QHBoxLayout(view_group)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(6)
        projection_icon = QLabel()
        projection_icon.setPixmap(_toolbar_icon("projection"))
        projection_icon.setToolTip("2D projection")
        view_layout.addWidget(projection_icon)
        self._projection_combo = QComboBox()
        self._projection_combo.addItems(_PROJECTION_MODES)
        self._projection_combo.currentTextChanged.connect(self._on_projection_changed)
        self._projection_combo.setMinimumContentsLength(6)
        self._projection_combo.setStyleSheet("QComboBox { padding-left: 5px; padding-right: 8px; }")
        view_layout.addWidget(self._projection_combo)
        view_section_layout.addWidget(view_group)

        self._fit_button = QPushButton("Fit")
        self._fit_button.setMinimumWidth(54)
        self._fit_button.clicked.connect(self.fit_views)
        view_section_layout.addWidget(self._fit_button)

        display_section = _group("display", "Display")
        top.addWidget(display_section)
        display_layout = display_section.layout()
        self._smooth_zoom_check = QCheckBox("Smooth zoom")
        self._smooth_zoom_check.setChecked(False)
        self._smooth_zoom_check.toggled.connect(self._on_smooth_zoom_toggled)
        display_layout.addWidget(self._smooth_zoom_check)

        self._navigator_check = QCheckBox("Navigator")
        self._navigator_check.setChecked(True)
        self._navigator_check.toggled.connect(self._on_navigator_toggled)
        display_layout.addWidget(self._navigator_check)

        self._scale_bar_check = QCheckBox("Scale bar")
        self._scale_bar_check.setChecked(True)
        self._scale_bar_check.toggled.connect(self._on_scale_bar_toggled)
        display_layout.addWidget(self._scale_bar_check)

        self._lo_group = QWidget()
        lo_layout = QHBoxLayout(self._lo_group)
        lo_layout.setContentsMargins(0, 0, 0, 0)
        lo_layout.setSpacing(6)
        lo_layout.addWidget(QLabel("Lo%:"))
        self._lo_spin = QDoubleSpinBox()
        self._lo_spin.setRange(0.0, 50.0)
        self._lo_spin.setDecimals(3)
        self._lo_spin.setSingleStep(0.01)
        self._lo_spin.setValue(0.1)
        self._lo_spin.setMinimumWidth(95)
        self._lo_spin.valueChanged.connect(self._on_contrast_changed)
        lo_layout.addWidget(self._lo_spin)
        display_layout.addWidget(self._lo_group)

        self._hi_group = QWidget()
        hi_layout = QHBoxLayout(self._hi_group)
        hi_layout.setContentsMargins(0, 0, 0, 0)
        hi_layout.setSpacing(6)
        hi_layout.addWidget(QLabel("Hi%:"))
        self._hi_spin = QDoubleSpinBox()
        self._hi_spin.setRange(50.0, 100.0)
        self._hi_spin.setDecimals(3)
        self._hi_spin.setSingleStep(0.001)
        self._hi_spin.setValue(100.0)
        self._hi_spin.setMinimumWidth(95)
        self._hi_spin.valueChanged.connect(self._on_contrast_changed)
        hi_layout.addWidget(self._hi_spin)
        display_layout.addWidget(self._hi_group)

        self._advanced_scaling_button = QPushButton("Adv. Scaling")
        self._advanced_scaling_button.setMinimumWidth(96)
        self._advanced_scaling_button.clicked.connect(self._open_advanced_scaling)
        display_layout.addWidget(self._advanced_scaling_button)

        self._compare_split_slider = QSlider(Qt.Orientation.Horizontal)
        self._compare_split_slider.setRange(5, 95)
        self._compare_split_slider.setValue(50)
        self._compare_split_slider.setFixedWidth(90)
        self._compare_split_slider.valueChanged.connect(self._refresh_view)
        self._compare_split_slider.setVisible(False)
        view_section_layout.addWidget(self._compare_split_slider)

        top.addStretch()

        root.addLayout(top)

        self._bar_3d = QWidget()
        bar_3d_layout = QHBoxLayout(self._bar_3d)
        bar_3d_layout.setContentsMargins(0, 0, 0, 0)
        bar_3d_layout.setSpacing(8)
        bar_3d_layout.addWidget(QLabel("Render:"))
        self._volume_method_combo = QComboBox()
        self._volume_method_combo.currentIndexChanged.connect(self._on_volume_method_changed)
        bar_3d_layout.addWidget(self._volume_method_combo)
        self._volume_slider_label = QLabel("Gain:")
        bar_3d_layout.addWidget(self._volume_slider_label)
        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.valueChanged.connect(self._on_volume_slider_changed)
        bar_3d_layout.addWidget(self._volume_slider)
        self._volume_slider_value = QLabel("1.00")
        self._volume_slider_value.setMinimumWidth(40)
        bar_3d_layout.addWidget(self._volume_slider_value)
        bar_3d_layout.addWidget(QLabel("Downsample:"))
        self._downsample_combo = QComboBox()
        self._downsample_combo.addItems(["1x", "2x", "4x"])
        self._downsample_combo.currentIndexChanged.connect(self._schedule_3d_refresh)
        bar_3d_layout.addWidget(self._downsample_combo)
        self._smooth_check = QCheckBox("Smooth")
        self._smooth_check.setChecked(True)
        self._smooth_check.toggled.connect(self._schedule_3d_refresh)
        bar_3d_layout.addWidget(self._smooth_check)
        self._reset_3d_button = QPushButton("Reset View")
        self._reset_3d_button.clicked.connect(self._reset_3d_views)
        bar_3d_layout.addWidget(self._reset_3d_button)
        bar_3d_layout.addStretch()
        root.addWidget(self._bar_3d)

        self._refresh_volume_method_options()
        self._bar_3d.setVisible(False)

    def _build_panes(self, root: QVBoxLayout) -> None:
        body = QHBoxLayout()
        body.setSpacing(10)

        panes = QHBoxLayout()
        panes.setSpacing(10)
        self._input_pane = _PaneWidget("Original")
        self._output_pane = _PaneWidget("Deconvolved")
        self._input_pane.view2d.link_to(self._output_pane.view2d)
        self._input_pane.view2d.cursorMoved.connect(
            lambda x, y, inside: self._on_cursor_moved("Original", x, y, inside)
        )
        self._input_pane.view2d.rightSplitDragged.connect(self._on_split_dragged)
        self._input_pane.view2d.panDragStarted.connect(self._set_pan_drag_cursor)
        self._input_pane.view2d.panDragFinished.connect(self._refresh_split_cursor)
        self._output_pane.view2d.cursorMoved.connect(
            lambda x, y, inside: self._on_cursor_moved("Deconvolved", x, y, inside)
        )
        self._output_pane.view2d.rightSplitDragged.connect(self._on_split_dragged)
        self._output_pane.view2d.panDragStarted.connect(self._set_pan_drag_cursor)
        self._output_pane.view2d.panDragFinished.connect(self._refresh_split_cursor)
        self._input_pane.cameraStateChanged.connect(self._sync_3d_from_input)
        self._output_pane.cameraStateChanged.connect(self._sync_3d_from_output)
        panes.addWidget(self._input_pane, stretch=1)
        panes.addWidget(self._output_pane, stretch=1)
        body.addLayout(panes, stretch=1)

        z_panel = QWidget()
        z_layout = QVBoxLayout(z_panel)
        z_layout.setContentsMargins(0, 0, 0, 0)
        z_layout.setSpacing(6)
        z_title = QLabel("Z:")
        z_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_layout.addWidget(z_title)
        self._z_slider = QSlider(Qt.Orientation.Vertical)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.valueChanged.connect(self._refresh_view)
        z_layout.addWidget(self._z_slider, stretch=1)
        self._z_label = QLabel("0/0")
        self._z_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_layout.addWidget(self._z_label)
        body.addWidget(z_panel)

        root.addLayout(body, stretch=1)

        self._time_bar = QWidget()
        time_layout = QHBoxLayout(self._time_bar)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(8)
        time_layout.addWidget(QLabel("T:"))
        self._t_slider = QSlider(Qt.Orientation.Horizontal)
        self._t_slider.setMinimum(0)
        self._t_slider.setMaximum(0)
        self._t_slider.valueChanged.connect(self._on_time_changed)
        time_layout.addWidget(self._t_slider, stretch=1)
        self._t_label = QLabel("0/0")
        self._t_label.setMinimumWidth(50)
        time_layout.addWidget(self._t_label)
        root.addWidget(self._time_bar)

    def set_input_data(self, channels: list[np.ndarray], metadata: dict) -> None:
        self._tiled_provider = None
        self._tiled_item = None
        self._hist_cache.clear()
        self._max_cache.clear()
        self._sum_cache.clear()
        self._input_channels = channels
        self._loaded_input_timepoint = None
        self._metadata = metadata
        self._preview_by_t.clear()
        channels_meta = self._display_channels()
        self._channel_colors = _resolve_channel_colors(channels_meta)
        self._rebuild_channel_buttons(channels_meta)
        self._reset_channel_scaling()
        self._refresh_volume_method_options(channels_meta)
        size_t = max(int(metadata.get("size_t", 1)), 1)
        size_z = max(int(metadata.get("size_z", 1)), 1)
        default_t = max(0, min(int(metadata.get("default_t", 0)), size_t - 1))
        default_z = max(0, min(int(metadata.get("default_z", size_z // 2 if size_z > 1 else 0)), size_z - 1))
        self._t_slider.blockSignals(True)
        self._t_slider.setMaximum(size_t - 1)
        self._t_slider.setValue(default_t)
        self._t_slider.blockSignals(False)
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(size_z - 1)
        self._z_slider.setValue(default_z)
        self._z_slider.blockSignals(False)
        self._time_bar.setVisible(size_t > 1)
        self._update_labels()
        self._fit_on_next_render = True
        self._ensure_mode_valid()
        self._sync_advanced_scaling_window()
        self._refresh_view()

    def set_tiled_input_provider(self, provider) -> None:
        """Enable progressive pyramid rendering for the original pane."""
        try:
            from omero_browser_qt.omero_viewer import TiledImageItem
        except Exception:
            self._tiled_provider = None
            self._tiled_item = None
            return
        self._tiled_provider = provider
        self._tiled_item = TiledImageItem(provider)
        self._fit_on_next_render = True
        if self._mode_combo.currentText() == _THREE_D_MODE:
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentText(_TWO_D_MODE)
            self._mode_combo.blockSignals(False)
        self._refresh_view()

    def set_input_timepoint_data(self, timepoint: int, channels_zyx: list[np.ndarray]) -> None:
        self._hist_cache.clear()
        self._max_cache.clear()
        self._sum_cache.clear()
        self._input_channels = list(channels_zyx)
        self._loaded_input_timepoint = int(timepoint)
        self._ensure_channel_scaling_defaults()
        self._ensure_mode_valid()
        self._sync_advanced_scaling_window()
        self._refresh_view()

    def clear_preview_results(self) -> None:
        self._hist_cache.clear()
        self._max_cache.clear()
        self._sum_cache.clear()
        self._preview_by_t.clear()
        self._sync_advanced_scaling_window()
        self._refresh_view()

    def set_preview_result(self, timepoint: int, channels_zyx: list[np.ndarray]) -> None:
        # Evict cached histograms and SUM planes for the previous deconvolved
        # arrays at this timepoint so a rerun doesn't reuse stale data.
        old = self._preview_by_t.get(int(timepoint))
        if old:
            for arr in old:
                key = id(arr)
                self._hist_cache.pop(key, None)
                self._max_cache.pop(key, None)
                sum_plane = self._sum_cache.pop(key, None)
                if sum_plane is not None:
                    self._hist_cache.pop(id(sum_plane), None)
                    self._max_cache.pop(id(sum_plane), None)
        self._preview_by_t[int(timepoint)] = channels_zyx
        self._ensure_channel_scaling_defaults()
        self._sync_advanced_scaling_window()
        self._refresh_view()

    def current_timepoint(self) -> int:
        return self._t_slider.value()

    def has_time_axis(self) -> bool:
        return self._t_slider.maximum() > 0

    def set_timepoint(self, timepoint: int) -> None:
        timepoint = max(0, min(timepoint, self._t_slider.maximum()))
        self._t_slider.setValue(timepoint)

    def has_preview_for_timepoint(self, timepoint: int) -> bool:
        return int(timepoint) in self._preview_by_t

    def current_preview_channels(self) -> list[np.ndarray]:
        return list(self._preview_by_t.get(self.current_timepoint(), []))

    def lo_percentile(self) -> float:
        return float(self._lo_spin.value())

    def hi_percentile(self) -> float:
        return float(self._hi_spin.value())

    def movie_render_state(self) -> dict[str, object]:
        """Return a GUI-thread snapshot of the current 2-D rendering settings."""
        self._ensure_channel_scaling_defaults()
        return {
            "projection": self._projection_combo.currentText(),
            "z_index": int(self._z_slider.value()),
            "active_channels": self._active_channel_indices(),
            "channel_colors": list(self._channel_colors),
            "advanced_scaling_active": bool(self._advanced_scaling_active),
            "channel_scaling": [
                {
                    "original": dict(state.get("original") or {}),
                    "deconvolved": dict(state.get("deconvolved") or {}),
                    "gamma": float(state.get("gamma", 1.0)),
                }
                for state in self._channel_scaling
            ],
            "lo_percentile": float(self._lo_spin.value()),
            "hi_percentile": float(self._hi_spin.value()),
        }

    def current_view_images(self) -> dict[str, QImage]:
        """Return PNG-ready images matching the current Original/Deconvolved panes."""
        self._ensure_channel_scaling_defaults()
        mode = self._mode_combo.currentText()
        timepoint = self.current_timepoint()
        if mode == _THREE_D_MODE and self._can_show_3d():
            return {
                "original": self._input_pane.grab_3d_image(),
                "deconvolved": self._output_pane.grab_3d_image() if timepoint in self._preview_by_t else QImage(),
            }

        projection = self._projection_combo.currentText()
        original = self._build_2d_image(self._input_channels, projection, pane="original")
        preview = self._preview_by_t.get(timepoint)
        deconvolved = (
            self._build_2d_image(preview, projection, pane="deconvolved")
            if preview
            else QImage()
        )
        return {
            "original": original,
            "deconvolved": deconvolved,
        }

    def current_comparison_image(self) -> QImage:
        """Return the active display-normalized comparison image."""
        self._ensure_channel_scaling_defaults()
        if self._mode_combo.currentText() == _THREE_D_MODE:
            images = self.current_view_images()
            deconvolved = images.get("deconvolved") or QImage()
            if not deconvolved.isNull():
                return deconvolved
            return images.get("original") or QImage()
        preview = self._preview_by_t.get(self.current_timepoint())
        if not preview:
            return QImage()
        return _rgb_to_qimage(self._build_compare_rgb(preview, self._projection_combo.currentText()))

    def set_lo_percentile(self, value: float) -> None:
        self._lo_spin.setValue(value)

    def set_hi_percentile(self, value: float) -> None:
        self._hi_spin.setValue(value)

    def _on_smooth_zoom_toggled(self, enabled: bool) -> None:
        self._smooth_zoom_enabled = bool(enabled)
        self._input_pane.set_smooth_zoom(self._smooth_zoom_enabled)
        self._output_pane.set_smooth_zoom(self._smooth_zoom_enabled)

    def _on_scale_bar_toggled(self, enabled: bool) -> None:
        self._scale_bar_enabled = bool(enabled)
        self._refresh_scale_bar_state()

    def _on_navigator_toggled(self, enabled: bool) -> None:
        self._navigator_enabled = bool(enabled)
        self._refresh_navigator_state()

    def _view_mode_text(self) -> str:
        text = self._view_selector.currentText() if hasattr(self, "_view_selector") else "2D Both"
        if text.startswith("2D "):
            return text[3:]
        if text.startswith("3D "):
            return text[3:]
        return text

    def _view_dimension_text(self) -> str:
        text = self._view_selector.currentText() if hasattr(self, "_view_selector") else "2D Both"
        return _THREE_D_MODE if text.startswith("3D ") else _TWO_D_MODE

    def _on_view_mode_changed(self, text: str) -> None:
        requested_mode = _THREE_D_MODE if str(text).startswith("3D ") else _TWO_D_MODE
        if requested_mode == _THREE_D_MODE and not self._can_show_3d():
            self._view_selector.blockSignals(True)
            self._view_selector.setCurrentText("2D Both")
            self._view_selector.blockSignals(False)
            requested_mode = _TWO_D_MODE
        if self._mode_combo.currentText() != requested_mode:
            self._mode_combo.setCurrentText(requested_mode)
        self._on_compare_mode_changed(self._view_mode_text())
        self._apply_view_selector()
        self._refresh_split_cursor()
        self._refresh_view()

    def _on_compare_mode_changed(self, mode: str) -> None:
        mode = mode[3:] if mode.startswith(("2D ", "3D ")) else mode
        self._compare_split_slider.setVisible(mode == "Linked split")
        if mode == "Blink":
            self._blink_timer.start()
        else:
            self._blink_timer.stop()
            self._blink_show_deconvolved = False
        self._refresh_split_cursor()
        self._refresh_view()

    def _on_split_dragged(self, ratio: float) -> None:
        if self._mode_combo.currentText() != _TWO_D_MODE:
            return
        if self._view_mode_text() != "Linked split":
            return
        value = max(5, min(95, int(round(float(ratio) * 100.0))))
        if self._compare_split_slider.value() == value:
            return
        self._compare_split_slider.setValue(value)

    def _refresh_split_cursor(self) -> None:
        cursor = (
            Qt.CursorShape.SplitHCursor
            if self._mode_combo.currentText() == _TWO_D_MODE and self._view_mode_text() == "Linked split"
            else Qt.CursorShape.ArrowCursor
        )
        self._input_pane.view2d.set_interaction_cursor(cursor)
        self._output_pane.view2d.set_interaction_cursor(cursor)

    def _set_pan_drag_cursor(self) -> None:
        self._input_pane.view2d.set_interaction_cursor(Qt.CursorShape.ClosedHandCursor)
        self._output_pane.view2d.set_interaction_cursor(Qt.CursorShape.ClosedHandCursor)

    def _on_blink_timer(self) -> None:
        self._blink_show_deconvolved = not self._blink_show_deconvolved
        if self._mode_combo.currentText() == _TWO_D_MODE:
            self._refresh_view()

    def _pixel_size_x_um(self) -> Optional[float]:
        for key in ("pixel_size_x", "pixel_size_y"):
            value = self._metadata.get(key)
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if value_f > 0:
                return value_f
        for key in ("pixel_size_xy_nm", "pixel_size_nm"):
            value = self._metadata.get(key)
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if value_f > 0:
                return value_f / 1000.0
        return None

    def _display_um_per_scene_pixel(self, channels_zyx: Optional[list[np.ndarray]]) -> Optional[float]:
        px_um = self._pixel_size_x_um()
        if px_um is None or not channels_zyx:
            return None
        for idx in self._active_channel_indices():
            if 0 <= idx < len(channels_zyx):
                stack = channels_zyx[idx]
                if stack.ndim >= 2:
                    return px_um * _display_stride(tuple(int(v) for v in stack.shape[-2:]))
        for stack in channels_zyx:
            if stack.ndim >= 2:
                return px_um * _display_stride(tuple(int(v) for v in stack.shape[-2:]))
        return px_um

    def _refresh_scale_bar_state(self) -> None:
        input_um = (
            self._pixel_size_x_um()
            if self._tiled_item is not None
            else self._display_um_per_scene_pixel(self._input_channels)
        )
        preview = self._preview_by_t.get(self.current_timepoint())
        output_um = self._display_um_per_scene_pixel(preview)
        self._input_pane.set_scale_bar(self._scale_bar_enabled, input_um)
        self._output_pane.set_scale_bar(self._scale_bar_enabled, output_um)

    def fit_views(self) -> None:
        if self._mode_combo.currentText() == _THREE_D_MODE:
            self._reset_3d_views()
            return
        self._input_pane.fit_2d()
        self._output_pane.fit_2d()

    def refresh_view(self) -> None:
        self._refresh_view()

    def _reset_channel_scaling(self) -> None:
        self._channel_scaling = []
        self._channel_scaling_projection = self._active_scaling_projection()
        self._ensure_channel_scaling_defaults()

    def _ensure_channel_scaling_defaults(self) -> None:
        projection = self._active_scaling_projection()
        if self._channel_scaling_projection != projection:
            self._channel_scaling = []
            self._channel_scaling_projection = projection
        channel_count = len(self._display_channels())
        while len(self._channel_scaling) < channel_count:
            idx = len(self._channel_scaling)
            self._channel_scaling.append(
                {
                    "original": {
                        "min": 0.0,
                        "max": self._pane_channel_max_value(idx, "original", projection),
                    },
                    "deconvolved": {
                        "min": 0.0,
                        "max": self._pane_channel_max_value(idx, "deconvolved", projection),
                    },
                    "gamma": 1.0,
                }
            )
        if len(self._channel_scaling) > channel_count:
            self._channel_scaling = self._channel_scaling[:channel_count]
        for idx, state in enumerate(self._channel_scaling):
            original_max = self._pane_channel_max_value(idx, "original", projection)
            deconvolved_max = self._pane_channel_max_value(idx, "deconvolved", projection)
            if not isinstance(state.get("original"), dict):
                state["original"] = {"min": 0.0, "max": original_max}
            if not isinstance(state.get("deconvolved"), dict):
                state["deconvolved"] = {"min": 0.0, "max": deconvolved_max}
            original_state = dict(state.get("original") or {})
            deconvolved_state = dict(state.get("deconvolved") or {})
            if original_max > 0.0 and float(original_state.get("max", 0.0)) <= 0.0:
                original_state["max"] = original_max
            if deconvolved_max > 0.0 and float(deconvolved_state.get("max", 0.0)) <= 0.0:
                deconvolved_state["max"] = deconvolved_max
            state["original"] = original_state
            state["deconvolved"] = deconvolved_state
            state["gamma"] = min(max(float(state.get("gamma", 1.0)), 0.10), 5.00)

    def _set_advanced_scaling_active(self, active: bool) -> None:
        self._advanced_scaling_active = bool(active)
        enabled = not self._advanced_scaling_active
        self._lo_group.setEnabled(enabled)
        self._hi_group.setEnabled(enabled)
        if self._advanced_scaling_active:
            self._advanced_scaling_button.setStyleSheet("font-weight: 700;")
        else:
            self._advanced_scaling_button.setStyleSheet("")

    def _ensure_advanced_scaling_window(self) -> _AdvancedScalingWindow:
        if self._advanced_scaling_window is None:
            window = _AdvancedScalingWindow(self)
            window.closed.connect(self._on_advanced_scaling_closed)
            window.channelVisibilityChanged.connect(self._on_advanced_channel_visibility_changed)
            window.channelColorChanged.connect(self._on_advanced_channel_color_changed)
            window.channelLevelsChanged.connect(self._on_advanced_channel_levels_changed)
            window.autoRequested.connect(self._on_advanced_channel_auto_requested)
            window.resetRequested.connect(self._on_advanced_channel_reset_requested)
            self._advanced_scaling_window = window
        return self._advanced_scaling_window

    def _open_advanced_scaling(self) -> None:
        if not self._display_channels():
            return
        window = self._ensure_advanced_scaling_window()
        self._sync_advanced_scaling_window()
        self._set_advanced_scaling_active(True)
        window.show()
        window.raise_()
        window.activateWindow()
        self._refresh_view()

    def _on_advanced_scaling_closed(self) -> None:
        self._set_advanced_scaling_active(False)
        self._refresh_view()

    def _sync_advanced_scaling_window(self) -> None:
        window = self._advanced_scaling_window
        if window is None:
            return
        self._ensure_channel_scaling_defaults()
        channels = self._display_channels()
        visible = [btn.isChecked() for btn in self._channel_buttons]
        names = [str(ch.get("name", f"Ch {idx}")) for idx, ch in enumerate(channels)]
        colors = [
            self._channel_colors[idx] if idx < len(self._channel_colors) else _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)]
            for idx in range(len(channels))
        ]
        maxima = [self._channel_maxima(idx) for idx in range(len(channels))]
        histograms = [self._channel_histogram_bundle(idx) for idx in range(len(channels))]
        window.set_channel_data(
            names,
            colors,
            visible,
            self._channel_scaling[:len(channels)],
            maxima,
            histograms,
        )

    def _active_scaling_projection(self) -> str:
        if self._mode_combo.currentText() != _TWO_D_MODE:
            return "Slice"
        return self._projection_combo.currentText()

    def _pane_channel_source(self, index: int, pane: str) -> Optional[np.ndarray]:
        if pane == "original":
            if 0 <= index < len(self._input_channels):
                return self._input_channels[index]
            return None
        preview = self._preview_by_t.get(self.current_timepoint())
        if preview and 0 <= index < len(preview):
            return preview[index]
        return None

    def _pane_channel_projection_source(
        self,
        index: int,
        pane: str,
        projection: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        source = self._pane_channel_source(index, pane)
        if source is None:
            return None
        projection = projection or self._active_scaling_projection()
        if projection == "SUM":
            key = id(source)
            if key not in self._sum_cache:
                self._sum_cache[key] = np.asarray(source).sum(axis=0).astype(np.float64)
            return self._sum_cache[key]
        return source

    def _pane_channel_max_value(self, index: int, pane: str, projection: Optional[str] = None) -> float:
        source = self._pane_channel_projection_source(index, pane, projection)
        if source is None:
            return 0.0
        arr = np.asarray(source)
        if arr.size == 0:
            return 0.0
        key = id(source)
        cached = self._max_cache.get(key)
        if cached is not None:
            return cached
        try:
            max_val = float(np.nanmax(arr))
        except ValueError:
            return 0.0
        if not np.isfinite(max_val):
            return 0.0
        result = max(max_val, 0.0)
        self._max_cache[key] = result
        return result

    def _channel_maxima(self, index: int, projection: Optional[str] = None) -> dict[str, float]:
        original_max = self._pane_channel_max_value(index, "original", projection)
        deconvolved_max = self._pane_channel_max_value(index, "deconvolved", projection)
        return {
            "original": original_max,
            "deconvolved": deconvolved_max,
            "shared": max(original_max, deconvolved_max, 1.0),
        }

    def _channel_histogram_counts(self, index: int, pane: str, hist_max: float, projection: Optional[str] = None) -> np.ndarray:
        source = self._pane_channel_projection_source(index, pane, projection)
        if source is None:
            return np.zeros(512, dtype=np.int64)
        finite = _finite_sample(np.asarray(source))
        if finite.size == 0:
            return np.zeros(512, dtype=np.int64)
        finite = np.clip(finite, 0.0, hist_max)
        counts, _ = np.histogram(finite, bins=512, range=(0.0, hist_max))
        return counts.astype(np.int64, copy=False)

    def _channel_histogram_bundle(self, index: int) -> dict[str, object]:
        projection = self._active_scaling_projection()
        maxima = self._channel_maxima(index, projection)
        original_hist_max = max(float(maxima.get("original", 0.0)), 1.0)
        deconvolved_hist_max = max(float(maxima.get("deconvolved", 0.0)), 1.0)
        return {
            "original_bin_edges": np.linspace(0.0, original_hist_max, 513, dtype=np.float64),
            "deconvolved_bin_edges": np.linspace(0.0, deconvolved_hist_max, 513, dtype=np.float64),
            "original": self._channel_histogram_counts(index, "original", original_hist_max, projection),
            "deconvolved": self._channel_histogram_counts(index, "deconvolved", deconvolved_hist_max, projection),
            "has_original": self._pane_channel_source(index, "original") is not None,
            "has_deconvolved": self._pane_channel_source(index, "deconvolved") is not None,
        }

    def _channel_contrast(self, stack: np.ndarray, index: int, pane: str, projection: str = "Slice") -> tuple[float, float, float]:
        if self._advanced_scaling_active and 0 <= index < len(self._channel_scaling):
            self._ensure_channel_scaling_defaults()
            maxima = self._channel_maxima(index, projection)
            pane_max = max(float(maxima.get(pane, 0.0)), 0.0)
            levels = self._channel_scaling[index]
            pane_levels = dict(levels.get(pane) or {})
            lo = min(max(float(pane_levels.get("min", 0.0)), 0.0), pane_max)
            hi = min(max(float(pane_levels.get("max", pane_max)), lo), pane_max)
            gamma = min(max(float(levels.get("gamma", 1.0)), 0.10), 5.00)
            return lo, hi, gamma
        lo, hi = self._cached_percentiles(stack, self._lo_spin.value(), self._hi_spin.value(), projection)
        return lo, hi, 1.0

    def _display_channels(self) -> list[dict]:
        channels = list(self._metadata.get("channels", []))
        names = self._metadata.get("channel_names", [])
        result: list[dict] = []
        declared_count = max(int(self._metadata.get("size_c", 0)), 0)
        for i in range(max(len(self._input_channels), len(channels), declared_count)):
            src = dict(channels[i]) if i < len(channels) else {}
            if "name" not in src:
                src["name"] = names[i] if i < len(names) else f"Ch {i}"
            src.setdefault("active", True)
            result.append(src)
        if _channels_look_like_rgb(result):
            for i, ch in enumerate(result[:3]):
                ch["name"] = _RGB_CHANNEL_NAMES[i]
                ch["color"] = _RGB_CHANNEL_COLORS[i]
        return result

    def _rebuild_channel_buttons(self, channels: list[dict]) -> None:
        for btn in self._channel_buttons:
            self._channel_bar.removeWidget(btn)
            btn.deleteLater()
        self._channel_buttons.clear()
        for i, channel in enumerate(channels):
            name = channel.get("name", f"Ch {i}")
            btn = _ChannelButton(name, i)
            btn.setCheckable(True)
            btn.setChecked(bool(channel.get("active", True)))
            btn.toggled.connect(lambda checked, idx=i: self._on_channel_button_toggled(idx, checked))
            btn.colorPickRequested.connect(self._on_channel_button_color_pick)
            self._channel_bar.insertWidget(self._channel_bar.count() - 1, btn)
            self._channel_buttons.append(btn)
            self._apply_channel_button_style(i)

    def _apply_channel_button_style(self, index: int) -> None:
        if index < 0 or index >= len(self._channel_buttons) or index >= len(self._channel_colors):
            return
        r, g, b = self._channel_colors[index]
        self._channel_buttons[index].setStyleSheet(
            f"QPushButton {{ color: rgb({r},{g},{b}); font-weight: bold; border: 2px solid rgb({r},{g},{b}); padding: 2px 8px; }}"
            f"QPushButton:checked {{ background-color: rgba({r},{g},{b},60); }}"
        )

    def _on_channel_button_toggled(self, index: int, checked: bool) -> None:
        window = self._advanced_scaling_window
        if window is not None:
            window.set_channel_visibility(index, checked)
        self._refresh_view()

    def _on_channel_button_color_pick(self, index: int) -> None:
        """Open a colour dialog to change the channel colour."""
        if index < 0 or index >= len(self._channel_colors):
            return
        current = QColor(*self._channel_colors[index])
        color = QColorDialog.getColor(current, self, "Select channel colour")
        if not color.isValid():
            return
        self._on_advanced_channel_color_changed(index, (color.red(), color.green(), color.blue()))

    def _set_channel_button_checked(self, index: int, checked: bool) -> None:
        if index < 0 or index >= len(self._channel_buttons):
            return
        btn = self._channel_buttons[index]
        if btn.isChecked() == checked:
            return
        btn.blockSignals(True)
        btn.setChecked(checked)
        btn.blockSignals(False)

    def _active_channel_indices(self) -> list[int]:
        return [
            i for i, btn in enumerate(self._channel_buttons)
            if btn.isChecked()
        ]

    def _on_mode_changed(self, mode: str) -> None:
        if mode == _THREE_D_MODE and not self._can_show_3d():
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentText(_TWO_D_MODE)
            self._mode_combo.blockSignals(False)
            mode = _TWO_D_MODE
        if mode == _THREE_D_MODE:
            self._reset_3d_on_next_render = True
        self._bar_3d.setVisible(mode == _THREE_D_MODE)
        self._projection_combo.setEnabled(mode == _TWO_D_MODE)
        self._smooth_zoom_check.setEnabled(mode == _TWO_D_MODE)
        self._navigator_check.setEnabled(mode == _TWO_D_MODE)
        self._scale_bar_check.setEnabled(mode == _TWO_D_MODE)
        self._z_slider.setEnabled(mode == _TWO_D_MODE and self._z_slider.maximum() > 0 and self._projection_combo.currentText() == "Slice")
        self._refresh_navigator_state()
        self._refresh_view()

    def _on_projection_changed(self, _projection: str) -> None:
        self._z_slider.setEnabled(
            self._mode_combo.currentText() == _TWO_D_MODE
            and self._z_slider.maximum() > 0
            and self._projection_combo.currentText() == "Slice"
        )
        self._channel_scaling_projection = None
        self._ensure_channel_scaling_defaults()
        self._sync_advanced_scaling_window()
        self._refresh_view()

    def _on_time_changed(self, value: int) -> None:
        self._update_labels()
        self.timepointChanged.emit(value)
        self._sync_advanced_scaling_window()
        self._refresh_view()

    def _on_contrast_changed(self) -> None:
        if self._mode_combo.currentText() == _THREE_D_MODE:
            self._schedule_3d_refresh()
        else:
            # Debounce: wait until the user pauses before re-rendering, so
            # rapidly spinning the spinbox doesn't queue dozens of expensive
            # percentile + pixmap operations.
            self._contrast_2d_timer.start()

    def _on_advanced_channel_visibility_changed(self, index: int, visible: bool) -> None:
        self._set_channel_button_checked(index, visible)
        self._refresh_view()

    def _on_advanced_channel_color_changed(self, index: int, color: object) -> None:
        if index < 0 or index >= len(self._channel_colors):
            return
        rgb = tuple(int(v) for v in color)
        self._channel_colors[index] = rgb
        channels_meta = self._metadata.setdefault("channels", [])
        while len(channels_meta) <= index:
            channels_meta.append({})
        channels_meta[index]["color"] = rgb
        self._apply_channel_button_style(index)
        if self._advanced_scaling_window is not None:
            self._advanced_scaling_window.set_channel_color(index, rgb)
        self._refresh_view()

    def _on_advanced_channel_levels_changed(self, index: int, payload: object) -> None:
        if index < 0:
            return
        self._ensure_channel_scaling_defaults()
        if index >= len(self._channel_scaling):
            return
        data = dict(payload) if isinstance(payload, dict) else {}
        original = dict(data.get("original") or self._channel_scaling[index].get("original") or {})
        deconvolved = dict(data.get("deconvolved") or self._channel_scaling[index].get("deconvolved") or {})
        gamma = float(data.get("gamma", self._channel_scaling[index].get("gamma", 1.0)))
        projection = self._active_scaling_projection()
        self._channel_scaling[index] = {
            "original": {
                "min": float(original.get("min", 0.0)),
                "max": float(original.get("max", self._pane_channel_max_value(index, "original", projection))),
            },
            "deconvolved": {
                "min": float(deconvolved.get("min", 0.0)),
                "max": float(deconvolved.get("max", self._pane_channel_max_value(index, "deconvolved", projection))),
            },
            "gamma": gamma,
        }
        self._on_contrast_changed()

    def _on_advanced_channel_auto_requested(self, index: int) -> None:
        if index < 0 or index >= len(self._channel_scaling):
            return
        bundle = self._channel_histogram_bundle(index)
        original_bin_edges = np.asarray(bundle["original_bin_edges"], dtype=np.float64)
        deconvolved_bin_edges = np.asarray(bundle["deconvolved_bin_edges"], dtype=np.float64)
        original_counts = np.asarray(bundle["original"], dtype=np.float64)
        deconvolved_counts = np.asarray(bundle["deconvolved"], dtype=np.float64)
        projection = self._active_scaling_projection()
        original_max = self._pane_channel_max_value(index, "original", projection)
        deconvolved_max = self._pane_channel_max_value(index, "deconvolved", projection)
        if bool(bundle.get("has_original", False)) and np.any(original_counts > 0):
            original_lo = _percentile_from_hist(original_bin_edges, original_counts, self._lo_spin.value())
            original_hi = _percentile_from_hist(original_bin_edges, original_counts, self._hi_spin.value())
        else:
            original_lo = 0.0
            original_hi = original_max
        if bool(bundle.get("has_deconvolved", False)) and np.any(deconvolved_counts > 0):
            deconvolved_lo = _percentile_from_hist(deconvolved_bin_edges, deconvolved_counts, self._lo_spin.value())
            deconvolved_hi = _percentile_from_hist(deconvolved_bin_edges, deconvolved_counts, self._hi_spin.value())
        else:
            deconvolved_lo = 0.0
            deconvolved_hi = deconvolved_max
        gamma = float(self._channel_scaling[index].get("gamma", 1.0))
        self._channel_scaling[index] = {
            "original": {
                "min": float(max(original_lo, 0.0)),
                "max": float(min(max(original_hi, original_lo), original_max)),
            },
            "deconvolved": {
                "min": float(max(deconvolved_lo, 0.0)),
                "max": float(min(max(deconvolved_hi, deconvolved_lo), deconvolved_max)),
            },
            "gamma": gamma,
        }
        self._sync_advanced_scaling_window()
        self._on_contrast_changed()

    def _on_advanced_channel_reset_requested(self, index: int) -> None:
        if index < 0 or index >= len(self._channel_scaling):
            return
        self._channel_scaling[index] = {
            "original": {
                "min": 0.0,
                "max": self._pane_channel_max_value(index, "original", self._active_scaling_projection()),
            },
            "deconvolved": {
                "min": 0.0,
                "max": self._pane_channel_max_value(index, "deconvolved", self._active_scaling_projection()),
            },
            "gamma": 1.0,
        }
        self._sync_advanced_scaling_window()
        self._on_contrast_changed()

    def _on_volume_method_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._available_volume_methods):
            return
        self._active_volume_method = self._available_volume_methods[index]
        self._set_volume_slider_ui(
            self._active_volume_method,
            self._volume_method_values[self._active_volume_method],
        )
        self._schedule_3d_refresh()

    def _on_volume_slider_changed(self, value: int) -> None:
        method = self._active_volume_method
        self._volume_method_values[method] = value
        self._volume_slider_value.setText(f"{value / 100:.2f}")
        self._schedule_3d_refresh()

    def _schedule_3d_refresh(self) -> None:
        if self._mode_combo.currentText() == _THREE_D_MODE:
            self._refresh_3d_timer.start()

    def _refresh_volume_method_options(self, channels: Optional[list[dict]] = None) -> None:
        current = self._active_volume_method
        fluorescence_like = _channels_look_fluorescence_like(channels or self._display_channels())
        methods = [m for m in _VOLUME_METHODS if m != "minip" or not fluorescence_like]
        if current not in methods:
            current = methods[0]
        self._available_volume_methods = methods
        self._active_volume_method = current
        self._volume_method_combo.blockSignals(True)
        self._volume_method_combo.clear()
        self._volume_method_combo.addItems([_VOLUME_METHOD_LABELS[m] for m in methods])
        self._volume_method_combo.setCurrentIndex(methods.index(current))
        self._volume_method_combo.blockSignals(False)
        self._set_volume_slider_ui(current, self._volume_method_values[current])

    def _set_volume_slider_ui(self, method: str, value: int) -> None:
        spec = _VOLUME_METHOD_UI[method]
        min_val, max_val = spec["range"]
        clamped = max(min_val, min(int(value), max_val))
        self._volume_slider_label.setText(spec["label"])
        self._volume_slider.blockSignals(True)
        self._volume_slider.setRange(min_val, max_val)
        self._volume_slider.setValue(clamped)
        self._volume_slider.blockSignals(False)
        self._volume_slider_value.setText(f"{clamped / 100:.2f}")
        self._smooth_check.setVisible(method in _INTERPOLATION_TOGGLE_METHODS)

    def _apply_view_selector(self) -> None:
        mode = self._view_mode_text()
        if self._mode_combo.currentText() == _TWO_D_MODE and mode not in ("Both", "Original", "Deconvolved"):
            self._input_pane.setVisible(True)
            self._output_pane.setVisible(False)
            self._refresh_navigator_state()
            return
        self._input_pane.setVisible(mode in ("Both", "Original"))
        self._output_pane.setVisible(mode in ("Both", "Deconvolved"))
        self._refresh_navigator_state()

    def _refresh_navigator_state(self) -> None:
        enabled = bool(self._navigator_enabled)
        mode = self._view_mode_text()
        compare_mode = mode not in ("Both", "Original", "Deconvolved")
        input_enabled = enabled and self._mode_combo.currentText() == _TWO_D_MODE
        output_enabled = input_enabled and mode == "Deconvolved" and not compare_mode
        self._input_pane.set_navigator_enabled(input_enabled)
        self._output_pane.set_navigator_enabled(output_enabled)

    def _update_labels(self) -> None:
        t = self._t_slider.value()
        z = self._z_slider.value()
        self._t_label.setText(f"{t}/{max(self._t_slider.maximum(), 0)}")
        z_max = max(self._z_slider.maximum(), 0)
        z_width = max(1, len(str(z_max)))
        self._z_label.setText(f"{z:0{z_width}d}/{z_max}")

    def _ensure_mode_valid(self) -> None:
        can_show_3d = self._can_show_3d()
        if self._mode_combo.currentText() == _THREE_D_MODE and not can_show_3d:
            self._mode_combo.setCurrentText(_TWO_D_MODE)

    def _can_show_3d(self) -> bool:
        if self._tiled_provider is not None:
            return False
        return _HAS_VISPY and bool(self._input_channels) and max(int(self._metadata.get("size_z", 1)), 1) > 1

    def _refresh_view(self) -> None:
        self._update_labels()
        self._apply_view_selector()
        if not self._input_channels:
            self._input_pane.show_placeholder("Open an image to view it.")
            self._output_pane.show_placeholder("Run deconvolution to preview results.")
            return
        if self._loaded_input_timepoint is not None and self._loaded_input_timepoint != self.current_timepoint():
            self._input_pane.show_placeholder(f"Loading T={self.current_timepoint()}…")
            preview = self._preview_by_t.get(self.current_timepoint())
            if preview:
                self._output_pane.show_content()
            else:
                self._output_pane.show_placeholder(
                    f"No deconvolved preview for T={self.current_timepoint()}.\nRun Deconvolution to preview this timepoint."
                )
            return

        mode = self._mode_combo.currentText()
        if mode == _THREE_D_MODE and self._can_show_3d():
            self._refresh_view_3d()
        else:
            self._refresh_view_2d()

    def _refresh_view_2d(self) -> None:
        timepoint = self.current_timepoint()
        projection = self._projection_combo.currentText()
        self._z_slider.setEnabled(self._z_slider.maximum() > 0 and projection == "Slice")
        self._input_pane.set_mode(_TWO_D_MODE)
        self._output_pane.set_mode(_TWO_D_MODE)
        preview = self._preview_by_t.get(timepoint)
        compare_mode = self._view_mode_text()

        if preview and compare_mode not in ("Both", "Original", "Deconvolved"):
            self._input_pane.set_pixmap(QPixmap.fromImage(self.current_comparison_image()))
            self._output_pane.setVisible(False)
            self._input_pane.setVisible(True)
            self._refresh_scale_bar_state()
            if self._fit_on_next_render:
                self.fit_views()
                self._fit_on_next_render = False
            return

        if self._tiled_provider is not None and self._tiled_item is not None:
            input_pixmap = self._build_2d_pixmap(self._input_channels, projection, pane="original")
            self._tiled_item.set_overview(input_pixmap)
            contrast = {}
            for idx in self._active_channel_indices():
                if 0 <= idx < len(self._input_channels):
                    contrast[idx] = self._channel_contrast(
                        self._input_channels[idx],
                        idx,
                        "original",
                        projection,
                    )[:2]
            self._tiled_item.set_display(
                self._active_channel_indices(),
                self._channel_colors,
                contrast,
                int(self._z_slider.value()),
                timepoint,
                projection,
            )
            self._input_pane.set_graphics_item(self._tiled_item)
        else:
            input_pixmap = self._build_2d_pixmap(self._input_channels, projection, pane="original")
            self._input_pane.set_pixmap(input_pixmap)

        if preview:
            output_was_placeholder = self._output_pane._display_stack.currentIndex() == 1
            output_pixmap = self._build_2d_pixmap(preview, projection, pane="deconvolved")
            self._output_pane.set_pixmap(output_pixmap)
            if output_was_placeholder:
                self._output_pane.copy_2d_view_from(self._input_pane)
        else:
            self._output_pane.show_placeholder(
                f"No deconvolved preview for T={timepoint}.\nRun Deconvolution to preview this timepoint."
            )

        self._refresh_scale_bar_state()

        if self._fit_on_next_render:
            self.fit_views()
            self._fit_on_next_render = False

    def _refresh_view_3d(self) -> None:
        timepoint = self.current_timepoint()
        self._input_pane.set_mode(_THREE_D_MODE)
        self._output_pane.set_mode(_THREE_D_MODE)
        self._z_slider.setEnabled(False)
        if self._reset_3d_on_next_render:
            input_camera_state = None
            output_camera_state = None
        else:
            input_camera_state = self._input_pane.camera_state()
            output_camera_state = self._output_pane.camera_state()

        input_stacks = self._build_3d_channel_payload(self._input_channels, pane="original")
        self._input_pane.load_3d(
            input_stacks,
            method=self._active_volume_method,
            slider_val=self._volume_method_values[self._active_volume_method] / 100.0,
            interpolation="linear" if self._smooth_check.isChecked() else "nearest",
            downsample=[1, 2, 4][self._downsample_combo.currentIndex()],
            pixel_size_x=self._metadata.get("pixel_size_x"),
            pixel_size_z=self._metadata.get("pixel_size_z"),
            preserve_camera_state=input_camera_state,
        )

        preview = self._preview_by_t.get(timepoint)
        if preview:
            output_was_placeholder = self._output_pane._display_stack.currentIndex() == 1
            output_stacks = self._build_3d_channel_payload(preview, pane="deconvolved")
            self._output_pane.load_3d(
                output_stacks,
                method=self._active_volume_method,
                slider_val=self._volume_method_values[self._active_volume_method] / 100.0,
                interpolation="linear" if self._smooth_check.isChecked() else "nearest",
                downsample=[1, 2, 4][self._downsample_combo.currentIndex()],
                pixel_size_x=self._metadata.get("pixel_size_x"),
                pixel_size_z=self._metadata.get("pixel_size_z"),
                preserve_camera_state=output_camera_state if not output_was_placeholder else input_camera_state,
            )
            if output_was_placeholder:
                self._sync_3d_from_input(self._input_pane.camera_state())
        else:
            self._output_pane.show_placeholder(
                f"No deconvolved preview for T={timepoint}.\nRun Deconvolution to preview this timepoint."
            )
        self._reset_3d_on_next_render = False

    def _cached_percentiles(
        self, stack: np.ndarray, lo_pct: float, hi_pct: float, projection: str = "Slice"
    ) -> tuple[float, float]:
        """Return (lo, hi) contrast limits using a per-array cached histogram.

        For Slice/MIP the 3-D stack is used directly — its value range covers
        both projections.  For SUM the summed plane is cached separately (in
        ``_sum_cache``) so it stays alive with a stable id(); SUM values are
        n_z× larger than per-slice values and need their own histogram range.
        """
        if projection == "SUM":
            stack_key = id(stack)
            if stack_key not in self._sum_cache:
                self._sum_cache[stack_key] = stack.sum(axis=0).astype(np.float64)
            src = self._sum_cache[stack_key]
        else:
            src = stack
        key = id(src)
        if key not in self._hist_cache:
            flat = _finite_sample(src)
            if flat.size == 0:
                return 0.0, 1.0
            vmin, vmax = float(flat.min()), float(flat.max())
            if vmax <= vmin:
                vmax = vmin + 1.0
            counts, bin_edges = np.histogram(flat, bins=1024, range=(vmin, vmax))
            self._hist_cache[key] = (bin_edges, counts)
        bin_edges, counts = self._hist_cache[key]
        lo = _percentile_from_hist(bin_edges, counts, lo_pct)
        hi = _percentile_from_hist(bin_edges, counts, hi_pct)
        return lo, hi

    def _build_2d_pixmap(
        self,
        channels_zyx: list[np.ndarray],
        projection: str,
        *,
        pane: str,
    ) -> QPixmap:
        slices: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]] = []
        for idx in self._active_channel_indices():
            if idx >= len(channels_zyx):
                continue
            stack = channels_zyx[idx]
            plane = _project_stack(stack, projection, self._z_slider.value())
            slices.append((plane, self._channel_colors[idx], self._channel_contrast(stack, idx, pane, projection)))
        return _composite_to_pixmap(slices)

    def _build_2d_image(
        self,
        channels_zyx: list[np.ndarray],
        projection: str,
        *,
        pane: str,
    ) -> QImage:
        slices: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]] = []
        for idx in self._active_channel_indices():
            if idx >= len(channels_zyx):
                continue
            stack = channels_zyx[idx]
            plane = _project_stack(stack, projection, self._z_slider.value())
            slices.append((plane, self._channel_colors[idx], self._channel_contrast(stack, idx, pane, projection)))
        return _rgb_to_qimage(_composite_to_rgb(slices))

    def _build_2d_rgb(
        self,
        channels_zyx: list[np.ndarray],
        projection: str,
        *,
        pane: str,
    ) -> np.ndarray:
        slices: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]] = []
        for idx in self._active_channel_indices():
            if idx >= len(channels_zyx):
                continue
            stack = channels_zyx[idx]
            plane = _project_stack(stack, projection, self._z_slider.value())
            slices.append((plane, self._channel_colors[idx], self._channel_contrast(stack, idx, pane, projection)))
        if not slices:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        stride = _display_stride(slices[0][0].shape)
        if stride > 1:
            slices = [(arr[::stride, ::stride], color, contrast) for arr, color, contrast in slices]
        return _composite_to_rgb(slices)

    def _build_compare_rgb(self, preview: list[np.ndarray], projection: str) -> np.ndarray:
        original = self._build_2d_rgb(self._input_channels, projection, pane="original")
        deconvolved = self._build_2d_rgb(preview, projection, pane="deconvolved")
        if original.size == 0:
            return deconvolved
        if deconvolved.size == 0:
            return original
        h = min(original.shape[0], deconvolved.shape[0])
        w = min(original.shape[1], deconvolved.shape[1])
        original = original[:h, :w]
        deconvolved = deconvolved[:h, :w]
        mode = self._view_mode_text()
        if mode == "Linked split":
            split = int(round(w * self._compare_split_slider.value() / 100.0))
            out = deconvolved.copy()
            out[:, :split] = original[:, :split]
            return out
        if mode == "Blink":
            return deconvolved if self._blink_show_deconvolved else original
        if mode == "Difference":
            diff = np.abs(deconvolved.astype(np.int16) - original.astype(np.int16)).astype(np.float32)
            diff *= 2.0
            return np.clip(diff, 0, 255).astype(np.uint8)
        if mode == "Ratio":
            orig_lum = original.astype(np.float32).mean(axis=2) + 1.0
            deconv_lum = deconvolved.astype(np.float32).mean(axis=2) + 1.0
            log_ratio = np.log2(deconv_lum / orig_lum)
            scaled = np.clip(log_ratio / 2.0, -1.0, 1.0)
            out = np.zeros_like(original, dtype=np.float32)
            positive = scaled > 0
            negative = scaled < 0
            out[..., 0] = np.where(positive, scaled * 255.0, 0.0)
            out[..., 2] = np.where(negative, -scaled * 255.0, 0.0)
            out[..., 1] = np.clip((1.0 - np.abs(scaled)) * 70.0, 0.0, 70.0)
            return out.astype(np.uint8)
        if mode == "Original":
            return original
        return deconvolved

    def _on_cursor_moved(self, pane: str, scene_x: float, scene_y: float, inside: bool) -> None:
        if not inside:
            self.cursorInfoChanged.emit("")
            return
        channels = self._input_channels if pane == "Original" else self._preview_by_t.get(self.current_timepoint(), [])
        if not channels:
            self.cursorInfoChanged.emit("")
            return
        projection = self._projection_combo.currentText()
        stride = 1
        for idx in self._active_channel_indices():
            if 0 <= idx < len(channels):
                stride = _display_stride(tuple(int(v) for v in channels[idx].shape[-2:]))
                break
        x = int(round(scene_x * stride))
        y = int(round(scene_y * stride))
        z = int(self._z_slider.value()) if projection == "Slice" else -1
        values: list[str] = []
        for idx in self._active_channel_indices():
            if idx >= len(channels):
                continue
            stack = channels[idx]
            try:
                plane = _project_stack(stack, projection, self._z_slider.value())
                if 0 <= y < plane.shape[0] and 0 <= x < plane.shape[1]:
                    values.append(f"Ch{idx + 1}={float(plane[y, x]):.4g}")
            except Exception:
                continue
        px_um = self._pixel_size_x_um()
        physical = f" | {x * px_um:.2f} um, {y * px_um:.2f} um" if px_um else ""
        z_text = f" Z={z}" if z >= 0 else ""
        value_text = " | " + ", ".join(values) if values else ""
        self.cursorInfoChanged.emit(
            f"{pane} X={x} Y={y}{z_text} T={self.current_timepoint()}{physical}{value_text}"
        )

    def _build_3d_channel_payload(
        self,
        channels_zyx: list[np.ndarray],
        *,
        pane: str,
    ) -> list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]]:
        result: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]] = []
        for idx in self._active_channel_indices():
            if idx >= len(channels_zyx):
                continue
            stack = channels_zyx[idx]
            result.append((stack, self._channel_colors[idx], self._channel_contrast(stack, idx, pane)))
        return result

    def _reset_3d_views(self) -> None:
        self._input_pane.reset_camera()
        self._output_pane.reset_camera()
        self._sync_3d_from_input(self._input_pane.camera_state())

    def _sync_3d_from_input(self, state: object) -> None:
        if self._syncing_camera:
            return
        self._syncing_camera = True
        try:
            self._output_pane.apply_camera_state(state)
        finally:
            self._syncing_camera = False

    def _sync_3d_from_output(self, state: object) -> None:
        if self._syncing_camera:
            return
        self._syncing_camera = True
        try:
            self._input_pane.apply_camera_state(state)
        finally:
            self._syncing_camera = False


if _HAS_VISPY:

    class _ChannelColormap(BaseColormap):
        glsl_map = ""

        def __init__(self, rgb: tuple[int, int, int], *, translucent_boost: bool = False):
            r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
            if translucent_boost:
                self.glsl_map = (
                    "vec4 channel_cmap(float t) {\n"
                    "    float c = clamp(pow(t, 0.55) * 1.18, 0.0, 1.0);\n"
                    "    float a = clamp(pow(t, 1.35) * 0.82, 0.0, 1.0);\n"
                    f"    return vec4({r:.4f} * c, {g:.4f} * c, {b:.4f} * c, a);\n"
                    "}\n"
                )
            else:
                self.glsl_map = (
                    "vec4 channel_cmap(float t) {\n"
                    f"    return vec4({r:.4f} * t, {g:.4f} * t, {b:.4f} * t, clamp(t * 1.2, 0.0, 1.0));\n"
                    "}\n"
                )
            super().__init__()
