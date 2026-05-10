"""Simple PyQt6 launcher for ci_rl_dl training."""

from __future__ import annotations

import logging
import math
import multiprocessing
import queue
import re
import sys
import threading
from pathlib import Path

# Windows taskbar: set AppUserModelID so the taskbar shows our icon.
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ci.gui_train")

try:
    from PyQt6.QtCore import QSize, Qt, QTimer
    from PyQt6.QtGui import QIcon, QPainter, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QCheckBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QDoubleSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - exercised only without PyQt6.
    print(f"PyQt6 is required for gui_train.py and is not available: {exc}")
    raise SystemExit(1)

from train import TrainConfig, train

try:
    from PyQt6.QtSvg import QSvgRenderer
except ImportError:  # pragma: no cover - fallback still loads the SVG directly.
    QSvgRenderer = None

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - GUI gracefully falls back to text only.
    FigureCanvas = None
    Figure = None


SCRIPT_DIR = Path(__file__).resolve().parent
ICON_PATH = SCRIPT_DIR / "icon.svg"


def _load_app_icon() -> QIcon:
    """Build a multi-size icon from the bundled SVG for crisp Windows display."""
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


TRAINING_PRESETS = {
    "Small smoke test": {
        "output_suffix": "gui_small_test",
        "num_volumes": 4,
        "volume_shape": "5,32,32",
        "patch_size": 24,
        "z_context": 1,
        "batch_size": 2,
        "epochs": 1,
        "steps": 2,
        "learning_rate": 1e-3,
        "base_channels": 8,
        "residual_scale": 1.0,
        "rl_iterations": 2,
        "rl_iteration_pool": "2",
        "train_samples_per_epoch": 4,
        "val_samples": 2,
        "reconvolution_weight": 0.0,
        "negative_residual_weight": 0.05,
        "max_negative_residual_fraction": 0.25,
        "synthetic_complexity": "standard",
        "synthetic_morphology": "mixed",
        "microscope_type": "widefield",
        "psf_mismatch": "none",
        "model_type": "GatedResidualUNet25D",
        "use_conditioning": True,
        "residual_bound_fraction": 0.35,
        "residual_bound_scale": 0.05,
        "num_workers": 1,
        "data_loader_workers": 0,
        "volume_cache_size": 2,
        "mixed_precision": True,
    },
    "Small learning curve": {
        "output_suffix": "gui_small_learning",
        "num_volumes": 4,
        "volume_shape": "5,32,32",
        "patch_size": 24,
        "z_context": 1,
        "batch_size": 2,
        "epochs": 220,
        "steps": 0,
        "learning_rate": 1e-3,
        "base_channels": 8,
        "residual_scale": 1.0,
        "rl_iterations": 2,
        "rl_iteration_pool": "2",
        "train_samples_per_epoch": 4,
        "val_samples": 2,
        "reconvolution_weight": 0.0,
        "negative_residual_weight": 0.05,
        "max_negative_residual_fraction": 0.25,
        "synthetic_complexity": "standard",
        "synthetic_morphology": "mixed",
        "microscope_type": "widefield",
        "psf_mismatch": "none",
        "model_type": "GatedResidualUNet25D",
        "use_conditioning": True,
        "residual_bound_fraction": 0.35,
        "residual_bound_scale": 0.05,
        "num_workers": 1,
        "data_loader_workers": 0,
        "volume_cache_size": 2,
        "mixed_precision": True,
    },
    "Medium widefield": {
        "output_suffix": "gui_medium_widefield_v3",
        "num_volumes": 250,
        "volume_shape": "24,128,128",
        "patch_size": 96,
        "z_context": 2,
        "batch_size": 6,
        "epochs": 15,
        "steps": 0,
        "learning_rate": 5e-4,
        "base_channels": 24,
        "residual_scale": 1.0,
        "rl_iterations": 50,
        "rl_iteration_pool": "15,25,35,50,80",
        "train_samples_per_epoch": 3000,
        "val_samples": 384,
        "reconvolution_weight": 0.01,
        "negative_residual_weight": 0.08,
        "max_negative_residual_fraction": 0.20,
        "synthetic_complexity": "full",
        "synthetic_morphology": "mixed",
        "microscope_type": "widefield",
        "psf_mismatch": "mild",
        "psf_mismatch_moderate_fraction": 0.0,
        "model_type": "GatedResidualUNet25D",
        "use_conditioning": True,
        "residual_bound_fraction": 0.35,
        "residual_bound_scale": 0.05,
        "num_workers": 0,
        "data_loader_workers": 4,
        "volume_cache_size": 8,
        "mixed_precision": True,
    },
    "Medium confocal": {
        "output_suffix": "gui_medium_confocal_v3",
        "num_volumes": 250,
        "volume_shape": "24,128,128",
        "patch_size": 96,
        "z_context": 2,
        "batch_size": 6,
        "epochs": 15,
        "steps": 0,
        "learning_rate": 5e-4,
        "base_channels": 24,
        "residual_scale": 1.0,
        "rl_iterations": 50,
        "rl_iteration_pool": "15,25,35,50,80",
        "train_samples_per_epoch": 3000,
        "val_samples": 384,
        "reconvolution_weight": 0.01,
        "negative_residual_weight": 0.08,
        "max_negative_residual_fraction": 0.20,
        "synthetic_complexity": "full",
        "synthetic_morphology": "mixed",
        "microscope_type": "confocal",
        "psf_mismatch": "mild",
        "psf_mismatch_moderate_fraction": 0.0,
        "model_type": "GatedResidualUNet25D",
        "use_conditioning": True,
        "residual_bound_fraction": 0.35,
        "residual_bound_scale": 0.05,
        "num_workers": 0,
        "data_loader_workers": 4,
        "volume_cache_size": 8,
        "mixed_precision": True,
    },
    "Large widefield quality": {
        "output_suffix": "gui_large_widefield_quality_v3",
        "num_volumes": 1000,
        "volume_shape": "32,256,256",
        "patch_size": 128,
        "z_context": 2,
        "batch_size": 16,
        "epochs": 30,
        "steps": 0,
        "learning_rate": 5e-4,
        "base_channels": 32,
        "residual_scale": 1.0,
        "rl_iterations": 50,
        "rl_iteration_pool": "15,25,35,50,80",
        "train_samples_per_epoch": 12000,
        "val_samples": 1000,
        "reconvolution_weight": 0.02,
        "negative_residual_weight": 0.08,
        "max_negative_residual_fraction": 0.20,
        "synthetic_complexity": "full",
        "synthetic_morphology": "mixed",
        "microscope_type": "widefield",
        "psf_mismatch": "mild",
        "psf_mismatch_moderate_fraction": 0.2,
        "model_type": "GatedResidualUNet25D",
        "use_conditioning": True,
        "residual_bound_fraction": 0.35,
        "residual_bound_scale": 0.05,
        "num_workers": 0,
        "data_loader_workers": 8,
        "volume_cache_size": 8,
        "mixed_precision": True,
    },
    "Large confocal quality": {
        "output_suffix": "gui_large_confocal_quality_v3",
        "num_volumes": 1000,
        "volume_shape": "32,256,256",
        "patch_size": 128,
        "z_context": 2,
        "batch_size": 16,
        "epochs": 30,
        "steps": 0,
        "learning_rate": 5e-4,
        "base_channels": 32,
        "residual_scale": 1.0,
        "rl_iterations": 50,
        "rl_iteration_pool": "15,25,35,50,80",
        "train_samples_per_epoch": 12000,
        "val_samples": 1000,
        "reconvolution_weight": 0.02,
        "negative_residual_weight": 0.08,
        "max_negative_residual_fraction": 0.20,
        "synthetic_complexity": "full",
        "synthetic_morphology": "mixed",
        "microscope_type": "confocal",
        "psf_mismatch": "mild",
        "psf_mismatch_moderate_fraction": 0.2,
        "model_type": "GatedResidualUNet25D",
        "use_conditioning": True,
        "residual_bound_fraction": 0.35,
        "residual_bound_scale": 0.05,
        "num_workers": 0,
        "data_loader_workers": 8,
        "volume_cache_size": 8,
        "mixed_precision": True,
    },
    "Large mixed quality": {
        "output_suffix": "gui_large_mixed_quality_v3",
        "num_volumes": 1000,
        "volume_shape": "32,256,256",
        "patch_size": 128,
        "z_context": 2,
        "batch_size": 16,
        "epochs": 30,
        "steps": 0,
        "learning_rate": 5e-4,
        "base_channels": 32,
        "residual_scale": 1.0,
        "rl_iterations": 50,
        "rl_iteration_pool": "15,25,35,50,80",
        "train_samples_per_epoch": 12000,
        "val_samples": 1000,
        "reconvolution_weight": 0.02,
        "negative_residual_weight": 0.08,
        "max_negative_residual_fraction": 0.20,
        "synthetic_complexity": "full",
        "synthetic_morphology": "mixed",
        "microscope_type": "mixed",
        "psf_mismatch": "mild",
        "psf_mismatch_moderate_fraction": 0.2,
        "model_type": "GatedResidualUNet25D",
        "use_conditioning": True,
        "residual_bound_fraction": 0.35,
        "residual_bound_scale": 0.05,
        "num_workers": 0,
        "data_loader_workers": 8,
        "volume_cache_size": 8,
        "mixed_precision": True,
    },
}


class TrainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ci_rl_dl training")
        app_icon = _load_app_icon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        self.messages: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.step_loss_points: list[tuple[int, float]] = []
        self.epoch_loss_points: list[tuple[int, float, float]] = []
        self._auto_output_path: str | None = None

        root = QWidget()
        layout = QVBoxLayout(root)
        top_form = QFormLayout()
        columns = QHBoxLayout()
        left_form = QFormLayout()
        right_form = QFormLayout()
        columns.addLayout(left_form, stretch=1)
        columns.addLayout(right_form, stretch=1)

        self.preset = QComboBox()
        self.preset.addItems(list(TRAINING_PRESETS))
        self.preset.currentTextChanged.connect(self.apply_preset)
        top_form.addRow("Preset", self.preset)

        self.output_dir = QLineEdit(str(Path("training_runs") / "gui_run"))
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse_output)
        row = QHBoxLayout()
        row.addWidget(self.output_dir)
        row.addWidget(browse)
        top_form.addRow("Output folder", row)

        self.num_volumes = self.spin(1, 10000, 24)
        self.volume_shape = QLineEdit("16,96,96")
        self.patch_size = self.spin(16, 512, 64)
        self.z_context = self.spin(0, 8, 2)
        self.batch_size = self.spin(1, 128, 4)
        self.epochs = self.spin(1, 1000, 2)
        self.steps = self.spin(0, 1_000_000, 0)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(1e-6, 1.0)
        self.lr.setValue(1e-3)
        self.base_channels = self.spin(4, 128, 16)
        self.residual_scale = QDoubleSpinBox()
        self.residual_scale.setDecimals(3)
        self.residual_scale.setRange(0.01, 10.0)
        self.residual_scale.setSingleStep(0.05)
        self.residual_scale.setValue(1.0)
        self.rl_iterations = self.spin(1, 500, 8)
        self.rl_iteration_pool = QLineEdit("")
        self.train_samples = self.spin(1, 1_000_000, 256)
        self.val_samples = self.spin(1, 100_000, 64)
        self.reconvolution_weight = QDoubleSpinBox()
        self.reconvolution_weight.setDecimals(4)
        self.reconvolution_weight.setRange(0.0, 10.0)
        self.reconvolution_weight.setSingleStep(0.01)
        self.reconvolution_weight.setValue(0.0)
        self.negative_residual_weight = QDoubleSpinBox()
        self.negative_residual_weight.setDecimals(4)
        self.negative_residual_weight.setRange(0.0, 10.0)
        self.negative_residual_weight.setSingleStep(0.01)
        self.negative_residual_weight.setValue(0.05)
        self.max_negative_residual_fraction = QDoubleSpinBox()
        self.max_negative_residual_fraction.setDecimals(3)
        self.max_negative_residual_fraction.setRange(0.0, 1.0)
        self.max_negative_residual_fraction.setSingleStep(0.05)
        self.max_negative_residual_fraction.setValue(0.25)
        self.synthetic_complexity = QComboBox()
        self.synthetic_complexity.addItems(["standard", "full"])
        self.synthetic_morphology = QComboBox()
        self.synthetic_morphology.addItems(["mixed", "generic", "dna", "mitotic", "membrane", "actin", "dendrite", "puncta"])
        self.microscope_type = QComboBox()
        self.microscope_type.addItems(["widefield", "confocal", "mixed"])
        self.microscope_type.currentTextChanged.connect(self.update_output_suffix_for_microscope)
        self.psf_mismatch = QComboBox()
        self.psf_mismatch.addItems(["none", "mild", "moderate"])
        self.psf_mismatch_moderate_fraction = QDoubleSpinBox()
        self.psf_mismatch_moderate_fraction.setDecimals(2)
        self.psf_mismatch_moderate_fraction.setRange(0.0, 1.0)
        self.psf_mismatch_moderate_fraction.setSingleStep(0.05)
        self.psf_mismatch_moderate_fraction.setValue(0.0)
        self.model_type = QComboBox()
        self.model_type.addItems(["GatedResidualUNet25D", "ResidualUNet25D"])
        self.use_conditioning = QCheckBox()
        self.use_conditioning.setChecked(True)
        self.residual_bound_fraction = QDoubleSpinBox()
        self.residual_bound_fraction.setDecimals(3)
        self.residual_bound_fraction.setRange(0.0, 2.0)
        self.residual_bound_fraction.setSingleStep(0.05)
        self.residual_bound_fraction.setValue(0.35)
        self.residual_bound_scale = QDoubleSpinBox()
        self.residual_bound_scale.setDecimals(3)
        self.residual_bound_scale.setRange(0.0, 1.0)
        self.residual_bound_scale.setSingleStep(0.01)
        self.residual_bound_scale.setValue(0.05)
        self.num_workers = self.spin(0, 256, 0)
        self.data_loader_workers = self.spin(0, 64, 0)
        self.volume_cache_size = self.spin(0, 128, 8)
        self.mixed_precision = QCheckBox()
        self.mixed_precision.setChecked(True)
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])

        left_form.addRow("Volumes", self.num_volumes)
        left_form.addRow("Volume shape Z,Y,X", self.volume_shape)
        left_form.addRow("Synthetic complexity", self.synthetic_complexity)
        left_form.addRow("Synthetic morphology", self.synthetic_morphology)
        left_form.addRow("Microscope type", self.microscope_type)
        left_form.addRow("Patch size", self.patch_size)
        left_form.addRow("Z context", self.z_context)
        left_form.addRow("Batch size", self.batch_size)
        left_form.addRow("Epochs", self.epochs)
        left_form.addRow("Steps (0 = epoch)", self.steps)
        left_form.addRow("Train samples/epoch", self.train_samples)
        left_form.addRow("Validation samples", self.val_samples)

        right_form.addRow("Learning rate", self.lr)
        right_form.addRow("Base channels", self.base_channels)
        right_form.addRow("Residual scale", self.residual_scale)
        right_form.addRow("RL iterations", self.rl_iterations)
        right_form.addRow("RL iteration pool", self.rl_iteration_pool)
        right_form.addRow("PSF mismatch", self.psf_mismatch)
        right_form.addRow("Moderate mismatch frac.", self.psf_mismatch_moderate_fraction)
        right_form.addRow("Model type", self.model_type)
        right_form.addRow("Conditioning", self.use_conditioning)
        right_form.addRow("Residual bound frac.", self.residual_bound_fraction)
        right_form.addRow("Residual bound scale", self.residual_bound_scale)
        right_form.addRow("Reconvolution weight", self.reconvolution_weight)
        right_form.addRow("Neg. residual weight", self.negative_residual_weight)
        right_form.addRow("Max neg. residual frac.", self.max_negative_residual_fraction)
        right_form.addRow("CPU workers (0 = auto)", self.num_workers)
        right_form.addRow("Loader workers (0 = auto)", self.data_loader_workers)
        right_form.addRow("Volume cache/worker", self.volume_cache_size)
        right_form.addRow("Mixed precision", self.mixed_precision)
        right_form.addRow("Device", self.device)

        layout.addLayout(top_form)
        layout.addLayout(columns)

        self.start_button = QPushButton("Start training")
        self.start_button.clicked.connect(self.start_or_stop_training)
        layout.addWidget(self.start_button)

        layout.addWidget(QLabel("Live loss"))
        self._build_loss_plot(layout)

        layout.addWidget(QLabel("Progress"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.drain_messages)
        self.timer.start(250)
        self.apply_preset(self.preset.currentText())

    def preset_output_path(self, preset: dict[str, object], microscope_type: str | None = None) -> Path:
        suffix = str(preset["output_suffix"])
        micro = (microscope_type or str(preset["microscope_type"])).strip().lower()
        if micro and not suffix.endswith(f"_{micro}"):
            suffix = f"{suffix}_{micro}"
        return Path("training_runs") / suffix

    def _build_loss_plot(self, layout: QVBoxLayout) -> None:
        if FigureCanvas is None or Figure is None:
            self.loss_plot_note = QLabel("Live graph unavailable: matplotlib Qt backend could not be loaded.")
            layout.addWidget(self.loss_plot_note)
            self.figure = None
            self.canvas = None
            self.ax = None
            return

        self.figure = Figure(figsize=(6.5, 4.55), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(385)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self._update_loss_plot()

    def _reset_loss_plot(self) -> None:
        self.step_loss_points.clear()
        self.epoch_loss_points.clear()
        self._update_loss_plot()

    def _update_loss_plot(self) -> None:
        if self.ax is None or self.canvas is None:
            return
        self.ax.clear()
        epoch_y_values: list[float] = []
        if self.step_loss_points:
            steps, losses = zip(*self.step_loss_points)
            self.ax.plot(steps, losses, color="#1f77b4", linewidth=1.2, alpha=0.35, label="step train")
            self.ax.set_xlabel("step")
        if self.epoch_loss_points:
            epochs = [row[0] for row in self.epoch_loss_points]
            train = [row[1] for row in self.epoch_loss_points]
            val = [row[2] for row in self.epoch_loss_points]
            epoch_y_values = [v for v in train + val if math.isfinite(v)]
            epoch_x = epochs
            if self.step_loss_points:
                max_step = max(step for step, _ in self.step_loss_points)
                scale = max_step / max(max(epochs), 1)
                epoch_x = [epoch * scale for epoch in epochs]
            self.ax.plot(epoch_x, train, marker="o", color="#2ca02c", linewidth=1.8, label="epoch train")
            self.ax.plot(epoch_x, val, marker="o", color="#d62728", linewidth=1.8, label="epoch val")
        self.ax.set_title("Training loss")
        self.ax.set_ylabel("loss")
        self.ax.grid(True, alpha=0.25)
        if epoch_y_values:
            ymin = min(epoch_y_values)
            ymax = max(epoch_y_values)
            pad = max((ymax - ymin) * 0.12, abs(ymax) * 0.03, 1e-6)
            self.ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
        if self.step_loss_points or self.epoch_loss_points:
            self.ax.legend(loc="best")
        else:
            self.ax.text(0.5, 0.5, "Loss will appear when training starts", ha="center", va="center", transform=self.ax.transAxes)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def spin(self, lo: int, hi: int, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setValue(value)
        return box

    def browse_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_dir.text())
        if selected:
            self.output_dir.setText(selected)
            self._auto_output_path = None

    def apply_preset(self, name: str) -> None:
        preset = TRAINING_PRESETS.get(name)
        if not preset:
            return
        path = self.preset_output_path(preset)
        self.output_dir.setText(str(path))
        self._auto_output_path = str(path)
        self.num_volumes.setValue(int(preset["num_volumes"]))
        self.volume_shape.setText(str(preset["volume_shape"]))
        self.patch_size.setValue(int(preset["patch_size"]))
        self.z_context.setValue(int(preset["z_context"]))
        self.batch_size.setValue(int(preset["batch_size"]))
        self.epochs.setValue(int(preset["epochs"]))
        self.steps.setValue(int(preset["steps"]))
        self.lr.setValue(float(preset["learning_rate"]))
        self.base_channels.setValue(int(preset["base_channels"]))
        self.residual_scale.setValue(float(preset["residual_scale"]))
        self.rl_iterations.setValue(int(preset["rl_iterations"]))
        self.rl_iteration_pool.setText(str(preset.get("rl_iteration_pool", "")))
        self.train_samples.setValue(int(preset["train_samples_per_epoch"]))
        self.val_samples.setValue(int(preset["val_samples"]))
        self.reconvolution_weight.setValue(float(preset["reconvolution_weight"]))
        self.negative_residual_weight.setValue(float(preset["negative_residual_weight"]))
        self.max_negative_residual_fraction.setValue(float(preset["max_negative_residual_fraction"]))
        self.synthetic_complexity.setCurrentText(str(preset["synthetic_complexity"]))
        self.synthetic_morphology.setCurrentText(str(preset.get("synthetic_morphology", "mixed")))
        self.microscope_type.setCurrentText(str(preset["microscope_type"]))
        self.psf_mismatch.setCurrentText(str(preset.get("psf_mismatch", "none")))
        self.psf_mismatch_moderate_fraction.setValue(float(preset.get("psf_mismatch_moderate_fraction", 0.0)))
        self.model_type.setCurrentText(str(preset.get("model_type", "GatedResidualUNet25D")))
        self.use_conditioning.setChecked(bool(preset.get("use_conditioning", True)))
        self.residual_bound_fraction.setValue(float(preset.get("residual_bound_fraction", 0.35)))
        self.residual_bound_scale.setValue(float(preset.get("residual_bound_scale", 0.05)))
        self.num_workers.setValue(int(preset["num_workers"]))
        self.data_loader_workers.setValue(int(preset["data_loader_workers"]))
        self.volume_cache_size.setValue(int(preset["volume_cache_size"]))
        self.mixed_precision.setChecked(bool(preset["mixed_precision"]))

    def update_output_suffix_for_microscope(self, microscope_type: str) -> None:
        preset = TRAINING_PRESETS.get(self.preset.currentText())
        if not preset:
            return
        if self._auto_output_path is None and self.output_dir.text().strip():
            return
        if self._auto_output_path is not None and self.output_dir.text() != self._auto_output_path:
            self._auto_output_path = None
            return
        path = self.preset_output_path(preset, microscope_type)
        self.output_dir.setText(str(path))
        self._auto_output_path = str(path)

    def parse_volume_shape(self) -> tuple[int, int, int]:
        parts = [
            int(part.strip())
            for part in self.volume_shape.text().replace("x", ",").split(",")
            if part.strip()
        ]
        if len(parts) != 3:
            raise ValueError("Volume shape must be Z,Y,X, for example 24,128,128")
        if any(part <= 0 for part in parts):
            raise ValueError("Volume shape values must be positive")
        return parts[0], parts[1], parts[2]

    def start_or_stop_training(self) -> None:
        if self.worker and self.worker.is_alive():
            if not self.stop_event.is_set():
                self.stop_event.set()
                self.start_button.setText("Stopping...")
                self.start_button.setEnabled(False)
                self.log.append("Stop requested. Training will save checkpoints, logs, curves, and examples at the next safe point.")
                self.statusBar().showMessage("Stop requested; saving at next safe point")
            return
        try:
            config = TrainConfig(
                num_volumes=self.num_volumes.value(),
                volume_shape=self.parse_volume_shape(),
                patch_size=self.patch_size.value(),
                z_context=self.z_context.value(),
                batch_size=self.batch_size.value(),
                epochs=self.epochs.value(),
                steps=self.steps.value() or None,
                learning_rate=self.lr.value(),
                output_dir=Path(self.output_dir.text()),
                device=self.device.currentText(),
                mixed_precision=self.mixed_precision.isChecked(),
                base_channels=self.base_channels.value(),
                residual_scale=self.residual_scale.value(),
                rl_iterations=self.rl_iterations.value(),
                rl_iteration_pool=tuple(int(part.strip()) for part in self.rl_iteration_pool.text().replace(";", ",").split(",") if part.strip()),
                reconvolution_weight=self.reconvolution_weight.value(),
                negative_residual_weight=self.negative_residual_weight.value(),
                max_negative_residual_fraction=self.max_negative_residual_fraction.value(),
                train_samples_per_epoch=self.train_samples.value(),
                val_samples=self.val_samples.value(),
                synthetic_complexity=self.synthetic_complexity.currentText(),
                synthetic_morphology=self.synthetic_morphology.currentText(),
                microscope_type=self.microscope_type.currentText(),
                psf_mismatch=self.psf_mismatch.currentText(),
                psf_mismatch_moderate_fraction=self.psf_mismatch_moderate_fraction.value(),
                model_type=self.model_type.currentText(),
                use_conditioning=self.use_conditioning.isChecked(),
                residual_bound_fraction=self.residual_bound_fraction.value(),
                residual_bound_scale=self.residual_bound_scale.value(),
                num_workers=self.num_workers.value(),
                data_loader_workers=self.data_loader_workers.value(),
                volume_cache_size=self.volume_cache_size.value(),
            )
        except Exception as exc:
            self.log.append(f"Configuration error: {exc}")
            return
        self.stop_event.clear()
        self.start_button.setEnabled(True)
        self.start_button.setText("Stop training")
        self.log.append(f"Starting training with preset: {self.preset.currentText()}")
        self.statusBar().showMessage("Training started")
        self._reset_loss_plot()

        def run() -> None:
            try:
                train(config, progress=self.messages.put, stop_requested=self.stop_event.is_set)
            except Exception as exc:
                self.messages.put(f"ERROR: {exc}")
            finally:
                self.messages.put("__DONE__")

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def drain_messages(self) -> None:
        while True:
            try:
                msg = self.messages.get_nowait()
            except queue.Empty:
                break
            if msg == "__DONE__":
                self.start_button.setEnabled(True)
                self.start_button.setText("Start training")
                self.stop_event.clear()
                self.statusBar().showMessage("Training finished")
            else:
                self._record_loss_from_message(msg)
                self._update_status_from_message(msg)
                self.log.append(msg)

    def _record_loss_from_message(self, msg: str) -> None:
        step_match = re.search(r"Training step\s+(\d+)(?:/\d+)?:\s+loss=([0-9.eE+-]+)", msg)
        if step_match:
            self.step_loss_points.append((int(step_match.group(1)), float(step_match.group(2))))
            self._update_loss_plot()
            return

        epoch_match = re.search(
            r"Epoch\s+(\d+)(?:/\d+)?:\s+train=([0-9.eE+-]+)\s+val=([0-9.eE+-]+)",
            msg,
        )
        if epoch_match:
            self.epoch_loss_points.append(
                (
                    int(epoch_match.group(1)),
                    float(epoch_match.group(2)),
                    float(epoch_match.group(3)),
                )
            )
            self._update_loss_plot()

    def _update_status_from_message(self, msg: str) -> None:
        timing_match = re.search(r"elapsed=([^\s]+)\s+eta=([^\s]+)\s+finish=([^\s]+)", msg)
        if not timing_match:
            return
        progress_match = re.search(r"(Training step\s+\d+(?:/\d+)?|Epoch\s+\d+(?:/\d+)?)", msg)
        prefix = progress_match.group(1) if progress_match else "Training"
        elapsed, eta, finish = timing_match.groups()
        if eta == "unknown" or finish == "unknown":
            self.statusBar().showMessage(f"{prefix} | elapsed {elapsed} | estimating finish time")
            return
        self.statusBar().showMessage(f"{prefix} | elapsed {elapsed} | ETA {eta} | finish {finish}")


def main() -> int:
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    app_icon = _load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    window = TrainWindow()
    window.resize(720, 520)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
