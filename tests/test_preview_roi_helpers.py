from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))

from gui.gui_deconvolve_ci import (  # noqa: E402
    _crop_channels_to_roi,
    _expand_roi_rect,
    _normalise_roi_rect,
    _place_roi_channels_on_full_canvas,
)


def test_normalise_roi_rect_clamps_to_image_bounds():
    rect = _normalise_roi_rect({"x": -5, "y": 8, "width": 20, "height": 10}, 12, 14)
    assert rect == {"x": 0, "y": 8, "width": 14, "height": 4}


def test_normalise_roi_rect_rejects_tiny_selection():
    assert _normalise_roi_rect({"x": 3, "y": 3, "width": 1, "height": 8}, 20, 20) is None


def test_expand_roi_rect_clamps_padding_at_edges():
    rect = {"x": 2, "y": 3, "width": 5, "height": 6}
    expanded = _expand_roi_rect(rect, 12, 14, 4)
    assert expanded == {"x": 0, "y": 0, "width": 11, "height": 12}


def test_crop_channels_to_roi_keeps_full_z():
    channel = np.arange(2 * 6 * 7, dtype=np.float32).reshape(2, 6, 7)
    cropped = _crop_channels_to_roi([channel], {"x": 2, "y": 1, "width": 3, "height": 4})
    assert cropped[0].shape == (2, 4, 3)
    np.testing.assert_array_equal(cropped[0], channel[:, 1:5, 2:5])


def test_place_roi_channels_on_full_canvas_crops_compute_padding():
    source = np.zeros((2, 8, 9), dtype=np.float32)
    roi_result = np.ones((2, 6, 7), dtype=np.float32)
    payload = {
        "display_rect": {"x": 3, "y": 2, "width": 4, "height": 3},
        "compute_rect": {"x": 1, "y": 1, "width": 7, "height": 6},
    }
    out = _place_roi_channels_on_full_canvas([roi_result], [source], payload)
    assert out[0].shape == source.shape
    assert np.count_nonzero(out[0]) == 2 * 3 * 4
    np.testing.assert_array_equal(out[0][:, 2:5, 3:7], np.ones((2, 3, 4), dtype=np.float32))
    assert np.count_nonzero(out[0][:, :2, :]) == 0
