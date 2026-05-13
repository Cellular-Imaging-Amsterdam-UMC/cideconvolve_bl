from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = ROOT / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

import gui_deconvolve_ci as gui  # noqa: E402


def _params() -> dict:
    return {
        "method": "ci_rl",
        "niter_list": [10],
        "emission_wavelengths": [520.0],
        "excitation_wavelengths": [488.0],
        "pinhole_airy_units": [1.0],
    }


def test_batch_item_public_dict_excludes_live_source_object():
    item = gui._BatchItem(
        "omero",
        "Image A",
        "123",
        {"size_c": 3},
        source_obj=object(),
    )

    public = item.public_dict()

    assert public["source_type"] == "omero"
    assert public["locator"] == "123"
    assert "source_obj" not in public


def test_batch_output_name_uses_plain_decon_stem(tmp_path):
    params = _params()
    item = gui._BatchItem("file", "same-name.ome.tiff", "/data/a/same-name.ome.tiff")

    path = gui._batch_output_path(item, tmp_path, "OME-Zarr", params)

    assert Path(path).name == "same-name_decon.ome.zarr"


def test_batch_output_name_includes_projection_for_3d_sources(tmp_path):
    params = _params()
    item = gui._BatchItem(
        "file",
        "stack.ome.tiff",
        "/data/stack.ome.tiff",
        {"size_z": 25},
    )

    path = gui._batch_output_path(item, tmp_path, "OME-TIFF", params, "MIP")

    assert Path(path).name == "stack_decon_mip.ome.tiff"


def test_batch_leica_output_prefers_save_child_name(tmp_path):
    params = _params()
    item = gui._BatchItem(
        "leica",
        "Position 3",
        "/data/test.lif::Position 3",
        {"save_child_name": "Well_A01_Field_03", "size_z": 12},
    )

    path = gui._batch_output_path(item, tmp_path, "OME-TIFF", params, "MIP")

    assert Path(path).name == "Well_A01_Field_03_decon_mip.ome.tiff"


def test_batch_leica_output_reads_save_child_name_from_context(tmp_path):
    params = _params()
    item = gui._BatchItem(
        "leica",
        "Position 3",
        "/data/test.lif::Position 3",
        {"leica_context": {"save_child_name": "Series_007"}},
    )

    path = gui._batch_output_path(item, tmp_path, "OME-Zarr", params)

    assert Path(path).name == "Series_007_decon.ome.zarr"


def test_batch_output_display_shows_only_filename():
    text = r"C:\Users\p000881\Downloads\TestB\1\Position_7_ci_rl_20i_mip_51073c2c.ome.tiff"

    display = gui._format_batch_output_display(text, max_chars=48)

    assert "\\" not in display
    assert "/" not in display
    assert display == "Position_7_ci_rl_20i_mip_51073c2c.ome.tiff"


def test_batch_folder_display_keeps_last_folders_readable():
    text = r"C:\Users\p000881\Downloads\TestB\1"

    display = gui._format_batch_folder_display(text, max_chars=24)

    assert display == ".../TestB/1"


def test_batch_item_payload_includes_output_dir():
    item = gui._BatchItem(
        "file",
        "image.ome.tiff",
        "/data/image.ome.tiff",
        output_dir="/tmp/out",
    )

    payload = item.row_payload()

    assert payload["output_dir"] == "/tmp/out"


def test_batch_channel_validation_allows_single_or_per_channel_values():
    params = _params()
    gui._validate_channel_parameter_lists(params, 3)

    params = _params()
    params["emission_wavelengths"] = [488.0, 520.0, 610.0]
    gui._validate_channel_parameter_lists(params, 3)


def test_batch_channel_validation_allows_partial_lists_by_reusing_last_value():
    params = _params()
    params["emission_wavelengths"] = [488.0, 520.0]

    gui._validate_channel_parameter_lists(params, 3)


def test_batch_metadata_handles_channel_count_instead_of_list():
    params = _params()
    params["emission_wavelengths"] = [488.0, 520.0]

    meta = gui._streaming_output_metadata({"size_c": 3, "channels": 3}, params)

    assert len(meta["channels"]) == 3
    assert meta["channels"][0]["emission_wavelength"] == 488.0
    assert meta["channels"][1]["emission_wavelength"] == 520.0
    assert meta["channels"][2]["emission_wavelength"] == 520.0


def test_batch_metadata_overlay_drops_non_channel_sequence():
    overlay = gui._batch_metadata_overlay({"channels": 3, "channel_names": "RGB", "size_c": 3})

    assert "channels" not in overlay
    assert "channel_names" not in overlay
    assert overlay["size_c"] == 3
