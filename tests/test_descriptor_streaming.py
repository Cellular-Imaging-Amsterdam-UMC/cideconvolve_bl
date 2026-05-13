import json
from pathlib import Path


def test_descriptor_exposes_streaming_flags() -> None:
    descriptor = json.loads((Path(__file__).resolve().parent.parent / "descriptor.json").read_text())
    ids = {entry["id"] for entry in descriptor["inputs"]}
    for expected in {
        "output_format",
        "streaming",
        "tile_limits",
        "streaming_threshold_gb",
        "scene",
        "hcs_field",
    }:
        assert expected in ids
        assert f"@{expected}" in descriptor["command-line"]


def test_descriptor_convergence_matches_gui_wording() -> None:
    descriptor = json.loads((Path(__file__).resolve().parent.parent / "descriptor.json").read_text())
    convergence = next(entry for entry in descriptor["inputs"] if entry["id"] == "convergence")
    assert convergence["value-choices"] == ["auto", "fixed"]

