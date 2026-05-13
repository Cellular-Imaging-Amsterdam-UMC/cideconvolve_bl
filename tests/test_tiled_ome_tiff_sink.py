from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")
tifffile = pytest.importorskip("tifffile")

from core.streaming import TiledOmeTiffSink  # noqa: E402


def test_tiled_ome_tiff_sink_writes_level0_metadata_and_subifd(tmp_path):
    path = tmp_path / "tiny.ome.tiff"
    sink = TiledOmeTiffSink(
        path,
        shape=(1, 2, 1, 32, 32),
        metadata={
            "pixel_size_x": 0.455,
            "pixel_size_y": 0.455,
            "pixel_size_z": 1.0,
            "channel_names": ["Red", "Green"],
            "channels": [
                {"name": "Red", "color": (255, 0, 0), "emission_wavelength": 610.0},
                {"name": "Green", "color": (0, 255, 0), "emission_wavelength": 520.0},
            ],
            "save_child_name": "Test_Field_001",
        },
        levels=2,
    )

    for channel in range(2):
        sink.write_tile(
            t=0,
            c=channel,
            z=slice(0, 1),
            y=slice(0, 32),
            x=slice(0, 32),
            data=np.full((1, 32, 32), channel + 1, dtype=np.float32),
        )
    sink.build_pyramids()
    sink.validate()
    sink.close()

    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].dtype == np.float32
        assert tif.series[0].shape[-2:] == (32, 32)
        assert tif.pages[0].compression.name == "LZW"
        assert int(tif.pages[0].tags["Predictor"].value) == 3
        assert len(tif.pages[0].pages) == 1
        ome_xml = tif.ome_metadata or ""
        assert "PhysicalSizeX=\"0.455" in ome_xml
        assert "PhysicalSizeY=\"0.455" in ome_xml
        assert "PhysicalSizeZ=\"1.0" in ome_xml
        assert "Red" in ome_xml
        assert "Green" in ome_xml
        assert "Color=\"4278190335\"" in ome_xml
        assert "EmissionWavelength=\"610.0\"" in ome_xml
        payload = json.loads(tif.pages[0].tags[65000].value)
        assert payload["physical_pixel_sizes_um"]["x"] == 0.455
        assert payload["metadata"]["save_child_name"] == "Test_Field_001"


def test_tiled_ome_tiff_sink_accepts_non_16_multiple_image_sizes(tmp_path):
    path = tmp_path / "odd-size.ome.tiff"
    sink = TiledOmeTiffSink(
        path,
        shape=(1, 1, 1, 271, 303),
        metadata={"pixel_size_x": 0.06681, "pixel_size_y": 0.06681, "pixel_size_z": 0.2},
        levels=1,
    )
    sink.write_tile(
        t=0,
        c=0,
        z=slice(0, 1),
        y=slice(0, 271),
        x=slice(0, 303),
        data=np.ones((1, 271, 303), dtype=np.float32),
    )
    sink.validate()
    sink.close()

    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape[-2:] == (271, 303)
        assert tif.pages[0].tilelength % 16 == 0
        assert tif.pages[0].tilewidth % 16 == 0
