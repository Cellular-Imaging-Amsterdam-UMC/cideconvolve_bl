import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from core.deconvolve import deconvolve
from core.streaming import (
    InMemoryPyramidSink,
    InMemoryRegionSource,
    ProjectionPyramidSink,
    ZarrRegionSource,
    ZarrPyramidSink,
    compute_tile_regions,
    deconvolve_streaming,
    suggest_streaming_tile_size,
)


def _identity_psf_2d() -> np.ndarray:
    psf = np.zeros((1, 1), dtype=np.float32)
    psf[0, 0] = 1.0
    return psf


def _identity_psf_3d() -> np.ndarray:
    psf = np.zeros((1, 1, 1), dtype=np.float32)
    psf[0, 0, 0] = 1.0
    return psf


def _rl_identity(image: np.ndarray, psf: np.ndarray, _channel: int) -> np.ndarray:
    return deconvolve(
        image,
        psf,
        method="ci_rl",
        niter=2,
        background=1e-6,
        offset=0.0,
        start="observed",
        convergence="fixed",
        device="cpu",
        two_d_mode="legacy_2d",
    )


def test_compute_tile_regions_cover_image_with_halo() -> None:
    regions = compute_tile_regions((1, 25, 31), (10, 12), (3, 4))
    covered = np.zeros((25, 31), dtype=bool)
    for region in regions:
        covered[region.y_core, region.x_core] = True
        assert region.y_ext.start <= region.y_core.start
        assert region.y_ext.stop >= region.y_core.stop
        assert region.x_ext.start <= region.x_core.start
        assert region.x_ext.stop >= region.x_core.stop
    assert covered.all()


def test_streaming_identity_2d_matches_eager() -> None:
    rng = np.random.default_rng(2)
    image = rng.uniform(0.1, 20.0, size=(29, 33)).astype(np.float32)
    source = InMemoryRegionSource([image], {"pixel_size_x": 0.1, "pixel_size_y": 0.1})
    sink = InMemoryPyramidSink(source.shape, source.metadata)

    summary = deconvolve_streaming(
        source,
        sink,
        psf_for_channel=lambda _c: _identity_psf_2d(),
        deconvolve_tile=_rl_identity,
        tile_yx=(11, 13),
        halo_yx=(2, 2),
        build_pyramids=False,
    )

    eager = _rl_identity(image, _identity_psf_2d(), 0)
    np.testing.assert_allclose(sink.data[0, 0, 0], eager, rtol=1e-5, atol=1e-5)
    assert summary["tiles_completed"] == summary["tiles_total"]


def test_streaming_identity_3d_matches_eager() -> None:
    rng = np.random.default_rng(3)
    image = rng.uniform(0.1, 15.0, size=(3, 22, 27)).astype(np.float32)
    source = InMemoryRegionSource([image], {"pixel_size_x": 0.1, "pixel_size_z": 0.3})
    sink = InMemoryPyramidSink(source.shape, source.metadata)

    deconvolve_streaming(
        source,
        sink,
        psf_for_channel=lambda _c: _identity_psf_3d(),
        deconvolve_tile=_rl_identity,
        tile_yx=(9, 10),
        halo_yx=(1, 1),
        build_pyramids=False,
    )

    eager = _rl_identity(image, _identity_psf_3d(), 0)
    np.testing.assert_allclose(sink.data[0, 0], eager, rtol=1e-5, atol=1e-5)


def test_projection_sink_saves_z_projection_only() -> None:
    image = np.stack(
        [
            np.full((12, 14), 1.0, dtype=np.float32),
            np.full((12, 14), 3.0, dtype=np.float32),
            np.full((12, 14), 2.0, dtype=np.float32),
        ],
        axis=0,
    )
    source = InMemoryRegionSource([image], {"pixel_size_x": 0.1, "pixel_size_z": 0.3})
    inner = InMemoryPyramidSink((1, 1, 1, 12, 14), source.metadata)
    sink = ProjectionPyramidSink(inner, source_shape=source.shape, mode="mip")

    deconvolve_streaming(
        source,
        sink,
        psf_for_channel=lambda _c: _identity_psf_3d(),
        deconvolve_tile=lambda tile, _psf, _c: tile,
        tile_yx=(7, 8),
        halo_yx=(1, 1),
        build_pyramids=False,
    )

    assert inner.data.shape == (1, 1, 1, 12, 14)
    np.testing.assert_allclose(inner.data[0, 0, 0], 3.0)


def test_streaming_tile_suggestion_uses_memory_budget_for_deep_stack() -> None:
    tile = suggest_streaming_tile_size(
        (1, 1, 84, 2048, 2048),
        psf_xy_est=65,
        method="ci_rl",
        memory_budget_bytes=18 * 1024 ** 3,
    )

    assert 512 <= tile <= 1024
    assert tile % 64 == 0


def test_streaming_reads_regions_not_full_array() -> None:
    class CountingSource(InMemoryRegionSource):
        def __init__(self, channels, metadata):
            super().__init__(channels, metadata)
            self.read_shapes = []

        def read_region(self, **kwargs):
            out = super().read_region(**kwargs)
            self.read_shapes.append(out.shape)
            return out

    image = np.ones((40, 40), dtype=np.float32)
    source = CountingSource([image], {})
    sink = InMemoryPyramidSink(source.shape, source.metadata)

    deconvolve_streaming(
        source,
        sink,
        psf_for_channel=lambda _c: _identity_psf_2d(),
        deconvolve_tile=lambda tile, _psf, _c: tile,
        tile_yx=(16, 16),
        halo_yx=(2, 2),
        build_pyramids=False,
    )

    assert len(source.read_shapes) > 1
    assert all(shape[-2] < 40 or shape[-1] < 40 for shape in source.read_shapes)


def test_zarr_pyramid_sink_writes_multiscales_when_zarr_available(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    shape = (1, 1, 1, 32, 34)
    metadata = {
        "pixel_size_x": 0.45499,
        "pixel_size_y": 0.45499,
        "pixel_size_z": 1.25,
        "channel_names": ["CH1"],
        "channels": [
            {
                "name": "CH1",
                "color": (255, 0, 0),
                "active": True,
                "window_start": 12,
                "window_end": 345,
            }
        ],
    }
    sink = ZarrPyramidSink(tmp_path / "out.ome.zarr", shape=shape, metadata=metadata, levels=3)
    tile = np.ones((1, 32, 34), dtype=np.float32)
    sink.write_tile(t=0, c=0, z=slice(0, 1), y=slice(0, 32), x=slice(0, 34), data=tile)
    sink.build_pyramids()
    sink.validate()
    sink.close()

    import zarr

    root = zarr.open(str(tmp_path / "out.ome.zarr"), mode="r")
    assert tuple(root["0"].shape) == shape
    assert tuple(root["1"].shape)[-2:] == (16, 17)
    assert "multiscales" in root.attrs
    channel = root.attrs["omero"]["channels"][0]
    assert channel["label"] == "CH1"
    assert channel["color"] == "FF0000"
    assert channel["family"] == "linear"
    assert channel["window"]["start"] == 12.0
    assert channel["window"]["end"] == 345.0
    assert root.attrs["omero"]["rdefs"]["model"] == "color"
    scale = root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [1, 1, 1.25, 0.45499, 0.45499]
    assert root.attrs["_creator"]["physical_pixel_sizes_um"]["x"] == 0.45499
    source = ZarrRegionSource(tmp_path / "out.ome.zarr")
    assert source.metadata["channels"][0]["color"] == (255, 0, 0)
    assert source.metadata["channels"][0]["window_start"] == 12.0
    assert source.metadata["channels"][0]["window_end"] == 345.0
    manifest = json.loads((tmp_path / "out.ome.zarr" / ".cideconvolve_stream_manifest.json").read_text())
    assert manifest["shape"] == list(shape)
