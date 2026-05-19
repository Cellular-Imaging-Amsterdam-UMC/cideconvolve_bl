import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.deconvolve import _pixel_size_to_backend_nm, deconvolve
from core.deconvolve_ci import _estimate_background_local_plane


def test_metadata_pixel_sizes_are_normalized_to_backend_nm() -> None:
    assert _pixel_size_to_backend_nm(0.067) == pytest.approx(67.0)
    assert _pixel_size_to_backend_nm(67.0) == pytest.approx(67.0)
    assert _pixel_size_to_backend_nm(None) is None


def test_local_background_accepts_metadata_um_pixel_size() -> None:
    image = torch.ones((443, 374), dtype=torch.float32)
    bg = _estimate_background_local_plane(image, 0.067, radius_um=0.5)
    assert bg == pytest.approx(1.0)


def test_deconvolve_2d_widefield_accepts_metadata_um_pixel_size() -> None:
    image = np.full((32, 32), 100.0, dtype=np.float32)
    image[14:18, 14:18] = 200.0
    psf = np.zeros((3, 3, 3), dtype=np.float32)
    psf[1, 1, 1] = 1.0

    out = deconvolve(
        image,
        psf,
        method="ci_rl",
        niter=1,
        background="auto",
        damping=0.0,
        offset=0.0,
        start="observed",
        convergence="fixed",
        pixel_size_xy=0.067,
        pixel_size_z=0.2883,
        microscope_type="widefield",
        two_d_mode="auto",
        device="cpu",
    )

    assert out.shape == image.shape
    assert np.all(np.isfinite(out))
