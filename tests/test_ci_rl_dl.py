import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch
import tifffile

from core.deconvolve_ci_dl import (
    CONDITIONING_CHANNELS,
    GatedResidualUNet25D,
    ResidualUNet25D,
    deconvolve_ci_rl_dl,
    input_channel_count,
    load_residual_unet_checkpoint,
)
from core.deconvolve import deconvolve
from training.train import SyntheticVolumeDataset, TrainConfig, generate_training_data, generate_synthetic_gt


class CiRlDlSmokeTests(unittest.TestCase):
    def test_synthetic_gt_shape(self) -> None:
        rng = np.random.default_rng(1)
        vol = generate_synthetic_gt((5, 24, 24), rng)
        self.assertEqual(vol.shape, (5, 24, 24))
        self.assertEqual(vol.dtype, np.float32)
        self.assertGreater(float(vol.max()), 0.0)

    def test_model_forward_shape(self) -> None:
        channels = input_channel_count(1, True)
        model = ResidualUNet25D(channels, base_channels=4)
        x = torch.randn(2, channels, 24, 24)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 24, 24))

    def test_gated_model_forward_shape_and_details(self) -> None:
        channels = input_channel_count(1, True, CONDITIONING_CHANNELS)
        model = GatedResidualUNet25D(channels, base_channels=4, z_radius=1)
        x = torch.randn(2, channels, 24, 24)
        y = model(x)
        details = model.forward_details(x)
        self.assertEqual(tuple(y.shape), (2, 1, 24, 24))
        self.assertEqual(tuple(details["gate"].shape), (2, 1, 24, 24))
        self.assertTrue(torch.all(details["gate"] >= 0))
        self.assertTrue(torch.all(details["gate"] <= 1))

    def test_dataset_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = TrainConfig(
                num_volumes=3,
                volume_shape=(5, 24, 24),
                patch_size=16,
                z_context=1,
                rl_iterations=1,
                output_dir=root,
                quick_test=True,
            )
            generate_training_data(config, root)
            ds = SyntheticVolumeDataset(root, "train", patch_size=16, z_radius=1, samples_per_epoch=2, seed=3)
            item = ds[0]
            self.assertEqual(tuple(item["input"].shape), (7, 16, 16))
            self.assertEqual(tuple(item["target_residual"].shape), (1, 16, 16))

    def test_dataset_conditioning_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = TrainConfig(
                num_volumes=3,
                volume_shape=(5, 24, 24),
                patch_size=16,
                z_context=1,
                rl_iterations=1,
                output_dir=root,
                quick_test=True,
                use_conditioning=True,
            )
            generate_training_data(config, root)
            ds = SyntheticVolumeDataset(root, "train", patch_size=16, z_radius=1, samples_per_epoch=2, seed=3, use_conditioning=True)
            item = ds[0]
            self.assertEqual(tuple(item["input"].shape), (7 + len(CONDITIONING_CHANNELS), 16, 16))

    def test_rl_pool_and_psf_mismatch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = TrainConfig(
                num_volumes=4,
                volume_shape=(5, 24, 24),
                patch_size=16,
                z_context=1,
                rl_iterations=1,
                rl_iteration_pool=(1, 2),
                rl_iteration_weights=(0.0, 1.0),
                output_dir=root,
                quick_test=True,
                synthetic_artifact_level="strong",
                psf_mismatch="mild",
            )
            generate_training_data(config, root)
            sample_dirs = sorted((root / "data").glob("*/*"))
            self.assertTrue(sample_dirs)
            for sample_dir in sample_dirs:
                meta = json.loads((sample_dir / "metadata.json").read_text())
                self.assertEqual(meta["rl_requested_iterations"], 2)
                self.assertEqual(meta["rl_iteration_pool"], [1, 2])
                self.assertEqual(meta["rl_iteration_weights"], [0.0, 1.0])
                self.assertEqual(meta["noise"]["synthetic_artifact_level"], "strong")
                self.assertTrue(meta["psf_aberration"]["enabled"])
                self.assertTrue((sample_dir / f"ci_rl_iter_{meta['rl_requested_iterations']:03d}.tif").exists())
                self.assertTrue((sample_dir / "forward_psf.tif").exists())
                self.assertTrue((sample_dir / "deconv_psf.tif").exists())
                fwd = tifffile.imread(sample_dir / "forward_psf.tif")
                dec = tifffile.imread(sample_dir / "deconv_psf.tif")
                self.assertEqual(fwd.shape, dec.shape)
                self.assertGreater(float(np.mean(np.abs(fwd - dec))), 0.0)

    def test_xy_supersampling_saves_highres_training_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = TrainConfig(
                num_volumes=3,
                volume_shape=(5, 12, 14),
                patch_size=16,
                z_context=1,
                rl_iterations=1,
                output_dir=root,
                quick_test=True,
                super_sample_xy=2,
            )
            generate_training_data(config, root)
            sample_dir = sorted((root / "data").glob("*/*"))[0]
            meta = json.loads((sample_dir / "metadata.json").read_text())
            gt = tifffile.imread(sample_dir / "gt.tif")
            raw = tifffile.imread(sample_dir / "raw.tif")
            raw_observed = tifffile.imread(sample_dir / "raw_observed.tif")
            self.assertEqual(meta["super_sample_xy"], 2)
            self.assertEqual(tuple(gt.shape), (5, 24, 28))
            self.assertEqual(tuple(raw.shape), (5, 24, 28))
            self.assertEqual(tuple(raw_observed.shape), (5, 12, 14))

    def test_legacy_checkpoint_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            model = ResidualUNet25D(input_channel_count(1, True), base_channels=4)
            torch.save({"model_type": "ResidualUNet25D", "model_kwargs": {"input_channels": 7, "base_channels": 4}, "state_dict": model.state_dict()}, path)
            loaded, checkpoint = load_residual_unet_checkpoint(path, device="cpu")
            self.assertIsInstance(loaded, ResidualUNet25D)
            self.assertEqual(checkpoint["model_type"], "ResidualUNet25D")

    def test_ci_rl_dl_without_model_returns_rl_smoke(self) -> None:
        image = np.zeros((3, 16, 16), dtype=np.float32)
        image[:, 7:9, 7:9] = 10
        psf = np.zeros((3, 5, 5), dtype=np.float32)
        psf[1, 2, 2] = 1
        out = deconvolve_ci_rl_dl(
            image,
            psf,
            rl_kwargs={
                "niter": 1,
                "convergence": "fixed",
                "tiling": "none",
                "device": "cpu",
                "two_d_mode": "legacy_2d",
                "start": "observed",
            },
            return_diagnostics=True,
        )
        self.assertEqual(out["result"].shape, image.shape)
        self.assertTrue(np.all(out["result"] >= 0))
        self.assertEqual(out["diagnostics"]["dl_refinement"], "skipped_no_model_path")

    def test_ci_rl_dl_multichannel_without_model(self) -> None:
        image = np.zeros((2, 3, 12, 12), dtype=np.float32)
        image[:, :, 5:7, 5:7] = 5
        psf = np.zeros((3, 3, 3), dtype=np.float32)
        psf[1, 1, 1] = 1
        out = deconvolve_ci_rl_dl(
            image,
            psf,
            rl_kwargs={
                "niter": 1,
                "convergence": "fixed",
                "tiling": "none",
                "device": "cpu",
                "two_d_mode": "legacy_2d",
                "start": "observed",
            },
        )
        self.assertEqual(out.shape, image.shape)
        self.assertTrue(np.all(out >= 0))

    def test_deconvolve_method_ci_rl_dl_no_model(self) -> None:
        image = np.zeros((3, 12, 12), dtype=np.float32)
        image[:, 5:7, 5:7] = 5
        psf = np.zeros((3, 3, 3), dtype=np.float32)
        psf[1, 1, 1] = 1
        out = deconvolve(
            image,
            psf,
            method="ci_rl_dl",
            niter=1,
            convergence="fixed",
            device="cpu",
            two_d_mode="legacy_2d",
            start="observed",
        )
        self.assertEqual(out.shape, image.shape)
        self.assertTrue(np.all(out >= 0))


if __name__ == "__main__":
    unittest.main()
