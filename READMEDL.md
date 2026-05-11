# CIDeconvolve — DL Refinement (`ci_rl_dl`) and Training

> **Back to [README.md](README.md)**

This document covers the experimental `ci_rl_dl` deep learning refinement method, training pipeline, and Python API.

---

## `ci_rl_dl` — Experimental 2.5D DL Refinement

`ci_rl_dl` is an experimental research pipeline in `deconvolve_ci_dl.py`. It runs the existing `ci_rl` deconvolution first, then optionally applies a small 2.5D residual U-Net that predicts a correction to the central plane. New V2 training runs use a gated, bounded residual model by default so the correction is constrained relative to the local `ci_rl` signal:

```text
raw image -> ci_rl -> gated residual U-Net -> clamp(result >= 0)
```

This is not a replacement for physics-based deconvolution. Treat trained models as sample- and microscope-specific refiners, and validate carefully before using results for quantitative microscopy. The first model family is deliberately small: a 2.5D residual U-Net inspired by U-Net image-to-image architectures and residual learning, trained on synthetic microscopy-like volumes generated from the same PSF/RL stack.

---

## Training

### Quick training smoke test

```bash
python train.py --quick-test --output-dir training_runs/quick_test
```

### Full synthetic experiment preset

```bash
python train.py \
    --full-experiment \
    --output-dir training_runs/full_ci_rl_dl \
    --device cuda \
    --mixed-precision \
    --rl-iteration-pool 50,80,100 \
    --rl-iteration-weights 0.35,0.40,0.25 \
    --psf-mismatch mild \
    --psf-mismatch-moderate-fraction 0.5 \
    --synthetic-artifact-level strong \
    --base-channels 48 \
    --residual-bound-fraction 0.75 \
    --residual-bound-scale 0.10 \
    --negative-residual-weight 0.01 \
    --max-negative-residual-fraction 0.50 \
    --intensity-retention-weight 0.05 \
    --gradient-weight 0.08 \
    --model-type GatedResidualUNet25D \
    --use-conditioning \
    --reconvolution-weight 0.02
```

The full preset uses 1000 synthetic volumes by default, richer synthetic structures, whole-volume train/validation/test splits, high-iteration `ci_rl` preprocessing, mild/moderate PSF mismatch, scalar conditioning channels, and the gated residual model. The current strong-restoration settings intentionally train on harder post-RL failure modes: PSF error, coma/astigmatism-like aberration, uneven illumination, haze/background offsets, row/column banding, read-noise outliers, hot pixels, clipping, ringing-prone PSF mismatch, and noisy texture. You can still override `--num-volumes`, `--volume-shape`, `--patch-size`, `--epochs`, and `--steps`.

V2 checkpoints store their conditioning and recommended inference settings in the sidecar JSON next to the model. The conditioning channels are compact metadata planes, not full PSF images: RL iteration count, PSF width summaries, pixel sizes, NA, wavelength, and microscope type for mixed models. Old `ResidualUNet25D` checkpoints remain loadable.

### Faster GPU training

Use multiple DataLoader workers and the per-worker volume cache:

```bash
python train.py --full-experiment --device cuda \
    --batch-size 16 \
    --data-loader-workers 8 \
    --volume-cache-size 8
```

Increasing `--batch-size` improves GPU occupancy but reduces optimizer updates
per epoch if `--train-samples-per-epoch` is unchanged. Increase
`--train-samples-per-epoch` proportionally when you want the same number of
gradient updates per epoch.

### Residual gating and bounds

The residual correction can be constrained with `--residual-scale`. The default
`1.0` preserves the current inference strength. For new gated checkpoints, the
network also predicts a gate and bounds the residual with
`--residual-bound-fraction` and `--residual-bound-scale`:

```text
refined = clamp(ci_rl + residual_scale * gate * bounded_residual, min=0)
```

For the strong V4 presets, the gate is kept but the residual bound is looser
than the original conservative model. This is meant to learn real restoration
after strong `ci_rl`, not only a tiny polish. A foreground intensity-retention
loss can be enabled to discourage the model from darkening biological signal
too aggressively while still allowing artifact/background correction. Training
logs include global train/validation losses plus validation losses by
morphology bucket (`generic`, `dna`, `mitotic`, `membrane`, `actin`,
`dendrite`, and `puncta`) where those structures are present. Per-bucket
example montages are written under the run's `examples/buckets/` folder.

### XY supersampling

Experimental XY supersampling is available with `--super-sample-xy 2`. In this
mode the clean GT and PSFs are generated on a 2x finer XY grid, the blurred
high-resolution image is averaged down to the simulated camera grid before
noise and offset are added, and the noisy camera raw image is then upsampled
before high-resolution `ci_rl` preprocessing and DL training. The GUI includes
a `Large widefield strong XY2` preset for this first hires experiment.

### Synthetic data generator

The full synthetic generator includes sparse spots and vesicles as well as
DAPI/chromatin-like nuclei, mitotic spindle/fiber fields, membrane/cytoplasm
signal, sheets, and diffuse biological background. Conservative training can
include a stronger negative-residual guard:

```bash
--negative-residual-weight 0.05
--max-negative-residual-fraction 0.25
```

The strong V4 presets reduce that brake to `0.01` and allow larger negative
corrections, because the generator now has more realistic GT and artifact
coverage.

### Small first experiment

```bash
python train.py \
    --num-volumes 24 \
    --volume-shape 16,96,96 \
    --patch-size 64 \
    --z-context 2 \
    --batch-size 4 \
    --epochs 2 \
    --output-dir training_runs/small_ci_rl_dl
```

### Training GUI

The training GUI can be launched with:

```bash
python gui_train.py
```

The GUI includes separate `Medium widefield strong`, `Medium confocal strong`,
`Large widefield strong`, `Large confocal strong`, and `Large mixed strong`
presets. The microscope-specific presets are the recommended first serious
runs; the mixed preset is useful for comparison and robustness checks.

---

## Python API

Use a trained model directly:

```python
from deconvolve_ci_dl import deconvolve_ci_rl_dl

out = deconvolve_ci_rl_dl(
    raw_volume,
    psf,
    model_path="training_runs/quick_test/checkpoints/final_model.pt",
    rl_kwargs={"niter": 20},
    return_diagnostics=True,
)
refined = out["result"]
```

Use the method through the main deconvolution API:

```python
from deconvolve import deconvolve

refined = deconvolve(
    raw_volume,
    psf,
    method="ci_rl_dl",
    niter=20,
    dl_model_path="training_runs/quick_test/checkpoints/final_model.pt",
    dl_z_context=1,
)
```

---

## References

- **U-Net Architecture:** Ronneberger, O., Fischer, P. & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 234–241. [doi:10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28)
- **Residual Learning:** He, K., Zhang, X., Ren, S. & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*, 770–778. [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
