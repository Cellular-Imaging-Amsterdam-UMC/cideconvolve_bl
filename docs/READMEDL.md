# CIDeconvolve — DL Refinement (`ci_rl_dl`) and Training

> **Back to [README.md](README.md)**

This document covers the experimental `ci_rl_dl` deep learning refinement method, training pipeline, and Python API.

---

## `ci_rl_dl` — Experimental 2.5D DL Refinement

`ci_rl_dl` is an experimental research pipeline in `deconvolve_ci_dl.py`. It runs the existing `ci_rl` deconvolution first, then optionally applies a small 2.5D residual U-Net that predicts a correction to the central plane. New V2 training runs use a gated, bounded residual model by default so the correction is constrained relative to the local `ci_rl` signal:

```text
raw image -> ci_rl -> gated residual U-Net -> clamp(result >= 0)
```

This is not a replacement for physics-based deconvolution. Treat trained models as sample- and microscope-specific refiners, and validate carefully before using results for quantitative microscopy. The first model family is deliberately small: a 2.5D residual U-Net inspired by U-Net image-to-image architectures, residual learning, and the broader idea that incorporating image formation / Richardson-Lucy structure into deep learning can improve microscopy deconvolution performance.

---

## Training

### Quick training smoke test

```bash
python training/train.py --quick-test --output-dir training_runs/quick_test
```

### Full synthetic experiment preset

```bash
python training/train.py \
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

The script-level full preset uses 1000 synthetic volumes by default, richer synthetic structures, whole-volume train/validation/test splits, high-iteration `ci_rl` preprocessing, mild/moderate PSF mismatch, scalar conditioning channels, and the gated residual model. It is still useful for custom experiments, but the GUI presets are now limited to the two default-model recipes. You can still override `--num-volumes`, `--volume-shape`, `--patch-size`, `--epochs`, and `--steps`.

V2 checkpoints store their conditioning and recommended inference settings in the sidecar JSON next to the model. The conditioning channels are compact metadata planes, not full PSF images: RL iteration count, PSF width summaries, pixel sizes, NA, wavelength, and microscope type for mixed models. Old `ResidualUNet25D` checkpoints remain loadable.

### Faster GPU training

Use multiple DataLoader workers and the per-worker volume cache:

```bash
python training/train.py --full-experiment --device cuda \
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

For custom strong-restoration experiments, the gate can be kept while making
the residual bound looser than the original conservative model. This is meant
to learn real restoration after strong `ci_rl`, not only a tiny polish. A
foreground intensity-retention loss can be enabled to discourage the model from
darkening biological signal too aggressively while still allowing
artifact/background correction. Training
logs include global train/validation losses plus validation losses by
morphology bucket (`generic`, `dna`, `mitotic`, `membrane`, `actin`,
`dendrite`, and `puncta`) where those structures are present. Per-bucket
example montages are written under the run's `examples/buckets/` folder.

### XY supersampling

Experimental XY supersampling is available with `--super-sample-xy 2`. In this
mode the clean GT and PSFs are generated on a 2x finer XY grid, the blurred
high-resolution image is averaged down to the simulated camera grid before
noise and offset are added, and the noisy camera raw image is then upsampled
before high-resolution `ci_rl` preprocessing and DL training. The current
default-model GUI presets both use `--super-sample-xy 2`.

### Synthetic data generator

The full synthetic generator includes sparse spots and vesicles as well as
DAPI/chromatin-like nuclei, mitotic spindle/fiber fields, membrane/cytoplasm
signal, sheets, and diffuse biological background. Conservative training can
include a stronger negative-residual guard:

```bash
--negative-residual-weight 0.05
--max-negative-residual-fraction 0.25
```

Older strong-restoration experiments reduced that brake to `0.01` and allowed
larger negative corrections. The current default-model GUI presets use `0.05`
with a lower maximum negative residual fraction, because preserving total
energy and background offset is more important for the bundled models.

### Small first experiment

```bash
python training/train.py \
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
python gui/gui_train.py
```

The GUI now intentionally exposes only the two presets used for the bundled
default models:

| GUI preset | Default model folder | Original run |
|---|---|---|
| `Large widefield standard long` | `models/defaultwidefield` | `U:\runs\gui_large_widefield_standard_long_v2_widefield` |
| `Large confocal standard long` | `models/defaultconfocal` | `U:\runs\gui_large_confocal_standard_long_v2_confocal` |

These presets use standard artifact severity, XY supersampling, tighter
intensity-retention and global-intensity losses, background-offset loss, and
training-time XY padding so the model can refine structure without
systematically changing background offset or total image energy. The run queue
supports sequential training, optional pruning of generated data after each run
while keeping a small evaluation subset, and automatic evaluation on retained
synthetic samples plus matching local OME-TIFF data.

### Training parameter reference

The GUI fields map directly to `training/train.py` flags. The two default-model
presets are just saved values for these parameters.

#### Presets and output

| GUI field / flag | Description |
|---|---|
| Preset | Selects one of the two default-model recipes: widefield or confocal standard long. |
| Output folder / `--output-dir` | Training run directory. The GUI appends the selected microscope suffix to the preset output name when the output path is still automatic. |
| `--quick-test` | Script-only smoke-test shortcut for a tiny run. Not exposed as a GUI preset anymore. |
| `--full-experiment` | Script-only preset shortcut for the older broad experiment recipe. |
| `--seed` | Random seed. The GUI uses the script default, `42`. |

#### Synthetic data

| GUI field / flag | Description |
|---|---|
| Volumes / `--num-volumes` | Number of synthetic volumes generated before patch sampling. |
| Volume shape Z,Y,X / `--volume-shape` | Synthetic volume size before patch extraction. |
| Synthetic complexity / `--synthetic-complexity` | `standard` or `full`; `full` enables richer morphology and background structure. |
| Artifact level / `--synthetic-artifact-level` | `standard` or `strong`; controls simulated acquisition artifacts. |
| XY supersampling / `--super-sample-xy` | Generates clean GT/PSF on a finer XY grid before camera-grid simulation. The default models use `2`. |
| `--super-sample-z` | Script flag reserved for future axial supersampling; currently fixed to `1`. |
| Synthetic morphology / `--synthetic-morphology` | Morphology family sampled by the generator, or `mixed` for all families. |
| Microscope type / `--microscope-type` | `widefield`, `confocal`, or `mixed`. The bundled defaults use separate widefield and confocal runs. |
| PSF mismatch / `--psf-mismatch` | Adds synthetic PSF error: `none`, `mild`, or `moderate`. |
| Moderate mismatch frac. / `--psf-mismatch-moderate-fraction` | Fraction of mild-mismatch samples that are promoted to moderate mismatch. |

#### Patch sampling and optimization

| GUI field / flag | Description |
|---|---|
| Patch size / `--patch-size` | XY crop size used for DL training patches. |
| Z context / `--z-context` | Number of neighboring planes on each side of the central plane. It determines the 2.5D input depth. |
| Batch size / `--batch-size` | DL training batch size. Increase until VRAM is efficiently used without spilling. |
| Epochs / `--epochs` | Number of full training epochs. |
| Steps / `--steps` | Optional hard limit on optimizer steps. `0` in the GUI means epoch-based training. |
| Train samples/epoch / `--train-samples-per-epoch` | Number of random patches sampled each epoch. |
| Validation samples / `--val-samples` | Number of validation patches used at each validation point. |
| Learning rate / `--learning-rate` | Optimizer learning rate. The default-model presets use `4e-4`. |
| Device / `--device` | `auto`, `cuda`, or `cpu`. |
| Mixed precision / `--mixed-precision` / `--no-mixed-precision` | Enables AMP on CUDA to reduce VRAM use and improve speed. |

#### RL preprocessing

| GUI field / flag | Description |
|---|---|
| RL iterations / `--rl-iterations` | Fallback CI-RL iteration count when no pool is used. |
| RL iteration pool / `--rl-iteration-pool` | Comma-separated CI-RL iteration counts sampled per synthetic volume. |
| RL iteration weights / `--rl-iteration-weights` | Sampling weights matching the iteration pool. The default models use `0.35,0.40,0.25` for `50,80,100`. |

#### Losses and intensity control

| GUI field / flag | Description |
|---|---|
| Reconvolution weight / `--reconvolution-weight` | Penalizes mismatch after reconvolving the refined output with the PSF. |
| Gradient weight / `--gradient-weight` | Encourages structural/edge agreement with the target. |
| Neg. residual weight / `--negative-residual-weight` | Penalizes overly negative DL residuals in signal regions. |
| Max neg. residual frac. / `--max-negative-residual-fraction` | Allowed negative correction as a fraction of normalized CI-RL signal before the penalty grows. |
| Intensity retention / `--intensity-retention-weight` | Foreground intensity-ratio penalty weight. |
| Retention min / max / `--intensity-retention-min`, `--intensity-retention-max` | Accepted foreground intensity-ratio range. The default models use `0.98` to `1.02`. |
| Global intensity / `--global-intensity-weight` | Total image-energy penalty weight. |
| Global min / max / `--global-intensity-min`, `--global-intensity-max` | Accepted total energy ratio. The default models use `0.99` to `1.01`. |
| Background offset / `--background-offset-weight` | Penalizes mean residual offset in background-like pixels. |
| Training XY padding / `--training-xy-padding` | Reflect-padding halo for DL training patches. The loss is cropped to the central patch to reduce border learning artifacts. |

#### Model and conditioning

| GUI field / flag | Description |
|---|---|
| Model type / `--model-type` | `GatedResidualUNet25D` is the current default; `ResidualUNet25D` remains loadable for older experiments. |
| Base channels / `--base-channels` | Width of the U-Net. Higher values increase capacity and VRAM use. |
| Conditioning / `--use-conditioning` / `--no-use-conditioning` | Adds scalar conditioning planes for RL iterations, PSF summaries, pixel size, NA, wavelength, and microscope class. |
| Residual scale / `--residual-scale` | Fixed multiplier on the predicted residual during training/inference. |
| Residual bound frac. / `--residual-bound-fraction` | Bounds residual magnitude relative to local normalized CI-RL signal. |
| Residual bound scale / `--residual-bound-scale` | Absolute residual-bound component. |

#### Parallel loading and queue options

| GUI field / flag | Description |
|---|---|
| CPU workers / `--num-workers` | CPU workers for synthetic raw volume generation; `0` auto-selects. |
| Loader workers / `--data-loader-workers` | PyTorch DataLoader workers; `0` auto-selects for CUDA and disables workers for CPU. |
| Volume cache/worker / `--volume-cache-size` | Number of loaded synthetic volumes cached per worker. |
| Prune data after each run | GUI-only queue option. Removes most generated samples after training while keeping a small evaluation subset. |
| Evaluate after each run | GUI-only queue option. Runs `tests/evaluate_ci_rl_dl_model.py` on retained synthetic samples and matching local OME-TIFF data after a run finishes. |

---

## Python API

Use a trained model directly:

```python
from deconvolve_ci_dl import deconvolve_ci_rl_dl

out = deconvolve_ci_rl_dl(
    raw_volume,
    psf,
    model_path="training_runs/quick_test/checkpoints/best_model.pt",
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
    dl_model_path="training_runs/quick_test/checkpoints/best_model.pt",
    dl_z_context=1,
)
```

---

## References

- **U-Net Architecture:** Ronneberger, O., Fischer, P. & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 234–241. [doi:10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28)
- **Residual Learning:** He, K., Zhang, X., Ren, S. & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*, 770–778. [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
- **Richardson-Lucy Network / image-formation-aware DL:** Li, Y. et al. (2022). "Incorporating the image formation process into deep learning improves network performance." *Nature Methods* **19**, 1427-1437. [doi:10.1038/s41592-022-01652-7](https://www.nature.com/articles/s41592-022-01652-7)
