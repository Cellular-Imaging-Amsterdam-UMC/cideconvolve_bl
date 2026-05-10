# PLAN: Add `ci_rl_dl` 2.5D deep-learning refinement to `cideconvolve`

## Goal

Implement an optional deep-learning refinement layer for the existing `cideconvolve` Richardson-Lucy workflow.

The first implementation should be deliberately conservative:

```text
raw image
  -> existing cideconvolve `ci_rl`
  -> 2.5D residual refinement model
  -> final image
```

The DL model should not replace physics-based deconvolution. It should learn a small residual correction after classical convergence, while preserving a reconvolution consistency check against the measured raw image.

Initial method name:

```text
ci_rl_dl
```

Proposed new files:

```text
cideconvolve/deconvolve_ci_dl.py
cideconvolve/train.py
cideconvolve/gui_train.py
```

Optional later files:

```text
cideconvolve/dl_models.py
cideconvolve/synthetic_gt.py
cideconvolve/dl_dataset.py
cideconvolve/dl_losses.py
cideconvolve/dl_metrics.py
examples/train_ci_rl_dl_example.py
examples/run_ci_rl_dl_example.py
```

The implementation may look at Richardson-Lucy-Net / RLN for inspiration, but the first version should use the simpler preferred strategy here: **2.5D residual U-Net refinement after existing `ci_rl` convergence**.

---

## High-level design

### Current classical part

Use existing functions from `deconvolve_ci.py` wherever possible:

- PSF generation / PSF loading
- padding and cropping helpers
- Richardson-Lucy update logic
- background / offset handling
- convergence diagnostics
- image I/O helpers, if already present
- metadata-derived microscope parameters

Do not duplicate PSF or core RL logic unless absolutely necessary.

### New DL part

The first DL model receives:

```text
raw 2.5D stack around z
ci_rl deconvolved 2.5D stack around z
optional residual / ratio channel for central plane
optional forward projection consistency channel
```

It predicts either:

```text
residual correction for central z plane
```

or, optionally later:

```text
refined central z plane directly
```

Preferred first version:

```python
x_final_z = clamp(x_ci_rl_z + residual_z, min=0)
```

This is safer than a direct image-to-image model because the physics-based result remains the baseline.

---

## Phase 0 — Repository inspection and minimal integration plan

### Tasks

- [ ] Inspect the current `cideconvolve` package structure.
- [ ] Identify the public or internal function that runs `ci_rl`.
- [ ] Identify existing PSF generation/loading functions.
- [ ] Identify existing tensor/image layout conventions:
  - [ ] 2D vs 3D shape
  - [ ] channel dimension position
  - [ ] z/y/x order
  - [ ] dtype conventions
  - [ ] CPU/GPU device handling
- [ ] Identify current CLI entry points.
- [ ] Decide whether `deconvolve_ci_dl.py` should expose:
  - [ ] a function API only
  - [ ] a CLI hook
  - [ ] both

### Acceptance criteria

- [ ] `ci_rl` can be called from the new file without changing existing behavior.
- [ ] A classical `ci_rl` run produces identical output before and after adding the new file.

---

## Phase 1 — Add `deconvolve_ci_dl.py`

### Purpose

Create a new module that wraps existing `ci_rl` and optionally applies a trained 2.5D DL refinement model.

### Main public functions

Suggested API:

```python
def deconvolve_ci_rl_dl(
    image,
    psf=None,
    model_path=None,
    optical_params=None,
    device="auto",
    rl_kwargs=None,
    dl_kwargs=None,
    return_diagnostics=False,
):
    """Run ci_rl followed by optional 2.5D DL residual refinement."""
```

Also add a lower-level function:

```python
def apply_dl_refinement_25d(
    raw,
    deconv,
    model,
    psf=None,
    device="auto",
    z_radius=2,
    batch_size=8,
    use_ratio_channel=True,
    use_forward_projection_channel=False,
    clamp_nonnegative=True,
):
    """Apply 2.5D residual refinement slice-by-slice."""
```

### Input shape convention

Pick one internal convention and document it clearly. Recommended internal convention:

```text
Z, Y, X
```

For 2D images, temporarily promote to:

```text
1, Y, X
```

For multichannel data, handle one channel at a time in the first version. Do not build multichannel support until the single-channel path is stable.

### 2.5D window logic

For each z plane:

```text
z window = z-radius ... z+radius
```

For borders, use reflection padding or edge repetition.

Recommended first version:

```text
z_radius = 2
number of z planes = 5
```

Input channels when `use_ratio_channel=True`:

```text
5 raw planes
5 ci_rl planes
1 ratio/residual plane
```

Total input channels:

```text
11
```

Ratio channel:

```python
ratio = raw_z / (reconvolved_deconv_z + eps)
```

If reconvolution is not implemented in the first commit, use a simpler residual channel:

```python
residual = raw_z - normalize_like_raw(deconv_z)
```

But reconvolution is preferred.

### Model loading

Use PyTorch.

Suggested checkpoint structure:

```python
{
    "model_type": "ResidualUNet25D",
    "model_kwargs": {...},
    "state_dict": model.state_dict(),
    "normalization": {...},
    "z_radius": 2,
    "input_channels": 11,
    "target": "residual",
    "version": 1,
}
```

### Diagnostics

Return diagnostics such as:

```text
rl_iterations
rl_convergence_metric
dl_model_path
dl_input_channels
dl_z_radius
reconvolution_error_before
reconvolution_error_after
intensity_sum_before_dl
intensity_sum_after_dl
```

### Acceptance criteria

- [ ] Running without `model_path` returns normal `ci_rl` output.
- [ ] Running with a dummy identity model changes nothing.
- [ ] Running with a random model completes without shape errors.
- [ ] Output shape equals input image shape.
- [ ] Output is non-negative by default.
- [ ] Diagnostics can be returned as a dictionary.

---

## Phase 2 — Add model architecture

This can initially live in `train.py`, but it is cleaner to create:

```text
cideconvolve/dl_models.py
```

If you want to keep the first implementation compact, define the model inside `train.py` and move it later.

### Preferred first model: small 2.5D residual U-Net

Model name:

```python
ResidualUNet25D
```

Input:

```text
N, C, Y, X
```

Output:

```text
N, 1, Y, X
```

Recommended architecture:

```text
encoder block 1: 32 channels
encoder block 2: 64 channels
encoder block 3: 128 channels
bottleneck:      256 channels
up block 2:      128 channels
up block 1:      64 channels
final:           1 channel residual
```

Use:

- Conv2D
- GroupNorm or InstanceNorm
- SiLU or ReLU
- skip connections
- bilinear upsampling or transposed convolution

Avoid an overly large model in the first version. The goal is a practical enhancer, not a hallucination-prone super-resolution model.

### Optional: residual output scaling

Add a small learnable or fixed residual scale:

```python
x_final = x_deconv + residual_scale * predicted_residual
```

Start with:

```text
residual_scale = 0.1 or 0.25
```

or train the model to output residuals in normalized units and clip the result.

### Acceptance criteria

- [ ] Model accepts `N,C,Y,X` and outputs `N,1,Y,X`.
- [ ] Model can run in mixed precision.
- [ ] Model can be exported and reloaded from a checkpoint.
- [ ] Inference works on CPU and CUDA.

---

## Phase 3 — Add synthetic ground-truth generation in `train.py`

### Purpose

Generate approximately 1000 synthetic 3D microscopy-like GT volumes, then generate blurred/noisy raw images and classical `ci_rl` deconvolved images.

The generated training folder should contain reproducible data and metadata.

### Folder layout

Recommended output structure:

```text
training_data_ci_rl_dl/
  config.yaml
  dataset_summary.json
  train/
    sample_000001/
      gt.tif
      raw.tif
      ci_rl.tif
      psf.tif
      metadata.json
    ...
  val/
    sample_000001/
      gt.tif
      raw.tif
      ci_rl.tif
      psf.tif
      metadata.json
  test/
    sample_000001/
      gt.tif
      raw.tif
      ci_rl.tif
      psf.tif
      metadata.json
  checkpoints/
  logs/
  figures/
```

### Synthetic GT volume types

Each volume should be a random mixture of microscopy-like structures:

- [ ] diffraction-limited spots
- [ ] small clustered spots
- [ ] larger Gaussian blobs
- [ ] nuclei-like ellipsoids
- [ ] hollow ring / vesicle-like structures
- [ ] membrane-like sheets or outlines
- [ ] short filaments
- [ ] diffuse cytoplasmic background
- [ ] sparse and dense regions
- [ ] empty/background-only regions

### Volume size

Recommended first version:

```text
Z = 32 or 48
Y = 256
X = 256
```

For faster testing:

```text
Z = 16
Y = 128
X = 128
```

For final first experiment:

```text
1000 volumes of 32 x 256 x 256
```

### Optical parameter randomisation

Randomise per volume:

```text
NA
emission wavelength
pixel size XY
z-step
microscope type: widefield first; confocal later
background level
Poisson noise level
Gaussian read noise
PSF mismatch
object density
object brightness distribution
```

First implementation can restrict to widefield-like PSFs if that is easiest.

### Raw image generation

For each GT volume:

```text
gt -> convolve with PSF -> add background -> add Poisson noise -> add Gaussian noise -> raw
```

Then run existing `ci_rl`:

```text
raw + psf -> ci_rl
```

Save all four core artifacts:

```text
gt.tif
raw.tif
ci_rl.tif
psf.tif
```

### Split policy

Split by full volume, not by patch:

```text
train: 80%
val:   10%
test:  10%
```

Do not randomly split patches from the same volume into train and validation.

### Acceptance criteria

- [ ] `train.py --generate-data` creates a complete training folder.
- [ ] Generation is reproducible with a random seed.
- [ ] `dataset_summary.json` records parameter ranges and number of samples.
- [ ] At least one montage figure is saved showing GT, raw, ci_rl and difference images.
- [ ] No validation/test volume is derived from a training volume.

---

## Phase 4 — Dataset and patch extraction

### Training samples

For each training step, load a random volume and extract a random 2.5D patch.

Suggested patch sizes:

```text
Y, X = 128, 128 for first tests
Y, X = 256, 256 for better model
z_radius = 2
```

Input tensor:

```text
raw z-window
ci_rl z-window
ratio/residual central channel
```

Target tensor:

Preferred:

```text
target residual = gt_z - ci_rl_z
```

Alternative:

```text
target image = gt_z
```

First version should use residual target.

### Normalisation

Use robust per-volume or per-patch normalisation. Save the choice in the checkpoint.

Recommended first version:

```text
scale = percentile(raw, 99.8)
raw_norm = raw / scale
ci_rl_norm = ci_rl / scale
gt_norm = gt / scale
```

Clip to a reasonable range, for example:

```text
0 to 2 or 0 to 4
```

### Data augmentation

Use simple augmentations:

- [ ] flips in X/Y
- [ ] 90-degree rotations
- [ ] intensity scaling
- [ ] mild extra noise
- [ ] optional z reversal only if physically acceptable

Avoid aggressive elastic deformation in the first version.

### Acceptance criteria

- [ ] Dataset returns tensors with stable shapes.
- [ ] Random patch extraction does not cross invalid z boundaries.
- [ ] The target residual reconstructs `gt_z` when added to `ci_rl_z`.
- [ ] A debug function saves a few input/target patch montages.

---

## Phase 5 — Loss functions

### First loss

Start simple:

```text
L = L1(x_final, gt)
```

where:

```text
x_final = ci_rl + predicted_residual
```

### Add SSIM or MS-SSIM later

Second version:

```text
L = L1 + 0.1 * SSIM_loss
```

### Add PSF reconvolution consistency

Important physics constraint:

```text
PSF * x_final should explain raw
```

Add:

```text
L_consistency = L1(convolve(x_final, psf), raw)
```

Final preferred loss:

```text
L = L1(x_final, gt)
  + lambda_ssim * SSIM_loss(x_final, gt)
  + lambda_consistency * L1(PSF * x_final, raw)
  + lambda_intensity * abs(sum(x_final) - sum(gt)) / sum(gt)
```

Suggested starting weights:

```text
lambda_ssim = 0.05 to 0.1
lambda_consistency = 0.05 to 0.2
lambda_intensity = 0.01
```

For the very first run, use only L1. Add the physics consistency after the training loop is stable.

### Acceptance criteria

- [ ] L1-only training decreases training and validation loss.
- [ ] Adding consistency loss does not destabilise training.
- [ ] Loss curves are saved as PNG and CSV.

---

## Phase 6 — Training loop in `train.py`

### Required CLI modes

`train.py` should support at least:

```bash
python -m cideconvolve.train --generate-data --out training_data_ci_rl_dl --n-volumes 1000
```

```bash
python -m cideconvolve.train --train --data training_data_ci_rl_dl --out training_data_ci_rl_dl/checkpoints
```

```bash
python -m cideconvolve.train --generate-data --train --out training_data_ci_rl_dl --n-volumes 1000
```

```bash
python -m cideconvolve.train --evaluate --data training_data_ci_rl_dl --checkpoint path/to/best.pt
```

### Training defaults

Recommended first defaults:

```text
model: ResidualUNet25D
z_radius: 2
patch_size: 128
batch_size: auto or 8
steps: 100000
validation_interval: 1000
checkpoint_interval: 5000
mixed_precision: true
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 1e-5
scheduler: cosine or ReduceLROnPlateau
```

### Graph output

During training save:

```text
loss_curve.png
loss_curve.csv
validation_examples_epoch_or_step_*.png
```

The graph should show:

```text
training loss
validation loss
optional L1 / SSIM / consistency components
```

Use matplotlib. Do not require a GUI for `train.py`.

### Checkpoints

Save:

```text
last.pt
best_val.pt
step_XXXXX.pt
```

Each checkpoint should include:

```python
{
    "model_type": "ResidualUNet25D",
    "model_kwargs": {...},
    "state_dict": ...,
    "optimizer_state_dict": ...,
    "step": ...,
    "best_val_loss": ...,
    "normalization": ...,
    "z_radius": ...,
    "input_channels": ...,
    "loss_config": ...,
    "data_config": ...,
}
```

### Acceptance criteria

- [ ] Training can resume from `last.pt`.
- [ ] Best validation checkpoint is saved.
- [ ] Loss graph is updated regularly.
- [ ] Example prediction montages are saved.
- [ ] Training can run without PyQt installed.

---

## Phase 7 — `gui_train.py`

### Purpose

Create a simple GUI launcher for synthetic data generation, training, and visual monitoring.

Use PyQt6 if that is already acceptable for the project.

### GUI features

Minimum controls:

- [ ] output training folder
- [ ] number of volumes
- [ ] volume size Z/Y/X
- [ ] patch size
- [ ] z-radius
- [ ] batch size
- [ ] number of steps
- [ ] learning rate
- [ ] generate data button
- [ ] start training button
- [ ] stop training button
- [ ] resume from checkpoint
- [ ] select GPU / CPU
- [ ] open loss graph
- [ ] preview random generated GT/raw/ci_rl sample
- [ ] preview current validation predictions

### Implementation recommendation

Do not duplicate training logic in the GUI. The GUI should call functions from `train.py` or launch `train.py` as a subprocess.

Preferred first version:

```text
gui_train.py launches train.py as subprocess and streams stdout/stderr into a text box
```

This is simpler and safer than putting the full training loop inside the GUI thread.

### Acceptance criteria

- [ ] GUI can generate a small test dataset.
- [ ] GUI can start training.
- [ ] GUI stays responsive during training.
- [ ] GUI can stop the subprocess.
- [ ] GUI can open the latest loss plot and validation montage.

---

## Phase 8 — Evaluation and benchmark integration

### Evaluation metrics

For synthetic test data, compute:

```text
L1 / MAE
MSE / RMSE
PSNR
SSIM
NRMSE
Fourier shell/ring correlation if available
intensity conservation error
reconvolution consistency error
```

For microscopy-style downstream validation, add later:

```text
spot count stability
segmentation stability
bead FWHM
false-positive spot rate
background noise level
z-profile width
```

### Compare methods

At minimum compare:

```text
raw
ci_rl
ci_rl_dl
```

Later compare:

```text
ci_rl_tv
ci_sparse_hessian
ci_rl_dl
```

### Required plots

Save to:

```text
training_data_ci_rl_dl/figures/evaluation/
```

Plots:

- [ ] metric boxplots: raw vs ci_rl vs ci_rl_dl
- [ ] example montages
- [ ] residual maps
- [ ] reconvolved-image error maps
- [ ] speed comparison
- [ ] quality vs speed scatter plot

### Acceptance criteria

- [ ] `ci_rl_dl` improves validation/test metrics over `ci_rl` on synthetic test data.
- [ ] Reconvolution error does not become much worse after DL refinement.
- [ ] Visual examples do not show obvious hallucinated structures.
- [ ] Inference overhead is measured and reported.

---

## Phase 9 — Add CLI integration to cideconvolve

Only after the function API works.

### Suggested user-facing options

```bash
--method ci_rl_dl
--dl-model path/to/best_val.pt
--dl-z-radius 2
--dl-batch-size 8
--dl-use-ratio-channel true
--dl-consistency-check true
--dl-save-diagnostics true
```

Alternative:

```bash
--method ci_rl --dl-refine path/to/best_val.pt
```

The second option may be cleaner because it makes DL refinement an optional post-processing step.

### Acceptance criteria

- [ ] Existing CLI behavior is unchanged for old methods.
- [ ] `--method ci_rl_dl` or `--dl-refine` works on a small test image.
- [ ] Method metadata records that DL refinement was used.
- [ ] Missing model path gives a clear error.

---

## Phase 10 — Real-data validation

Synthetic success is not enough.

### Suggested real tests

Use a small set of real microscopy images:

- [ ] bead stack with known bead size
- [ ] nuclei image
- [ ] spot/foci image
- [ ] filament or membrane image
- [ ] low-SNR image
- [ ] image with out-of-focus background

### Validation questions

- [ ] Does `ci_rl_dl` reduce residual blur?
- [ ] Does it preserve intensities?
- [ ] Does it introduce false puncta?
- [ ] Does it change spot counts too much?
- [ ] Does it improve segmentation stability?
- [ ] Does reconvolving the output still explain the raw image?

### Acceptance criteria

- [ ] No obvious hallucinations in low-SNR background.
- [ ] Spot counts and segmentation results remain plausible.
- [ ] Bead FWHM improves or stays comparable to `ci_rl`.
- [ ] Intensity conservation error is acceptable.

---

## Suggested implementation order for Codex

Use small Codex tasks. Do not ask Codex to implement everything at once.

### Task 1 — Create skeleton files

Prompt:

```text
Create cideconvolve/deconvolve_ci_dl.py, cideconvolve/train.py and cideconvolve/gui_train.py skeletons. Add documented function stubs for deconvolve_ci_rl_dl, apply_dl_refinement_25d, ResidualUNet25D, synthetic data generation, training, evaluation and GUI launcher. Do not change existing behavior.
```

Acceptance:

```text
Imports succeed. Existing tests still pass.
```

### Task 2 — Implement `ResidualUNet25D`

Prompt:

```text
Implement a small PyTorch 2D residual U-Net named ResidualUNet25D. It should accept N,C,Y,X and output N,1,Y,X. Add a minimal unit test with random tensors. Include checkpoint save/load helpers.
```

Acceptance:

```text
Random input produces correct output shape. Checkpoint reload gives identical output.
```

### Task 3 — Implement 2.5D inference wrapper

Prompt:

```text
Implement apply_dl_refinement_25d. It should create z-window input patches from raw and ci_rl deconvolved volumes, run the model slice-by-slice or in batches, add predicted residuals to ci_rl, clamp non-negative and return the refined volume. Include CPU and CUDA support.
```

Acceptance:

```text
Dummy zero model returns ci_rl unchanged. Output shape matches input.
```

### Task 4 — Wrap existing `ci_rl`

Prompt:

```text
Implement deconvolve_ci_rl_dl by calling the existing ci_rl path from deconvolve_ci.py, then optionally applying apply_dl_refinement_25d when a model_path is provided. Running without a model_path must behave exactly like ci_rl.
```

Acceptance:

```text
No-model ci_rl_dl equals ci_rl within numerical tolerance.
```

### Task 5 — Synthetic GT generator

Prompt:

```text
Implement synthetic 3D GT volume generation in train.py. Generate random mixtures of spots, blobs, ellipsoids, hollow vesicles, filaments, membranes and diffuse background. Save gt.tif and metadata.json. Add reproducible random seed support.
```

Acceptance:

```text
A small dataset of 5 volumes can be generated and visually inspected.
```

### Task 6 — Forward model and ci_rl preprocessing

Prompt:

```text
Extend the synthetic generator to create PSFs using existing cideconvolve PSF helpers, convolve GT with PSF, add Poisson/Gaussian/background noise, save raw.tif and psf.tif, then run ci_rl and save ci_rl.tif.
```

Acceptance:

```text
Each sample folder contains gt.tif, raw.tif, ci_rl.tif, psf.tif and metadata.json.
```

### Task 7 — Dataset class and patch sampler

Prompt:

```text
Create a PyTorch Dataset for random 2.5D patches from generated sample folders. Use robust percentile normalization. Return input tensor, residual target, central gt plane, central ci_rl plane, raw plane and metadata.
```

Acceptance:

```text
DataLoader returns stable tensor shapes and no train/val leakage.
```

### Task 8 — Training loop

Prompt:

```text
Implement train.py training loop with argparse. Support --generate-data, --train, --evaluate and combined --generate-data --train. Use AdamW, mixed precision, validation interval, checkpoint saving, resume, loss_curve.csv and loss_curve.png.
```

Acceptance:

```text
Training for 100 steps runs on a tiny generated dataset and produces checkpoints and a loss plot.
```

### Task 9 — Validation examples

Prompt:

```text
Add validation montage saving to train.py. Show gt, raw, ci_rl, ci_rl_dl, residual prediction and error maps for selected validation patches.
```

Acceptance:

```text
PNG validation montages are saved during training.
```

### Task 10 — Evaluation metrics

Prompt:

```text
Add evaluation mode to compare raw, ci_rl and ci_rl_dl on held-out test volumes. Compute MAE, RMSE, PSNR, SSIM if available, intensity conservation and reconvolution consistency. Save CSV and summary plots.
```

Acceptance:

```text
Evaluation produces metrics.csv and comparison figures.
```

### Task 11 — GUI launcher

Prompt:

```text
Implement gui_train.py as a PyQt6 launcher that runs train.py as a subprocess. Include fields for output folder, n volumes, volume size, patch size, steps, batch size, learning rate, checkpoint path and buttons for generate data, train, stop and open loss graph. Stream stdout/stderr into a text area.
```

Acceptance:

```text
GUI can generate a tiny dataset and start/stop a short training run.
```

### Task 12 — CLI integration

Prompt:

```text
Add cideconvolve CLI support for --method ci_rl_dl or alternatively --dl-refine path/to/model.pt. Preserve all existing methods and defaults. Save DL diagnostics in output metadata.
```

Acceptance:

```text
Old commands still work. A test command with ci_rl_dl produces output and diagnostics.
```

---

## Practical first experiment

### Dataset

```text
n_volumes = 1000
volume_size = 32 x 256 x 256
train/val/test = 800/100/100
z_radius = 2
patch_size = 128 or 256
```

### Training

```text
model = ResidualUNet25D
input = raw z-window + ci_rl z-window + ratio/residual channel
target = gt central plane - ci_rl central plane
steps = 100000 initially
batch_size = largest stable batch on RTX A5000 24 GB
mixed_precision = true
optimizer = AdamW
learning_rate = 1e-4
```

### Expected runtime on RTX A5000 24 GB

Rough estimate:

```text
small sanity dataset:     minutes
1000-volume generation:   several hours to overnight, depending on ci_rl preprocessing
2.5D training:            6 to 18 hours
full evaluation:          1 to 3 hours
```

### Success criteria

The first experiment is successful if:

- [ ] `ci_rl_dl` improves MAE/RMSE/SSIM over `ci_rl` on synthetic test volumes.
- [ ] The improvement is visible in validation montages.
- [ ] Reconvolution consistency is not substantially worse.
- [ ] No obvious false structures are generated in empty/background regions.
- [ ] Inference overhead is acceptable compared with running many more `ci_rl` iterations.

---

## Coding style notes

- Keep classical deconvolution and DL refinement separable.
- Make DL optional.
- Avoid changing existing `ci_rl`, `ci_rl_tv` or `ci_sparse_hessian` behavior.
- Add clear metadata to outputs when DL refinement is used.
- Use deterministic seeds where possible.
- Save configs and metrics so benchmark runs are reproducible.
- Keep the first model small and conservative.
- Add real-data validation before claiming biological reliability.

---

## Later extensions

After `ci_rl_dl` works:

- [ ] Add `ci_rl_tv_dl`.
- [ ] Add `ci_sparse_hessian_dl`.
- [ ] Train separate models for widefield and confocal.
- [ ] Add full 3D U-Net option.
- [ ] Add RL-informed/unrolled model inspired by RLN.
- [ ] Add OME-Zarr training data support.
- [ ] Add multi-channel training/inference.
- [ ] Add measured-PSF training.
- [ ] Add teacher-target training using slow high-quality deconvolution.
- [ ] Add benchmark report generation for speed vs quality.
