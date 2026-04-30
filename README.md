# CIDeconvolve

**GPU-accelerated 3-D / 2-D microscopy deconvolution with SHB Richardson-Lucy and sparse-Hessian regularisation.**

CIDeconvolve is a [BIAFLOWS](https://biaflows.neubias.org/)-compatible
workflow that deconvolves widefield and confocal fluorescence microscopy
images.  It reads OME-TIFF / OME-Zarr metadata where available, generates a
physically accurate PSF from the optical parameters, and applies one of three
native GPU-capable deconvolution methods: SHB-accelerated Richardson-Lucy,
SHB-RL with Total Variation regularisation, or a sparse-Hessian /
SPITFIRE-style variational solver — all via PyTorch.

| | |
|---|---|
| **Docker image** | `cellularimagingcf/w_cideconvolve` |
| **Version** | v1.5.0 |
| **Container type** | Singularity (pulled from Docker Hub) |
| **Methods** | `ci_rl` · `ci_rl_tv` · `ci_sparse_hessian` |
| **Benchmark** | built-in with timing metrics CSV and MIP montages |

---

## Recent updates in v1.5.0

- **Finite confocal pinholes:** confocal PSF generation now supports a
  user-facing pinhole diameter in Airy disk units via `--pinhole_airy`.
  Per-channel metadata pinhole sizes are converted to Airy units when NA,
  magnification, and emission wavelength are available.
- **Richer OME metadata reading:** OME MapAnnotation and SVI/Huygens XML
  annotations are now used for sample RI, immersion RI, wavelengths,
  acquisition mode, and custom pinhole Airy values when present.
- **Per-channel optics in the GUI:** emission wavelength, excitation
  wavelength, and confocal pinhole Airy units can be shown and edited as
  comma-separated per-channel values.
- **Improved GUI image viewing:** loaded-image logs show per-channel
  metadata, pinhole values, intensity statistics, and sample RI. The
  advanced scaling dialog uses faster histogram sampling, debounced slider
  updates, and works with slice, MIP, and SUM projections.
- **2D widefield auto mode:** single-plane widefield data can use the
  enhanced widefield-aware 2D model with expert controls for aggressiveness
  and background estimation.

---

## Methods

### `ci_rl` — Scaled Heavy Ball Accelerated Richardson-Lucy

Standard Richardson-Lucy enhanced with **Scaled Heavy Ball (SHB) momentum
acceleration** (Wang & Miller 2014).  Achieves 5–10× faster convergence
than vanilla RL at no extra per-iteration cost.  Includes Bertero boundary
correction weights and I-divergence convergence monitoring.

### `ci_rl_tv` — SHB-RL with Total Variation Regularisation

Same as `ci_rl` with an additional **Total Variation (TV) penalty** after
each RL update (Dey et al. 2006).  Suppresses noise amplification at high
iteration counts while preserving edges.  Controlled by the `--tv_lambda`
parameter (typical range 0.00005–0.001).

### `ci_sparse_hessian` — Sparse-Hessian Variational Deconvolution

A quality-focused **sparse-Hessian / SPITFIRE-style** variational method.
It combines the same FFT-based forward model and preprocessing stack used by
the RL-family methods with a sparse-Hessian prior that favours thin,
high-contrast structures while suppressing noise.  Controlled by
`--sparse_hessian_weight` and `--sparse_hessian_reg`.

### Stabilisation Options

The RL-family methods also support:

- **Noise-gated damping** via `--damping`
- **Positive offsetting** via `--offset`
- **Anscombe-domain Gaussian prefiltering** via `--prefilter_sigma`
- **Initial estimate selection** via `--start flat|observed|lowpass`
- **Enhanced 2D widefield restoration** via `--two_d_mode auto`, which uses a
  conservative widefield-aware 2D PSF model for `Z=1` data; `legacy_2d` keeps
  the old pure-2D RL path

For full algorithmic details see [DECONVOLVE_CI.MD](DECONVOLVE_CI.MD).

---

## Using CIDeconvolve with BIOMERO

[BIOMERO](https://github.com/NL-BioImaging/biomero) (BioImage Analysis in
OMERO) lets you run FAIR bioimage-analysis workflows from an OMERO server
on a SLURM-based HPC cluster.  CIDeconvolve is designed to plug directly
into this framework.

### How it works

1. The OMERO admin configures the workflow in
   **`slurm-config.ini`** on the SLURM submission host by adding a section
   for `W_CIDeconvolve`:

   ```ini
   [SLURM]
   # ... global SLURM settings ...

   [W_CIDeconvolve]
   # Override default SLURM resources for this workflow
   job_cpus=8
   job_memory=52G
   job_gres=gpu:2g.24gb
   ```

2. BIOMERO reads **`descriptor.json`** from the container to discover
   input parameters (method, iterations, device, PSF settings, benchmark
   options, etc.) and presents them in the OMERO web UI.

3. On submission, BIOMERO pulls the Singularity image from Docker Hub,
   transfers the selected images, and executes the workflow on the cluster.

4. Results (deconvolved images, benchmark montages, metrics CSV) are
   automatically uploaded back into OMERO.

> For full BIOMERO setup instructions see the
> [BIOMERO documentation](https://nl-bioimaging.github.io/biomero/)
> and the [NL-BIOMERO deployment repo](https://github.com/NL-BioImaging/NL-BIOMERO).

### SLURM job script

A ready-made SLURM script is provided for manual cluster submission
(outside of BIOMERO):

```bash
sbatch cideconvolve.slurm \
    --infolder /data/myimages \
    --outfolder /data/results \
    -- --method ci_rl --iterations 40 --benchmark True
```

See `cideconvolve.slurm` for full usage and resource settings.

---

## Building the Docker image locally

```bash
docker build -t w_cideconvolve:v1.4.2 -t w_cideconvolve:latest .
```

The Dockerfile builds on the **NVIDIA CUDA 12.6 runtime** image with
Python 3.11 and all pip dependencies — no Java, no conda, no compilation
step required.

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  (for GPU pass-through at runtime)
- A working `docker build` environment (Docker Desktop on Windows/macOS,
  or Docker Engine on Linux)

---

## Running locally with Docker

```bash
docker run --rm --gpus all \
    -v /path/to/input:/data/in \
    -v /path/to/output:/data/out \
    -v /tmp/gt:/data/gt \
    cellularimagingcf/w_cideconvolve \
    --infolder /data/in --outfolder /data/out --gtfolder /data/gt \
    --method ci_rl --iterations 40
```

Replace paths as needed.  The `--gpus all` flag enables NVIDIA GPU
pass-through.  Omit it to force CPU-only execution.

By default, image metadata is used for NA, wavelengths, pixel sizes,
microscope type, confocal pinhole, and refractive indices where present;
descriptor/CLI values are used as fallbacks when image metadata is missing. Add
`--overrule_image_metadata True` to force the descriptor/CLI values for those
fields.

### Benchmark mode

```bash
docker run --rm --gpus all \
    -v /path/to/input:/data/in \
    -v /path/to/output:/data/out \
    -v /tmp/gt:/data/gt \
    cellularimagingcf/w_cideconvolve \
    --infolder /data/in --outfolder /data/out --gtfolder /data/gt \
    --benchmark True --bench_crop True
```

Benchmark mode deconvolves the first input image with `ci_rl`,
`ci_rl_tv`, and `ci_sparse_hessian` at the requested iteration counts, writes a CSV with
timing metrics, and generates MIP montage images. Optional deconvolution-effect
image metrics can be enabled with `--compute_metrics True`.
See [metrics.md](metrics.md) for the metric formulas and interpretation.

---

## Running locally without Docker

### Requirements

- Python 3.10 or 3.11
- PyTorch 2.4+ with CUDA support
- For OMERO login/browser support and the shared XYZT/3D GUI viewer: `omero-browser-qt[viewer]==0.2.2`

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### CLI

```bash
python wrapper.py \
    --infolder ./infolder --outfolder ./outfolder --gtfolder ./gtfolder \
    --method ci_rl --iterations 40
```

### Launcher (GUI)

A PyQt6-based launcher provides a graphical interface with parameter
controls, folder pickers, and a live command preview:

```bash
python launcher.py
```

The launcher saves your last-used settings and can restore them on next
launch via the **Restore Last Settings** button.

The standalone deconvolution GUI also supports local OME-TIFF / OME-Zarr
opening, OMERO browsing, synchronized dual-pane XYZT / 3D viewing, per-channel
optics fields, SUM/MIP projections, and an advanced scaling dialog when the
GUI dependencies are installed.

---

## Parameters

All parameters are defined in `descriptor.json` and exposed on the
command line via `wrapper.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 40 | Number of RL iterations (comma-separated for per-channel) |
| `--tiling` | custom | Tiling mode: `none` or `custom` |
| `--tile_limits` | 512, 64 | Max tile dimensions `max_xy, max_z` (when tiling = `custom`) |
| `--method` | ci_rl | Deconvolution method: `ci_rl`, `ci_rl_tv`, or `ci_sparse_hessian` |
| `--tv_lambda` | 0.0001 | TV regularisation strength (only for `ci_rl_tv`) |
| `--sparse_hessian_weight` | 0.6 | Hessian-vs-sparsity balance (only for `ci_sparse_hessian`) |
| `--sparse_hessian_reg` | 0.98 | Data-vs-regulariser balance (only for `ci_sparse_hessian`) |
| `--device` | auto | Compute device: `auto`, `cpu`, `cuda` |
| `--overrule_image_metadata` | false | Replace image metadata with descriptor/CLI metadata values |
| `--na` | 1.4 | Numerical aperture fallback, or override when metadata is overruled |
| `--refractive_index` | oil (1.515) | Immersion medium RI fallback, or override when metadata is overruled |
| `--sample_ri` | prolong gold (1.47) | Sample/mounting medium RI fallback, or override when metadata is overruled |
| `--microscope_type` | confocal | `widefield` or `confocal` fallback, or override when metadata is overruled |
| `--two_d_mode` | auto | RL-family 2D widefield mode: `auto` (widefield-aware 2D PSF) or `legacy_2d` |
| `--emission_wl` | 520 | Emission wavelength fallback, or override when metadata is overruled |
| `--excitation_wl` | 488 | Excitation wavelength fallback, or override when metadata is overruled |
| `--pinhole_airy` | 1.00 | Confocal pinhole diameter in Airy disk units, comma-separated per channel; metadata pinhole sizes are converted when possible, or this value is used as fallback/override |
| `--background` | auto | Background subtraction: `auto`, numeric value, or `0` to disable |
| `--damping` | none | Noise-gated damping for `ci_rl` / `ci_rl_tv`: `none`, `auto`, or numeric |
| `--offset` | auto | Positive processing offset: `auto`, `none`, or numeric |
| `--prefilter_sigma` | 0.0 | Anscombe-domain Gaussian prefilter sigma in pixels |
| `--start` | flat | Initial estimate: `flat`, `observed`, or `lowpass` |
| `--convergence` | auto | Early-stopping convergence: `auto` or `none` |
| `--rel_threshold` | 0.005 | Relative change threshold for early stopping |
| `--two_d_wf_aggressiveness` | 0.6 | Expert tuning for enhanced 2D widefield auto mode |
| `--two_d_wf_bg_radius_um` | 2.0 | Background-estimator radius in micrometers for enhanced 2D widefield auto mode |
| `--two_d_wf_bg_scale` | 0.75 | Background-estimator scale factor for enhanced 2D widefield auto mode |
| `--pixel_size_xy` | 65 | Lateral pixel size fallback in nm, or override when metadata is overruled |
| `--pixel_size_z` | 200 | Axial pixel size fallback in nm, or override when metadata is overruled |
| `--projection` | none | Z-projection: `none`, `mip`, `sum` |
| `--benchmark` | false | Run benchmark mode |
| `--bench_crop` | false | Centre-crop image to tile-size limits before benchmarking |
| `--compute_metrics` | false | Compute optional deconvolution-effect image metrics |

---

## Metadata behavior

When `--overrule_image_metadata false`, image metadata wins and descriptor /
CLI values are used only as fallbacks. When it is true, descriptor / CLI
values replace image metadata.

OME-TIFF and OME-Zarr readers use standard OME fields for pixel size,
objective NA, magnification, immersion RI, wavelengths, acquisition mode, and
pinhole size. The OME-TIFF reader also understands benchmark-style
`MapAnnotation` keys such as `SampleRefractiveIndex` and `PinholeAiryUnits`,
plus SVI/Huygens XML annotations such as `RefrIndexMedium`,
`RefrIndexLensMedium`, `LambdaEm`, and `LambdaEx`.

For confocal data, physical metadata pinhole diameters are converted to Airy
disk units as:

```text
AU = pinhole_um / (1.22 * emission_um * magnification / NA)
```

Use `--pinhole_airy 0` for the legacy point-detector confocal model. Widefield
PSFs ignore the pinhole parameter.

---

## Project structure

```
wrapper.py              BIAFLOWS entrypoint — parameter parsing, benchmark runner, metrics
deconvolve.py           Core deconvolution engine + PSF generation
deconvolve_ci.py        CI SHB-RL / RLTV / sparse-Hessian implementation (PyTorch)
launcher.py             PyQt6 GUI launcher
gui_deconvolve_ci.py    GUI deconvolution panel
ci_dual_viewer.py       Dual-pane XYZT / 3D viewer widget used by the GUI
descriptor.json         BIAFLOWS/BIOMERO parameter descriptor
bioflows_local.py       Local BIAFLOWS compatibility shim
Dockerfile              Docker build (CUDA 12.6 runtime + Python 3.11)
requirements.txt        Python dependencies (local install)
requirements_gui.txt    Python dependencies for the GUI install
requirements_docker.txt Python dependencies (Docker)
version.txt             Project version marker
```

---

## References

- **SHB Acceleration:** Wang, Y. & Miller, E. L. (2014). "Scaled Heavy-Ball Acceleration of the Richardson-Lucy Algorithm for 3D Microscopy Image Restoration." *IEEE TIP* **23**(12), 5284–5297.
- **TV Regularisation:** Dey, N. et al. (2006). "Richardson-Lucy Algorithm With Total Variation Regularization for 3D Confocal Microscope Deconvolution." *Microsc. Res. Tech.* **69**(4), 260–266.
- **BIOMERO:** Luik, T. T., Rosas-Bertolini, R., Reits, E. A. J., Hoebe, R. A. & Krawczyk, P. M. (2024). "BIOMERO: A scalable and extensible image analysis framework." *Patterns* **5**(8), 101024. [doi:10.1016/j.patter.2024.101024](https://doi.org/10.1016/j.patter.2024.101024) · [GitHub](https://github.com/NL-BioImaging/biomero) · [Documentation](https://nl-bioimaging.github.io/biomero/)
- **BIAFLOWS:** Rubens, U. et al. (2020). "BIAFLOWS: A Collaborative Framework to Reproducibly Deploy and Benchmark Bioimage Analysis Workflows." *Patterns* **1**(3), 100040. [doi:10.1016/j.patter.2020.100040](https://doi.org/10.1016/j.patter.2020.100040)
- **PSF Generator:** Kirshner, H. et al. — [EPFL PSF Generator](https://bigwww.epfl.ch/algorithms/psfgenerator/)
- **Gibson–Lanni model:** Gibson, S. F. & Lanni, F. (1992). [doi:10.1364/JOSAA.9.000154](https://doi.org/10.1364/JOSAA.9.000154)
- **OMERO:** Allan, C. et al. (2012). "OMERO: flexible, model-driven data management for experimental biology." *Nat Methods* **9**, 245–253. [doi:10.1038/nmeth.1896](https://doi.org/10.1038/nmeth.1896)

---

## License

MIT — see [LICENSE](LICENSE).
