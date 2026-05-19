# cideconvolve_quickstart.spec — PyInstaller one-FOLDER build spec
# Build with:  pyinstaller cideconvolve_quickstart.spec
#              (run from the repository root)
#
# Produces:  dist/cideconvolve/cideconvolve.exe  (folder distribution)
#            Run cideconvolve.exe from inside dist/cideconvolve/ — it needs
#            the _internal/ sibling folder next to it.
#            Default DL models are copied to dist/cideconvolve/models/.
#            For release: zip the dist/cideconvolve/ folder itself.
#
# Why use this instead of cideconvolve.spec?
#   • Folder builds skip the single-file extraction step → app starts instantly.
#   • Incremental rebuilds are much faster (only changed files are re-written).
#   • Ideal for quick iteration / testing during development.
#   • Ship the whole dist/cideconvolve/ folder (e.g. zip it) for distribution.
#
# NOTE: icon.ico is used for the Windows executable icon.

import os
import pkgutil
import shutil
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
_SPEC_DIR = os.path.abspath(SPECPATH)
_ROOT = os.path.abspath(os.path.join(_SPEC_DIR, os.pardir))


def _collect_model_datas():
    """Return PyInstaller (source, destination) tuples for bundled DL models."""
    model_root = os.path.join(_ROOT, 'models')
    if not os.path.isdir(model_root):
        return []
    datas = []
    for dirpath, _, filenames in os.walk(model_root):
        rel_dir = os.path.relpath(dirpath, model_root)
        target_dir = 'models' if rel_dir == '.' else os.path.join('models', rel_dir)
        for filename in filenames:
            datas.append((os.path.join(dirpath, filename), target_dir))
    return datas

# ── Collect full PyQt6 ecosystem ──────────────────────────────────────────────
pyqt6_datas, pyqt6_binaries, pyqt6_hiddenimports = collect_all('PyQt6')

# ── Collect vispy (optional 3-D viewer in ci_dual_viewer) ────────────────────
vispy_datas, vispy_binaries, vispy_hiddenimports = collect_all('vispy')

# ── Collect torch (GPU deconvolution in deconvolve_ci) ───────────────────────
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

# ── Collect zarr + numcodecs (zarr I/O support) ──────────────────────────────
zarr_datas, zarr_binaries, zarr_hiddenimports = collect_all('zarr')
numcodecs_datas, numcodecs_binaries, numcodecs_hiddenimports = collect_all('numcodecs')

# ── Collect OME metadata stack (bioio → ome-types → xsdata) ─────────────────
ome_datas,    ome_binaries,    ome_hiddenimports    = collect_all('ome_types')
xsdata_datas, xsdata_binaries, xsdata_hiddenimports = collect_all('xsdata')
xspb_datas,   xspb_binaries,   xspb_hiddenimports   = collect_all('xsdata_pydantic_basemodel')

# ── Collect pydantic (runtime validation used by ome-types / bioio) ──────────
pyd_datas,  pyd_binaries,  pyd_hiddenimports  = collect_all('pydantic')
pyde_datas, pyde_binaries, pyde_hiddenimports = collect_all('pydantic_extra_types')

# ── Collect dask (lazy array loading in bioio) ───────────────────────────────
dask_datas, dask_binaries, dask_hiddenimports = collect_all('dask')

# ── Collect OpenGL (vispy rendering back-end) ────────────────────────────────
ogl_datas, ogl_binaries, ogl_hiddenimports = collect_all('OpenGL')

# ── Collect full OMERO + Ice ecosystem ───────────────────────────────────────
omero_datas, omero_binaries, omero_hiddenimports = collect_all('omero')
# Top-level Ice stubs generated as separate modules (omero_model_*, Glacier2_*, etc.)
ice_toplevel = [
    name for _, name, _ in pkgutil.iter_modules()
    if name.startswith(('omero_', 'Glacier2', 'IcePatch2', 'IceBox', 'IceGrid', 'IceStorm'))
]

# ── Collect bioio reader plugins (as installed per requirements_gui.txt) ─────
bioio_datas,    bioio_binaries,    bioio_hiddenimports    = collect_all('bioio')
bioio_b_datas,  bioio_b_binaries,  bioio_b_hiddenimports  = collect_all('bioio_base')
bioio_ot_datas, bioio_ot_binaries, bioio_ot_hiddenimports = collect_all('bioio_ome_tiff')
bioio_oz_datas, bioio_oz_binaries, bioio_oz_hiddenimports = collect_all('bioio_ome_zarr')
bioio_cz_datas, bioio_cz_binaries, bioio_cz_hiddenimports = collect_all('bioio_czi')
bioio_nd_datas, bioio_nd_binaries, bioio_nd_hiddenimports = collect_all('bioio_nd2')

# ── Collect imagecodecs (TIFF/OME-TIFF codec extensions required by tifffile) ─
imc_datas, imc_binaries, imc_hiddenimports = collect_all('imagecodecs')

# ── Collect imageio + imageio-ffmpeg (movie / MP4 export) ────────────────────
imgio_datas,  imgio_binaries,  imgio_hiddenimports  = collect_all('imageio')
imgff_datas,  imgff_binaries,  imgff_hiddenimports  = collect_all('imageio_ffmpeg')

# ── Collect matplotlib (colormaps + PSF fit chart) ───────────────────────────
mpl_datas, mpl_binaries, mpl_hiddenimports = collect_all('matplotlib')

# ── Collect omero_browser_qt (icons + source used by tree_model.__file__) ─────
obqt_datas, obqt_binaries, obqt_hiddenimports = collect_all('omero_browser_qt')

# ── Collect leica_browser_qt (Open Leica… button) ────────────────────────────
leica_datas, leica_binaries, leica_hiddenimports = collect_all('leica_browser_qt')

# ── Collect Pillow (movie frames / PNG exports / overlays) ───────────────────
pil_datas, pil_binaries, pil_hiddenimports = collect_all('PIL')

# ── Resolve exe icon (needs .ico on Windows) ─────────────────────────────────
_icon = os.path.join(_ROOT, 'gui', 'icon.ico')
if not os.path.exists(_icon):
    raise FileNotFoundError('gui/icon.ico is required for the Windows executable icon')
_splash_image = os.path.join(_ROOT, 'gui', 'CIDeconvolve-Splash.png')
if not os.path.exists(_splash_image):
    raise FileNotFoundError('gui/CIDeconvolve-Splash.png is required for the startup splash screen')

a = Analysis(
    [os.path.join(_ROOT, 'gui', 'gui_deconvolve_ci.py')],
    pathex=[_ROOT, os.path.join(_ROOT, 'gui')],
    binaries=(
        pyqt6_binaries + vispy_binaries + ogl_binaries + torch_binaries
        + zarr_binaries + numcodecs_binaries
        + ome_binaries + omero_binaries + obqt_binaries
        + leica_binaries + pil_binaries
        + xsdata_binaries + xspb_binaries
        + pyd_binaries + pyde_binaries + dask_binaries
        + imc_binaries
        + imgio_binaries + imgff_binaries
        + mpl_binaries
        + bioio_binaries + bioio_b_binaries
        + bioio_ot_binaries + bioio_oz_binaries
        + bioio_cz_binaries + bioio_nd_binaries
    ),
    datas=[
        (os.path.join(_ROOT, 'gui', 'icon.svg'), '.'),      # runtime window icon (loaded by the app)
        (os.path.join(_ROOT, 'gui', 'icon.ico'), '.'),      # Windows executable icon
    ] + pyqt6_datas + vispy_datas + ogl_datas + torch_datas
      + zarr_datas + numcodecs_datas
      + ome_datas + omero_datas + obqt_datas
      + leica_datas + pil_datas
      + xsdata_datas + xspb_datas
      + pyd_datas + pyde_datas + dask_datas
      + imgio_datas + imgff_datas
      + mpl_datas
      + imc_datas
      + bioio_datas + bioio_b_datas
      + bioio_ot_datas + bioio_oz_datas
      + bioio_cz_datas + bioio_nd_datas
      + _collect_model_datas(),  # default ci_rl_dl models
    hiddenimports=[
        # ── local modules ───────────────────────────────────────────────────
        'ci_dual_viewer',
        'core.deconvolve_ci',
        'core.deconvolve_ci_dl',
        'core.deconvolve',
        'core.streaming',
        'core._meta_helpers',
        'wrapper',
        # ── numeric / array ─────────────────────────────────────────────────
        'numpy',
        'numpy.core',
        'numpy.lib',
        # ── image I/O ───────────────────────────────────────────────────────
        'nd2',
        'tifffile',
        'bioio',
        'bioio.writers',
        'bioio_base',
        'bioio_base.types',
        'bioio_ome_tiff',           # bioio reader plugin for OME-TIFF
        'bioio_ome_zarr',           # bioio reader plugin for OME-Zarr
        'bioio_nd2',                # bioio reader plugin for ND2
        'bioio_czi',                # bioio reader plugin for CZI
        # ── vispy back-end ──────────────────────────────────────────────────
        'vispy',
        'vispy.scene',
        'vispy.color',
        'vispy.visuals',
        'vispy.visuals.volume',
        'vispy.visuals.transforms',
        'vispy.app.backends._pyqt6',
        # ── OpenGL (vispy renderer) ─────────────────────────────────────────
        'OpenGL',
        'OpenGL.GL',
        'OpenGL.platform.win32',
        # ── OMERO + Ice (optional Open OMERO… button) ───────────────────────
        'omero',
        'omero.gateway',
        'omero.util',
        'omero.util.sessions',
        'omero.rtypes',
        'omero.model',
        'omero.api',
        'omero.sys',
        'omero.clients',
        'omero.cmd',
        'omero.cmd.graphs',
        'Ice',
        'IcePy',
        'Glacier2',
        'omero_browser_qt',
        'omero_browser_qt.browser_dialog',
        'omero_browser_qt.gateway',
        'omero_browser_qt.image_loader',
        'omero_browser_qt.login_dialog',
        'omero_browser_qt.rendering',
        'omero_browser_qt.scale_bar',
        'omero_browser_qt.selection_context',
        'omero_browser_qt.tree_model',
        'omero_browser_qt.widgets',
        'omero_browser_qt.view_backends',
        'omero_browser_qt.omero_viewer',
        # ── Leica browser (optional Open Leica… button) ─────────────────────
        'leica_browser_qt',
        # ── GPU / hardware monitoring ───────────────────────────────────────
        'psutil',
        'pynvml',
        # ── OME metadata / serialisation stack ─────────────────────────────
        'ome_types',
        'xsdata',
        'xsdata_pydantic_basemodel',
        'xsdata_pydantic_basemodel.hooks',
        'xsdata_pydantic_basemodel.hooks.class_type',
        'xsdata_pydantic_basemodel.hooks.cli',
        'pydantic',
        'pydantic_extra_types',
        # ── dask (lazy loading) ─────────────────────────────────────────────
        'dask',
        'dask.array',
        'dask.dataframe',
        # ── numcodecs (zarr compression codecs) ─────────────────────────────
        'numcodecs',
        'numcodecs.abc',
        'numcodecs.blosc',
        'numcodecs.zstd',
        'numcodecs.gzip',
        'numcodecs.lz4',
        'numcodecs.lzma',
        'numcodecs.zlib',
        'numcodecs.bz2',
        'numcodecs.compat',
        'numcodecs.registry',
    ] + pyqt6_hiddenimports + vispy_hiddenimports + ogl_hiddenimports
      + torch_hiddenimports
      + zarr_hiddenimports + numcodecs_hiddenimports
      + ome_hiddenimports + omero_hiddenimports + obqt_hiddenimports + leica_hiddenimports + ice_toplevel
      + xsdata_hiddenimports + xspb_hiddenimports
      + pyd_hiddenimports + pyde_hiddenimports + dask_hiddenimports
      + imc_hiddenimports
      + imgio_hiddenimports + imgff_hiddenimports
      + mpl_hiddenimports
      + pil_hiddenimports
      + bioio_hiddenimports + bioio_b_hiddenimports
      + bioio_ot_hiddenimports + bioio_oz_hiddenimports
      + bioio_cz_hiddenimports + bioio_nd_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
        'tkinter', '_tkinter',
        'scipy', 'sklearn', 'IPython',
        'pytest',
        'notebook', 'nbformat', 'jupyter',
        'zmq', 'jedi', 'parso',
        'mkdocs', 'mkdocstrings',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

splash = Splash(
    _splash_image,
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(205, 58),
    text_size=10,
    text_color='#f4f2a0',
    text_default='0% - Starting CI Deconvolve...',
    always_on_top=True,
    minify_script=True,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    splash,
    a.scripts,
    [],                         # no binaries/datas embedded — goes into COLLECT
    name='cideconvolve',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                  # skip UPX compression for faster builds
    console=False,
    disable_windowed_traceback=False,
    icon=_icon,                 # icon.ico (see note at the top of this file)
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    splash.binaries,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='cideconvolve',        # → dist/cideconvolve/
)

# ── Post-build: remove the bare bootloader exe that PyInstaller writes to
#    dist/ as a side-effect of the EXE step.  Only the folder in
#    dist/cideconvolve/ is a working distribution.
import os as _os
_stale = _os.path.join(DISTPATH, 'cideconvolve.exe')
if _os.path.isfile(_stale):
    _os.remove(_stale)
    print(f'Removed stale bootloader: {_stale}')

# ── Post-build: keep default ci_rl_dl models as a visible folder next to
#    dist/cideconvolve/cideconvolve.exe so they can be inspected or replaced
#    without rebuilding the executable.
_dist_models = _os.path.join(DISTPATH, 'cideconvolve', 'models')
_src_models = _os.path.join(_ROOT, 'models')
if _os.path.isdir(_src_models):
    if _os.path.isdir(_dist_models):
        shutil.rmtree(_dist_models)
    shutil.copytree(_src_models, _dist_models)
    print(f'Copied default DL models: {_dist_models}')
else:
    print(f'Skipped default DL models: {_src_models} not found')
