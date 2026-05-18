# cideconvolve.spec — PyInstaller build spec (single-file)
# Build with:  pyinstaller cideconvolve.spec
#              (run from the repository root)
#
# Produces:  dist/cideconvolve.exe  (single-file executable)
#            dist/models/           (visible default DL models; also embedded)
#
# NOTE: icon.ico is used for the Windows executable icon.

import os
import pkgutil
import shutil
from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

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

# ── Collect bioio reader plugins (as installed per requirements_gui.txt) ─────
bioio_datas,    bioio_binaries,    bioio_hiddenimports    = collect_all('bioio')
bioio_b_datas,  bioio_b_binaries,  bioio_b_hiddenimports  = collect_all('bioio_base')
bioio_ot_datas, bioio_ot_binaries, bioio_ot_hiddenimports = collect_all('bioio_ome_tiff')
bioio_oz_datas, bioio_oz_binaries, bioio_oz_hiddenimports = collect_all('bioio_ome_zarr')
bioio_cz_datas, bioio_cz_binaries, bioio_cz_hiddenimports = collect_all('bioio_czi')
bioio_nd_datas, bioio_nd_binaries, bioio_nd_hiddenimports = collect_all('bioio_nd2')

# ── Resolve exe icon (needs .ico on Windows) ─────────────────────────────────
_icon = os.path.abspath('gui/icon.ico')
if not os.path.exists(_icon):
    raise FileNotFoundError('gui/icon.ico is required for the Windows executable icon')

a = Analysis(
    ['gui/gui_deconvolve_ci.py'],
    pathex=['.', 'gui'],
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
        ('gui/icon.svg', '.'),      # runtime window icon (loaded by the app)
        ('gui/icon.ico', '.'),      # Windows executable icon
    ] + pyqt6_datas + vispy_datas + ogl_datas + torch_datas
      + zarr_datas + numcodecs_datas
      + ome_datas + omero_datas + obqt_datas
      + leica_datas + pil_datas
      + xsdata_datas + xspb_datas
      + pyd_datas + pyde_datas + dask_datas
      + imc_datas
      + imgio_datas + imgff_datas
      + mpl_datas
      + bioio_datas + bioio_b_datas
      + bioio_ot_datas + bioio_oz_datas
      + bioio_cz_datas + bioio_nd_datas
      + Tree('models', prefix='models'),  # default ci_rl_dl models
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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='cideconvolve',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    icon=_icon,             # icon.ico (see note at the top of this file)
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Also copy the model folder next to the one-file executable.  The models are
# embedded above, but this visible folder lets releases swap/update defaults
# without rebuilding the executable.
_dist_models = os.path.join(DISTPATH, 'models')
if os.path.isdir(_dist_models):
    shutil.rmtree(_dist_models)
shutil.copytree(os.path.abspath('models'), _dist_models)
print(f'Copied default DL models: {_dist_models}')
