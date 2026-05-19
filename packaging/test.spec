# test.spec - tiny PyInstaller splash-screen test build
# Build from the repository root with:
#     pyinstaller packaging/test.spec
#
# Produces:
#     dist/testspec.exe
#
# This intentionally avoids torch, OMERO, PyQt6, bioio, and all other heavy
# application dependencies. It only packages testspec.py plus the same
# PyInstaller Splash configuration used by cideconvolve.spec.

import os

block_cipher = None
_SPEC_DIR = os.path.abspath(SPECPATH)
_ROOT = os.path.abspath(os.path.join(_SPEC_DIR, os.pardir))

_splash_image = os.path.join(_ROOT, 'gui', 'CIDeconvolve-Splash.png')
if not os.path.exists(_splash_image):
    raise FileNotFoundError('gui/CIDeconvolve-Splash.png is required for the startup splash screen')

a = Analysis(
    [os.path.join(_ROOT, 'testspec.py')],
    pathex=[_ROOT],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt6', 'PyQt5',
        'torch',
        'omero', 'omero_browser_qt',
        'bioio', 'bioio_base', 'bioio_ome_tiff', 'bioio_ome_zarr', 'bioio_czi', 'bioio_nd2',
        'numpy', 'scipy', 'matplotlib',
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
    a.scripts,
    splash,
    splash.binaries,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='testspec',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
