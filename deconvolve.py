"""
Microscopy image deconvolution module.

Reads OME-TIFF microscopy images, generates physically accurate PSFs from
image metadata, and performs deconvolution using CI SHB-accelerated
Richardson-Lucy (with optional Total Variation regularisation).

Backend:
    - CI RL / RLTV (PyTorch GPU/CPU): Scaled Heavy Ball accelerated
      Richardson-Lucy with optional TV regularisation, Bertero boundary
      weights, and I-divergence convergence monitoring.

Considerations:
    1. Incomplete metadata: If OME-TIFF files lack microscope type or NA,
       sensible defaults are used (widefield, NA=1.4). All metadata can be
       overridden via function parameters.
    2. Memory for large volumes: Approx memory ~8x image size.
"""

from __future__ import annotations

import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Fix DLL conflicts between conda-MKL (numpy) and pip-installed PyTorch on
# Windows.  Torch must be imported *before* numpy so that its own CUDA/OpenMP
# DLLs are loaded first.  Then we add conda's Library\bin so MKL can be found.
# KMP_DUPLICATE_LIB_OK silences the duplicate-OpenMP warning that remains.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # must come before numpy

if sys.platform == "win32":
    _conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    _lib_bin = os.path.join(_conda_prefix, "Library", "bin")
    if os.path.isdir(_lib_bin):
        os.add_dll_directory(_lib_bin)

import numpy as np
import tifffile

logger = logging.getLogger(__name__)

# OME XML namespace
_OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_DEFAULT_PINHOLE_AIRY_UNITS = 1.0


def _ome_enum_name(value: Any) -> str:
    """Return a compact lowercase name for OME enum-like values."""
    if value is None:
        return ""
    name = getattr(value, "name", None)
    text = str(name if name is not None else value).strip()
    return text.split(".")[-1].lower()


def _pinhole_size_to_um(size: Any, unit: Any) -> Optional[float]:
    """Convert metadata pinhole size to micrometers when possible."""
    if size is None:
        return None
    try:
        size_f = float(size)
    except (TypeError, ValueError):
        return None
    unit_name = _ome_enum_name(unit)
    if unit_name in ("", "µm", "um", "micrometer", "micrometre", "micrometers", "micrometres"):
        return size_f
    if unit_name in ("nm", "nanometer", "nanometre", "nanometers", "nanometres"):
        return size_f / 1000.0
    if unit_name in ("mm", "millimeter", "millimetre", "millimeters", "millimetres"):
        return size_f * 1000.0
    if unit_name in ("m", "meter", "metre", "meters", "metres"):
        return size_f * 1_000_000.0
    return None


def _calculate_pinhole_airy_units(
    pinhole_size: Any,
    pinhole_unit: Any,
    emission_wavelength_nm: Any,
    na: Any,
    magnification: Any,
) -> Optional[float]:
    """Convert detector-plane pinhole diameter metadata to Airy disk units."""
    pinhole_um = _pinhole_size_to_um(pinhole_size, pinhole_unit)
    try:
        emission_um = float(emission_wavelength_nm) / 1000.0
        na_f = float(na)
        mag_f = float(magnification)
    except (TypeError, ValueError):
        return None
    denom = 1.22 * emission_um * mag_f / max(na_f, 1e-12)
    if pinhole_um is None or denom <= 0.0:
        return None
    return float(pinhole_um / denom)


def _metadata_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        values = str(value).replace(";", ",").split(",")
    parsed: list[float] = []
    for item in values:
        number = _metadata_float(str(item).strip())
        if number is not None:
            parsed.append(number)
    return parsed


def _apply_map_metadata(meta: dict[str, Any], values: dict[str, Any]) -> set[str]:
    """Apply OME MapAnnotation metadata used by cideconvolve benchmark files."""
    if not values:
        return set()

    applied: set[str] = set()
    normalized = {str(key).strip().lower(): value for key, value in values.items()}

    sample_ri = _metadata_float(normalized.get("samplerefractiveindex"))
    if sample_ri is not None:
        meta["sample_refractive_index"] = sample_ri
        applied.add("sample_refractive_index")

    pinhole_values = _metadata_float_list(normalized.get("pinholeairyunits"))
    if pinhole_values:
        channels = meta.get("channels") or []
        if not channels:
            channels = [{}]
            meta["channels"] = channels
        for idx, ch in enumerate(channels):
            ch["pinhole_airy_units"] = (
                pinhole_values[idx] if idx < len(pinhole_values) else pinhole_values[-1]
            )
        applied.add("pinhole_airy_units")

    return applied


def _microscope_type_from_text(value: Any) -> Optional[str]:
    text = str(value or "").replace("_", " ").replace("-", " ").lower()
    if "confocal" in text or "multi photon" in text:
        return "confocal"
    if "wide" in text:
        return "widefield"
    return None


def _iter_annotation_dict_items(node: Any):
    if isinstance(node, dict):
        qname = node.get("qname")
        attrs = node.get("attributes") or {}
        if qname:
            yield str(qname), {str(k): str(v) for k, v in attrs.items()}
        for child in node.get("children") or []:
            yield from _iter_annotation_dict_items(child)
        for child in node.get("any_elements") or []:
            yield from _iter_annotation_dict_items(child)


def _apply_svi_xml_metadata(meta: dict[str, Any], items) -> set[str]:
    """Apply SVI/Huygens custom XML annotation values when present."""
    applied: set[str] = set()
    channels = meta.get("channels") or []
    channel_by_id = {
        str(ch.get("id", f"Channel:{idx}")).lower(): ch
        for idx, ch in enumerate(channels)
    }

    for qname, attrs in items:
        if qname.split("}")[-1] != "ChannelData":
            continue

        sample_ri = _metadata_float(attrs.get("RefrIndexMedium"))
        if sample_ri is not None and meta.get("sample_refractive_index") is None:
            meta["sample_refractive_index"] = sample_ri
            applied.add("sample_refractive_index")

        immersion_ri = _metadata_float(attrs.get("RefrIndexLensMedium"))
        if immersion_ri is not None and meta.get("refractive_index") is None:
            meta["refractive_index"] = immersion_ri
            applied.add("refractive_index")

        microscope_type = _microscope_type_from_text(attrs.get("MicroscopeSpec"))
        if microscope_type is not None and not meta.get("microscope_type"):
            meta["microscope_type"] = microscope_type
            applied.add("microscope_type")

        channel = channel_by_id.get(str(attrs.get("ChannelID", "")).lower())
        if channel is None:
            continue
        emission = _metadata_float(attrs.get("LambdaEm"))
        if emission is not None and channel.get("emission_wavelength") is None:
            channel["emission_wavelength"] = emission
            applied.add("emission_wavelength")
        excitation = _metadata_float(attrs.get("LambdaEx"))
        if excitation is not None and channel.get("excitation_wavelength") is None:
            channel["excitation_wavelength"] = excitation
            applied.add("excitation_wavelength")

    return applied


def _apply_pinhole_airy_units(
    meta: dict[str, Any],
    fallback_airy_units: Optional[float | Sequence[float]] = _DEFAULT_PINHOLE_AIRY_UNITS,
    *,
    overrule_metadata: bool = False,
) -> bool:
    """Populate per-channel pinhole Airy units; return True if metadata converted."""
    metadata_used = False
    if fallback_airy_units is None:
        fallbacks = [_DEFAULT_PINHOLE_AIRY_UNITS]
    elif isinstance(fallback_airy_units, Sequence) and not isinstance(fallback_airy_units, (str, bytes)):
        fallbacks = [float(value) for value in fallback_airy_units] or [_DEFAULT_PINHOLE_AIRY_UNITS]
    else:
        fallbacks = [float(fallback_airy_units)]
    for i, ch in enumerate(meta.get("channels") or []):
        fallback = fallbacks[i] if i < len(fallbacks) else fallbacks[-1]
        if ch.get("pinhole_airy_units") is not None and not overrule_metadata:
            metadata_used = True
        calculated = _calculate_pinhole_airy_units(
            ch.get("pinhole_size"),
            ch.get("pinhole_size_unit"),
            ch.get("emission_wavelength"),
            meta.get("na"),
            meta.get("magnification"),
        )
        if calculated is not None:
            ch["pinhole_airy_units_from_metadata"] = calculated
            metadata_used = True
        if overrule_metadata or ch.get("pinhole_airy_units") is None:
            ch["pinhole_airy_units"] = fallback if overrule_metadata or calculated is None else calculated
    return metadata_used

# ---------------------------------------------------------------------------
# Helper: detect GPU availability
# ---------------------------------------------------------------------------

def _get_device() -> str:
    """Return 'cuda:0' if CUDA is available, else 'cpu'."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# Phase 2: OME-TIFF Reader & Metadata Extraction
# ===========================================================================

def _parse_ome_xml(xml_path: Union[str, Path]) -> dict[str, Any]:
    """Parse an OME companion XML file and extract microscopy metadata.

    Falls back gracefully when fields are missing.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"ome": _OME_NS}
    meta: dict[str, Any] = {}

    # --- Objective ---
    obj = root.find(".//ome:Instrument/ome:Objective", ns)
    if obj is not None:
        na_str = obj.get("LensNA")
        meta["na"] = float(na_str) if na_str else None
        meta["magnification"] = (
            float(obj.get("NominalMagnification"))
            if obj.get("NominalMagnification")
            else None
        )
        meta["immersion"] = obj.get("Immersion")

    # --- ObjectiveSettings (refractive index) ---
    objset = root.find(".//ome:Image/ome:ObjectiveSettings", ns)
    if objset is not None:
        ri_str = objset.get("RefractiveIndex")
        meta["refractive_index"] = float(ri_str) if ri_str else None

    # --- Pixels ---
    pixels = root.find(".//ome:Image/ome:Pixels", ns)
    if pixels is not None:
        for dim in ("X", "Y", "Z"):
            key = f"PhysicalSize{dim}"
            val = pixels.get(key)
            meta[f"pixel_size_{dim.lower()}"] = float(val) if val else None
        meta["size_x"] = int(pixels.get("SizeX", 0))
        meta["size_y"] = int(pixels.get("SizeY", 0))
        meta["size_z"] = int(pixels.get("SizeZ", 0))
        meta["size_c"] = int(pixels.get("SizeC", 0))
        meta["size_t"] = int(pixels.get("SizeT", 0))

    # --- Channels ---
    channels = root.findall(".//ome:Image/ome:Pixels/ome:Channel", ns)
    ch_info: list[dict[str, Any]] = []
    microscope_type = None
    for ch in channels:
        info: dict[str, Any] = {}
        info["id"] = ch.get("ID")
        ex = ch.get("ExcitationWavelength")
        em = ch.get("EmissionWavelength")
        info["excitation_wavelength"] = float(ex) if ex else None
        info["emission_wavelength"] = float(em) if em else None
        info["pinhole_size"] = (
            float(ch.get("PinholeSize")) if ch.get("PinholeSize") else None
        )
        info["pinhole_size_unit"] = ch.get("PinholeSizeUnit")
        acq = ch.get("AcquisitionMode")
        info["acquisition_mode"] = acq
        if acq and "confocal" in acq.lower():
            microscope_type = "confocal"
        ch_info.append(info)

    meta["channels"] = ch_info
    meta["microscope_type"] = microscope_type or "widefield"

    map_values: dict[str, str] = {}
    for item in root.findall(".//ome:MapAnnotation/ome:Value/ome:M", ns):
        key = item.get("K")
        if key and item.text is not None:
            map_values[key] = item.text
    _apply_map_metadata(meta, map_values)

    _apply_svi_xml_metadata(
        meta,
        ((elem.tag, dict(elem.attrib)) for elem in root.iter()),
    )

    return meta


def _extract_bioio_metadata(img) -> dict[str, Any]:
    """Extract metadata from a bioio BioImage object."""
    meta: dict[str, Any] = {}
    try:
        pps = img.physical_pixel_sizes
    except Exception as exc:
        logger.warning("Could not read physical pixel sizes: %s", exc)
        pps = None
    meta["pixel_size_x"] = getattr(pps, "X", None) if pps is not None else None
    meta["pixel_size_y"] = getattr(pps, "Y", None) if pps is not None else None
    meta["pixel_size_z"] = getattr(pps, "Z", None) if pps is not None else None
    meta["size_x"] = img.dims.X
    meta["size_y"] = img.dims.Y
    meta["size_z"] = img.dims.Z
    meta["size_c"] = img.dims.C
    meta["size_t"] = img.dims.T
    meta["channel_names"] = img.channel_names

    # Try OME metadata for richer info
    try:
        ome = img.ome_metadata
        image = ome.images[0]
        pixels = image.pixels

        ch_info = []
        microscope_type = None
        for ch in pixels.channels:
            info: dict[str, Any] = {}
            info["id"] = getattr(ch, "id", None)
            info["excitation_wavelength"] = (
                float(ch.excitation_wavelength)
                if ch.excitation_wavelength is not None
                else None
            )
            info["emission_wavelength"] = (
                float(ch.emission_wavelength)
                if ch.emission_wavelength is not None
                else None
            )
            info["pinhole_size"] = (
                float(ch.pinhole_size) if ch.pinhole_size is not None else None
            )
            info["pinhole_size_unit"] = _ome_enum_name(getattr(ch, "pinhole_size_unit", None))
            acq = getattr(ch, "acquisition_mode", None)
            info["acquisition_mode"] = str(acq) if acq else None
            if acq and "confocal" in str(acq).lower():
                microscope_type = "confocal"
            ch_info.append(info)
        meta["channels"] = ch_info
        meta["microscope_type"] = microscope_type or "widefield"

        structured = getattr(ome, "structured_annotations", None)
        map_values: dict[str, str] = {}
        if structured is not None:
            for annotation in getattr(structured, "map_annotations", []) or []:
                value = getattr(annotation, "value", None)
                for item in getattr(value, "ms", []) or []:
                    key = getattr(item, "k", None)
                    val = getattr(item, "value", None)
                    if key and val is not None:
                        map_values[str(key)] = str(val)
        _apply_map_metadata(meta, map_values)

        xml_items = []
        if structured is not None:
            for annotation in getattr(structured, "xml_annotations", []) or []:
                value = getattr(annotation, "value", None)
                if hasattr(value, "model_dump"):
                    value = value.model_dump()
                elif hasattr(value, "dict"):
                    value = value.dict()
                xml_items.extend(_iter_annotation_dict_items(value))
        _apply_svi_xml_metadata(meta, xml_items)

        # Objective info
        if image.objective_settings is not None:
            ri = getattr(image.objective_settings, "refractive_index", None)
            meta["refractive_index"] = float(ri) if ri is not None else None

        # Try to get NA from instrument
        if ome.instruments:
            for instr in ome.instruments:
                if instr.objectives:
                    obj = instr.objectives[0]
                    if obj.lens_na is not None:
                        meta["na"] = float(obj.lens_na)
                    if obj.nominal_magnification is not None:
                        meta["magnification"] = float(obj.nominal_magnification)
                    if obj.immersion is not None:
                        meta["immersion"] = str(obj.immersion)
                    break
    except Exception as e:
        logger.warning("Could not extract full OME metadata: %s", e)

    return meta


def _is_hcs_zarr_plate(path: Path) -> bool:
    """Return True if *path* is an OME-Zarr HCS plate root."""
    try:
        import zarr
        store = zarr.open(str(path), mode="r")
        return "plate" in store.attrs
    except Exception:
        return False


def _first_hcs_zarr_field(path: Path) -> tuple[str, str, str]:
    """Return the first (row, column, field) entry from an HCS plate."""
    import zarr
    store = zarr.open(str(path), mode="r")
    plate = dict(store.attrs).get("plate", {})
    for well_entry in plate.get("wells", []):
        well_path = str(well_entry.get("path", ""))
        parts = well_path.split("/")
        if len(parts) < 2:
            continue
        well = store[well_path]
        well_attrs = dict(well.attrs)
        for image in well_attrs.get("well", {}).get("images", []):
            field = str(image.get("path", ""))
            if field:
                return parts[0], parts[1], field
        for key in sorted(well.keys()):
            if str(key).isdigit():
                return parts[0], parts[1], str(key)
    raise ValueError(f"No image fields found in HCS plate {path}")


def _load_first_hcs_zarr_field(path: Path) -> dict[str, Any]:
    """Load the first field from an OME-Zarr HCS plate root."""
    import zarr

    row, col, field = _first_hcs_zarr_field(path)
    store = zarr.open(str(path), mode="r")
    field_group = store[f"{row}/{col}/{field}"]
    field_attrs = dict(field_group.attrs)
    data = np.asarray(field_group["0"][:])

    while data.ndim > 5 and data.shape[0] == 1:
        data = data[0]
    if data.ndim == 5:
        n_t, n_c, n_z, n_y, n_x = data.shape
        data = data[0]
    elif data.ndim == 4:
        n_t = 1
        n_c, n_z, n_y, n_x = data.shape
    elif data.ndim == 3:
        n_t = 1
        n_c, n_y, n_x = data.shape
        n_z = 1
    else:
        raise ValueError(f"Unexpected HCS field data shape: {data.shape}")

    images = []
    for c in range(n_c):
        channel = data[c]
        if channel.ndim == 3 and channel.shape[0] == 1:
            channel = channel[0]
        images.append(np.asarray(channel, dtype=np.float32))

    meta: dict[str, Any] = {
        "size_x": n_x,
        "size_y": n_y,
        "size_z": n_z,
        "size_c": n_c,
        "size_t": n_t,
        "n_channels": n_c,
        "hcs_plate_field": f"{row}/{col}/{field}",
    }

    multiscales = field_attrs.get("multiscales", [])
    if multiscales:
        datasets = multiscales[0].get("datasets", [])
        if datasets:
            for transform in datasets[0].get("coordinateTransformations", []):
                if transform.get("type") != "scale":
                    continue
                scale = transform.get("scale", [])
                if len(scale) == 5:
                    meta["pixel_size_z"] = scale[2]
                    meta["pixel_size_y"] = scale[3]
                    meta["pixel_size_x"] = scale[4]
                elif len(scale) == 4:
                    meta["pixel_size_z"] = scale[1]
                    meta["pixel_size_y"] = scale[2]
                    meta["pixel_size_x"] = scale[3]
                elif len(scale) == 3:
                    meta["pixel_size_y"] = scale[1]
                    meta["pixel_size_x"] = scale[2]
                break

    omero_channels = field_attrs.get("omero", {}).get("channels", [])
    if omero_channels:
        meta["channel_names"] = [
            str(ch.get("label") or f"Ch{i}") for i, ch in enumerate(omero_channels)
        ]
        meta["channels"] = [{} for _ in omero_channels]

    logger.info("HCS plate detected; loaded first field %s/%s/%s", row, col, field)
    return {"images": images, "metadata": meta}


def load_image(
    path: Union[str, Path],
    *,
    # User overrides for missing/incorrect metadata
    na: Optional[float] = None,
    refractive_index: Optional[float] = None,
    microscope_type: Optional[str] = None,
    pixel_size_xy: Optional[float] = None,
    pixel_size_z: Optional[float] = None,
    emission_wavelengths: Optional[list[float]] = None,
    excitation_wavelengths: Optional[list[float]] = None,
    pinhole_airy_units: Optional[float | list[float]] = _DEFAULT_PINHOLE_AIRY_UNITS,
    sample_refractive_index: Optional[float] = 1.47,
    overrule_metadata: bool = True,
) -> dict[str, Any]:
    """Load an OME-TIFF image and extract microscopy metadata.

    Parameters
    ----------
    path : str or Path
        Path to an OME-TIFF file or a companion .ome file. If a companion
        file is given, the associated TIFF files are read automatically.
    na : float, optional
        Override numerical aperture (default: from metadata or 1.4).
    refractive_index : float, optional
        Override immersion medium refractive index (default: from metadata
        or 1.515 for oil).
    microscope_type : str, optional
        Override microscope type: "confocal" or "widefield".
    pixel_size_xy : float, optional
        Override lateral pixel size in micrometers.
    pixel_size_z : float, optional
        Override axial pixel size in micrometers.
    emission_wavelengths : list[float], optional
        Override emission wavelengths in nm, one per channel.
    excitation_wavelengths : list[float], optional
        Override excitation wavelengths in nm, one per channel.
    pinhole_airy_units : float or list[float], optional
        Confocal pinhole diameter(s) in Airy disk units. A list applies values
        per channel. Used as a fallback when image metadata is missing, or as
        an override when metadata is overruled.
    sample_refractive_index : float, optional
        Fallback refractive index of the sample medium (default 1.47).

    Returns
    -------
    dict with keys:
        'images': list of numpy arrays, one per channel, shape (Z,Y,X) or (Y,X)
        'metadata': dict with all microscopy parameters
    """
    path = Path(path)
    meta: dict[str, Any] = {}
    images: list[np.ndarray] = []

    # Check if we have a companion OME file
    companion_path = None
    if path.suffix == ".ome" and not path.name.endswith(".ome.tiff"):
        companion_path = path
    else:
        # Look for a companion file alongside the TIFF
        candidate = path.parent / path.name.replace(".ome.tiff", ".ome").replace(
            ".ome.tif", ".ome"
        )
        if candidate.exists() and candidate != path:
            companion_path = candidate
        # Also check common companion patterns (e.g., basename_ch.companion.ome)
        for f in path.parent.glob("*.companion.ome"):
            companion_path = f
            break

    # Parse companion OME XML if available
    if companion_path is not None:
        meta = _parse_ome_xml(companion_path)

    # OME-Zarr stores are directories — handle separately
    _is_zarr = path.is_dir() and path.suffix.lower() == ".zarr"

    # Read image data via bioio or tifffile
    try:
        from bioio import BioImage

        if _is_zarr:
            if _is_hcs_zarr_plate(path):
                data = _load_first_hcs_zarr_field(path)
                meta.update(data["metadata"])
                images.extend(data["images"])
            else:
                import bioio_ome_zarr
                img = BioImage(path, reader=bioio_ome_zarr.Reader)
                # If there are multiple images (scenes) take only the first
                if len(img.scenes) > 1:
                    logger.info(
                        "OME-Zarr contains %d images (scenes); using first: %s",
                        len(img.scenes), img.scenes[0],
                    )
                    img.set_scene(img.scenes[0])
                bioio_meta = _extract_bioio_metadata(img)
                for k, v in bioio_meta.items():
                    if k not in meta or meta[k] is None:
                        meta[k] = v
                for c in range(img.dims.C):
                    if img.dims.Z > 1:
                        channel_data = img.get_image_data("ZYX", T=0, C=c)
                    else:
                        channel_data = img.get_image_data("YX", T=0, C=c)
                    images.append(np.asarray(channel_data, dtype=np.float32))

        elif companion_path is not None and path.suffix == ".ome":
            # Find the associated TIFF files from the companion
            tiff_files = sorted(path.parent.glob("*.ome.tiff")) + sorted(
                path.parent.glob("*.ome.tif")
            )
            if tiff_files:
                # Read each channel file
                for tiff_path in tiff_files:
                    try:
                        img = BioImage(tiff_path)
                    except Exception:
                        # Binary OME-TIFFs (multi-file) are not supported
                        # by bioio-ome-tiff; fall back to tifffile.
                        logger.info("BioImage cannot read %s, using tifffile", tiff_path.name)
                        data = tifffile.imread(str(tiff_path))
                        images.append(np.asarray(data, dtype=np.float32))
                        continue
                    bioio_meta = _extract_bioio_metadata(img)
                    # Merge bioio metadata (companion XML takes priority)
                    for k, v in bioio_meta.items():
                        if k not in meta or meta[k] is None:
                            meta[k] = v
                    # Extract all channels from this file
                    for c in range(img.dims.C):
                        if img.dims.Z > 1:
                            channel_data = img.get_image_data("ZYX", T=0, C=c)
                        else:
                            channel_data = img.get_image_data("YX", T=0, C=c)
                        images.append(np.asarray(channel_data, dtype=np.float32))
            else:
                raise FileNotFoundError(
                    f"No .ome.tiff files found alongside companion {companion_path}"
                )
        else:
            img = BioImage(path)
            bioio_meta = _extract_bioio_metadata(img)
            for k, v in bioio_meta.items():
                if k not in meta or meta[k] is None:
                    meta[k] = v
            for c in range(img.dims.C):
                if img.dims.Z > 1:
                    channel_data = img.get_image_data("ZYX", T=0, C=c)
                else:
                    channel_data = img.get_image_data("YX", T=0, C=c)
                images.append(np.asarray(channel_data, dtype=np.float32))

    except Exception as exc:
        # Catch ImportError (bioio not installed) and UnsupportedFileFormatError
        # (no plugin for the specific format, e.g. TIFF without bioio-tifffile).
        # Re-raise anything that isn't one of these two known fallback cases.
        _is_unsupported = type(exc).__name__ == "UnsupportedFileFormatError"
        if not isinstance(exc, ImportError) and not _is_unsupported:
            raise
        if _is_zarr and isinstance(exc, ImportError):
            raise ImportError(
                "bioio-ome-zarr is required to read OME-Zarr files. "
                "Install with: pip install bioio-ome-zarr"
            )
        logger.warning("bioio cannot read this format, falling back to tifffile")
        if companion_path is not None and path.suffix == ".ome":
            tiff_files = sorted(path.parent.glob("*.ome.tiff")) + sorted(
                path.parent.glob("*.ome.tif")
            )
            for tiff_path in tiff_files:
                data = tifffile.imread(str(tiff_path))
                images.append(np.asarray(data, dtype=np.float32))
        else:
            data = tifffile.imread(str(path))
            if data.ndim == 4:  # Assume CZYX
                for c in range(data.shape[0]):
                    images.append(np.asarray(data[c], dtype=np.float32))
            elif data.ndim == 3:  # Single channel ZYX
                images.append(np.asarray(data, dtype=np.float32))
            elif data.ndim == 2:  # Single 2D image
                images.append(np.asarray(data, dtype=np.float32))

    def _use_value(current, value) -> bool:
        return value is not None and (overrule_metadata or current is None)

    # Apply user metadata values. With overrule_metadata=False, these are
    # fallbacks only; existing image metadata is preserved.
    if _use_value(meta.get("na"), na):
        meta["na"] = na
    if _use_value(meta.get("refractive_index"), refractive_index):
        meta["refractive_index"] = refractive_index
    if _use_value(meta.get("microscope_type"), microscope_type):
        meta["microscope_type"] = microscope_type
    if _use_value(meta.get("pixel_size_x"), pixel_size_xy):
        meta["pixel_size_x"] = pixel_size_xy
    if _use_value(meta.get("pixel_size_y"), pixel_size_xy):
        meta["pixel_size_y"] = pixel_size_xy
    if _use_value(meta.get("pixel_size_z"), pixel_size_z):
        meta["pixel_size_z"] = pixel_size_z
    if emission_wavelengths is not None:
        if "channels" not in meta:
            meta["channels"] = [{} for _ in emission_wavelengths]
        for i, wl in enumerate(emission_wavelengths):
            if i < len(meta["channels"]):
                ch = meta["channels"][i]
                if _use_value(ch.get("emission_wavelength"), wl):
                    ch["emission_wavelength"] = wl
    if excitation_wavelengths is not None:
        if "channels" not in meta:
            meta["channels"] = [{} for _ in excitation_wavelengths]
        for i, wl in enumerate(excitation_wavelengths):
            if i < len(meta["channels"]):
                ch = meta["channels"][i]
                if _use_value(ch.get("excitation_wavelength"), wl):
                    ch["excitation_wavelength"] = wl

    # Apply defaults for critical missing values (setdefault won't
    # replace an existing key whose value is None, so fix those too).
    # Track which keys received a fallback default so callers can
    # distinguish "from image metadata" vs "assumed default".
    _defaults = {
        "na": 1.4,
        "refractive_index": 1.515,
        "microscope_type": "widefield",
        "pixel_size_x": 0.065,
        "pixel_size_y": 0.065,
        "pixel_size_z": 0.2,
    }
    _defaulted = set()
    for _k, _v in _defaults.items():
        if meta.get(_k) is None:
            meta[_k] = _v
            _defaulted.add(_k)
    meta["_defaulted_keys"] = _defaulted
    if overrule_metadata and sample_refractive_index is not None:
        meta["sample_refractive_index"] = sample_refractive_index
    elif meta.get("sample_refractive_index") is None:
        meta["sample_refractive_index"] = (
            sample_refractive_index if sample_refractive_index is not None else 1.47
        )
    meta["n_channels"] = len(images)

    # Ensure channels list
    _em_defaulted = False
    if "channels" not in meta or not meta["channels"]:
        meta["channels"] = [
            {"emission_wavelength": 520.0} for _ in range(len(images))
        ]
        _em_defaulted = True
    # Fill in missing emission wavelengths with a default
    for ch in meta["channels"]:
        if ch.get("emission_wavelength") is None:
            ch["emission_wavelength"] = 520.0
            _em_defaulted = True
    if _em_defaulted:
        _defaulted.add("emission_wavelength")

    if _apply_pinhole_airy_units(
        meta,
        pinhole_airy_units,
        overrule_metadata=overrule_metadata,
    ):
        _defaulted.discard("pinhole_airy_units")
    else:
        _defaulted.add("pinhole_airy_units")

    logger.info(
        "Loaded %d channel(s), shape=%s, microscope=%s, NA=%.2f",
        len(images),
        images[0].shape if images else "N/A",
        meta["microscope_type"],
        meta["na"],
    )

    return {"images": images, "metadata": meta}


# ===========================================================================
# Phase 3: PSF Generation from Metadata
# ===========================================================================


def _estimate_two_d_wf_psf_z_nm(
    wavelength_nm: float,
    na: float,
    sample_ri: float,
    pixel_size_xy_nm: float,
) -> float:
    """Return a practical hidden-volume Z step for 2D widefield PSFs."""
    na_safe = max(float(na), 1e-6)
    sample_safe = max(float(sample_ri), na_safe + 1e-6)
    denom = max(sample_safe - np.sqrt(max(sample_safe ** 2 - na_safe ** 2, 1e-9)), 1e-6)
    nyquist_nm = wavelength_nm / (2.0 * denom)
    return float(max(min(nyquist_nm, 1000.0), pixel_size_xy_nm / 2.0))

def generate_psf(
    metadata: dict[str, Any],
    channel_idx: int = 0,
    *,
    psf_size_xy: Optional[int] = None,
    n_pix_pupil: int = 129,
    ri_coverslip: Optional[float] = None,
    ri_coverslip_design: Optional[float] = None,
    ri_immersion_design: Optional[float] = None,
    t_g: float = 170e3,
    t_g0: float = 170e3,
    t_i0: float = 100e3,
    z_p: float = 0.0,
    two_d_mode: str = "auto",
    pinhole_airy_units: Optional[float] = None,
) -> np.ndarray:
    """Generate a physically accurate PSF from microscopy metadata.

    Uses the CI Richards-Wolf / Kirchhoff PSF model with correct
    ``sqrt(cos θ)`` apodization and azimuthal averaging.  Vectorial
    for NA ≥ 0.9, scalar otherwise.  Gibson-Lanni correction is
    applied automatically when a refractive-index mismatch is detected.
    Confocal PSFs multiply emission × excitation (or square the
    emission PSF when no excitation wavelength is available).

    Parameters
    ----------
    metadata : dict
        Microscopy metadata as returned by load_image()['metadata'].
    channel_idx : int
        Which channel's wavelength to use for PSF computation.
    psf_size_xy : int, optional
        Lateral size of PSF in pixels. If None, auto-calculated as ~4x the
        Airy disk radius to ensure adequate extent.
    n_pix_pupil : int
        Pupil plane discretization (higher = more accurate but slower).

    Returns
    -------
    numpy.ndarray
        Normalized PSF, shape (Z,Y,X) for 3D or (Y,X) for 2D. Sum = 1.
    """
    from deconvolve_ci import ci_generate_psf

    na = metadata["na"]
    ri = metadata["refractive_index"]
    sample_ri = metadata.get("sample_refractive_index", 1.33)
    pix_xy_um = metadata["pixel_size_x"]
    pix_z_um = metadata.get("pixel_size_z", 0.3)
    n_z = metadata.get("size_z", 1)
    microscope_type = metadata.get("microscope_type", "widefield")

    # Channel wavelength
    ch = metadata["channels"][channel_idx]
    wavelength_nm = ch.get("emission_wavelength", 520.0)
    excitation_nm = ch.get("excitation_wavelength")
    pinhole_airy = ch.get("pinhole_airy_units", _DEFAULT_PINHOLE_AIRY_UNITS)
    if pinhole_airy_units is not None:
        pinhole_airy = pinhole_airy_units

    # Convert units to nm
    pix_xy_nm = pix_xy_um * 1000.0
    pix_z_nm = pix_z_um * 1000.0

    # Auto-calculate PSF lateral size if not specified
    # Airy disk radius ~ 0.61 * lambda / NA (in nm), then convert to pixels
    if psf_size_xy is None:
        airy_radius_nm = 0.61 * wavelength_nm / na
        airy_radius_px = airy_radius_nm / pix_xy_nm
        psf_size_xy = int(max(64, 2 * int(4 * airy_radius_px) + 1))
        # Ensure odd for centering
        if psf_size_xy % 2 == 0:
            psf_size_xy += 1

    use_2d_wf_auto = (
        n_z <= 1
        and microscope_type == "widefield"
        and str(two_d_mode).strip().lower() == "auto"
    )

    # 3D vs 2D
    is_3d = n_z > 1
    if use_2d_wf_auto:
        n_defocus = 65
        pix_z_nm = _estimate_two_d_wf_psf_z_nm(
            wavelength_nm, na, sample_ri, pix_xy_nm,
        )
    else:
        n_defocus = max(2 * n_z - 1, 1) if is_3d else 1

    psf = ci_generate_psf(
        na=na,
        wavelength_nm=wavelength_nm,
        pixel_size_xy_nm=pix_xy_nm,
        pixel_size_z_nm=pix_z_nm,
        n_xy=psf_size_xy,
        n_z=n_defocus,
        ri_immersion=ri,
        ri_sample=sample_ri,
        ri_coverslip=ri_coverslip if ri_coverslip is not None else ri,
        ri_coverslip_design=ri_coverslip_design if ri_coverslip_design is not None else ri,
        ri_immersion_design=ri_immersion_design if ri_immersion_design is not None else ri,
        t_g=t_g,
        t_g0=t_g0,
        t_i0=t_i0,
        z_p=z_p,
        microscope_type=microscope_type,
        excitation_nm=excitation_nm,
        pinhole_airy_units=pinhole_airy,
        n_pupil=n_pix_pupil,
    )

    # ci_generate_psf always returns 3D (n_z, n_xy, n_xy) — squeeze for 2D
    if not is_3d and not use_2d_wf_auto:
        psf = psf.squeeze(axis=0)

    logger.info("PSF generated: shape=%s, sum=%.6f", psf.shape, psf.sum())
    return psf


# ===========================================================================
# Phase 4: Deconvolution Engine
# ===========================================================================

# Available methods and their approximate memory multiplier relative to image
# size (Consideration 2: memory for large volumes)
METHODS = {
    "ci_rl": {"memory_factor": 8, "description": "CI SHB-accelerated RL (PyTorch GPU)"},
    "ci_rl_tv": {"memory_factor": 8, "description": "CI SHB-accelerated RL + TV (PyTorch GPU)"},
    "ci_sparse_hessian": {
        "memory_factor": 10,
        "description": "CI sparse-Hessian variational deconvolution (PyTorch GPU)",
    },
}


def deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    method: str = "ci_rl",
    *,
    # Richardson-Lucy parameters
    niter: int = 30,
    background: Union[int, str] = "auto",
    damping: Union[str, float] = 0.0,
    offset: Union[str, float] = "auto",
    prefilter_sigma: float = 0.0,
    start: str = "auto",
    convergence: str = "auto",
    rel_threshold: float = 0.005,
    check_every: int = 5,
    # Device override
    device: Optional[str] = None,
    # RLTV regularization
    tv_lambda: float = 1e-4,
    # Sparse-Hessian regularization
    sparse_hessian_weight: float = 0.6,
    sparse_hessian_reg: float = 0.98,
    # Physical voxel size
    pixel_size_xy: Optional[float] = None,
    pixel_size_z: Optional[float] = None,
    microscope_type: str = "widefield",
    two_d_mode: str = "auto",
    two_d_wf_aggressiveness: str = "balanced",
    two_d_wf_bg_radius_um: float = 0.5,
    two_d_wf_bg_scale: float = 1.0,
) -> np.ndarray:
    """Deconvolve an image using the specified method and PSF.

    Parameters
    ----------
    image : numpy.ndarray
        Input image, shape (Z,Y,X) for 3D or (Y,X) for 2D.
    psf : numpy.ndarray
        Point spread function, same dimensionality as image.
    method : str
        Deconvolution algorithm:
        - 'ci_rl' (default): CI SHB-accelerated Richardson-Lucy (PyTorch).
        - 'ci_rl_tv': CI SHB-accelerated RL + Total Variation.
        - 'ci_sparse_hessian': CI sparse-Hessian variational deconvolution.
    niter : int
        Number of iterations / optimisation steps (default: 30).
    background : int or str
        Background subtraction (default: 'auto').
    damping : str or float
        Noise-gated damping for RL-family methods (default: 0 / disabled).
    start : str
        Initial estimate for iterative solvers: ``"auto"``, ``"flat"``,
        ``"percentile_flat"``, ``"observed"``, ``"observed_bgsub"``,
        ``"lowpass"``, ``"lowpass_bgsub"``, or ``"hybrid"``.
    microscope_type : str
        Microscope mode. ``"widefield"`` enables the enhanced 2-D path when
        *image* is single-plane and *two_d_mode* is ``"auto"``.
    two_d_mode : str
        ``"auto"`` enables the enhanced 2-D widefield model; ``"legacy_2d"``
        keeps the historical pure-2-D behavior.
    two_d_wf_aggressiveness : str
        Expert tuning for 2-D widefield auto mode: ``"conservative"``,
        ``"balanced"``, or ``"strong"``.
    two_d_wf_bg_radius_um : float
        Background-estimator neighborhood radius in µm for 2-D WF auto mode.
    two_d_wf_bg_scale : float
        Multiplier applied to the auto-estimated 2-D widefield background.
    device : str, optional
        Force device ('cpu' or 'cuda'). Auto-detected if None.
    tv_lambda : float
        TV regularization strength for ci_rl_tv (default: 1e-4).
    sparse_hessian_weight : float
        Hessian-vs-sparsity balance for ci_sparse_hessian (default: 0.6).
    sparse_hessian_reg : float
        Data-vs-regulariser balance for ci_sparse_hessian (default: 0.98).

    Returns
    -------
    numpy.ndarray
        Deconvolved image (float32, non-negative), same shape as input.
    """
    if method not in METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(METHODS.keys())}"
        )

    # Crop PSF to image size when it is larger (e.g. n_defocus = 2*nz-1).
    if psf.ndim == image.ndim:
        slices = []
        for ax in range(psf.ndim):
            if psf.shape[ax] > image.shape[ax]:
                excess = psf.shape[ax] - image.shape[ax]
                lo = excess // 2
                slices.append(slice(lo, lo + image.shape[ax]))
            else:
                slices.append(slice(None))
        if any(s != slice(None) for s in slices):
            psf = psf[tuple(slices)].copy()
            logger.info("PSF cropped to image size: %s", psf.shape)

    return _deconvolve_ci_method(
        image, psf, niter=niter,
        method=method,
        tv_lambda=tv_lambda if method == "ci_rl_tv" else 0.0,
        damping=damping if method in ("ci_rl", "ci_rl_tv") else 0.0,
        sparse_hessian_weight=sparse_hessian_weight,
        sparse_hessian_reg=sparse_hessian_reg,
        background=background, offset=offset,
        prefilter_sigma=prefilter_sigma, start=start,
        convergence=convergence, rel_threshold=rel_threshold,
        pixel_size_xy=pixel_size_xy, pixel_size_z=pixel_size_z,
        microscope_type=microscope_type, two_d_mode=two_d_mode,
        two_d_wf_aggressiveness=two_d_wf_aggressiveness,
        two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
        two_d_wf_bg_scale=two_d_wf_bg_scale,
        check_every=check_every, device=device,
    )


# ---------------------------------------------------------------------------
# CI backend (deconvolve_ci module)
# ---------------------------------------------------------------------------

def _deconvolve_ci_method(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    method: str = "ci_rl",
    niter: int = 50,
    tv_lambda: float = 0.0,
    damping: Union[str, float] = 0.0,
    sparse_hessian_weight: float = 0.6,
    sparse_hessian_reg: float = 0.98,
    background: Union[int, str] = "auto",
    offset: Union[str, float] = "auto",
    prefilter_sigma: float = 0.0,
    start: str = "auto",
    convergence: str = "auto",
    rel_threshold: float = 0.005,
    check_every: int = 5,
    pixel_size_xy: Optional[float] = None,
    pixel_size_z: Optional[float] = None,
    microscope_type: str = "widefield",
    two_d_mode: str = "auto",
    two_d_wf_aggressiveness: str = "balanced",
    two_d_wf_bg_radius_um: float = 0.5,
    two_d_wf_bg_scale: float = 1.0,
    device: Optional[str] = None,
) -> np.ndarray:
    from deconvolve_ci import (
        ci_rl_deconvolve,
        ci_sparse_hessian_deconvolve,
    )

    common = dict(
        image=image,
        psf=psf,
        niter=niter,
        background=background,
        offset=offset,
        prefilter_sigma=prefilter_sigma,
        start=start,
        convergence=convergence,
        rel_threshold=rel_threshold,
        check_every=check_every,
        pixel_size_xy=pixel_size_xy,
        pixel_size_z=pixel_size_z,
        device=device,
    )

    if method == "ci_sparse_hessian":
        result = ci_sparse_hessian_deconvolve(
            sparse_hessian_weight=sparse_hessian_weight,
            sparse_hessian_reg=sparse_hessian_reg,
            **common,
        )
    else:
        result = ci_rl_deconvolve(
            tv_lambda=tv_lambda,
            damping=damping,
            microscope_type=microscope_type,
            two_d_mode=two_d_mode,
            two_d_wf_aggressiveness=two_d_wf_aggressiveness,
            two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
            two_d_wf_bg_scale=two_d_wf_bg_scale,
            **common,
        )
    return result["result"]



# ===========================================================================
# Phase 5: High-level convenience function
# ===========================================================================

def deconvolve_image(
    path: Union[str, Path],
    method: str = "ci_rl",
    channels: Optional[Sequence[int]] = None,
    *,
    # Metadata overrides (Consideration 1)
    na: Optional[float] = None,
    refractive_index: Optional[float] = None,
    microscope_type: Optional[str] = None,
    pixel_size_xy: Optional[float] = None,
    pixel_size_z: Optional[float] = None,
    emission_wavelengths: Optional[list[float]] = None,
    excitation_wavelengths: Optional[list[float]] = None,
    pinhole_airy_units: Optional[float | list[float]] = _DEFAULT_PINHOLE_AIRY_UNITS,
    sample_refractive_index: Optional[float] = 1.47,
    overrule_metadata: bool = True,
    # PSF options
    psf_size_xy: Optional[int] = None,
    n_pix_pupil: int = 129,
    ri_coverslip: Optional[float] = None,
    ri_coverslip_design: Optional[float] = None,
    ri_immersion_design: Optional[float] = None,
    t_g: float = 170e3,
    t_g0: float = 170e3,
    t_i0: float = 100e3,
    z_p: float = 0.0,
    # Deconvolution options
    niter: Union[int, list[int]] = 30,
    background: Union[int, str] = "auto",
    damping: Union[str, float] = 0.0,
    offset: Union[str, float] = "auto",
    prefilter_sigma: float = 0.0,
    start: str = "auto",
    convergence: str = "auto",
    rel_threshold: float = 0.005,
    check_every: int = 5,
    two_d_mode: str = "auto",
    two_d_wf_aggressiveness: str = "balanced",
    two_d_wf_bg_radius_um: float = 0.5,
    two_d_wf_bg_scale: float = 1.0,
    device: Optional[str] = None,
    tv_lambda: float = 1e-4,
    sparse_hessian_weight: float = 0.6,
    sparse_hessian_reg: float = 0.98,
) -> dict[str, Any]:
    """Load, generate PSFs, and deconvolve all channels of an OME-TIFF image.

    This is the main entry point for end-to-end deconvolution.

    Parameters
    ----------
    path : str or Path
        Path to OME-TIFF file or companion .ome file.
    method : str
        Deconvolution algorithm (see deconvolve() for options).
    channels : sequence of int, optional
        Which channels to process. None = all channels.
    na, refractive_index, microscope_type, pixel_size_xy, pixel_size_z,
    emission_wavelengths, sample_refractive_index
        Metadata overrides (see load_image() for details).
    psf_size_xy, n_pix_pupil
        PSF generation options (see generate_psf() for details).
    niter, background, device, tv_lambda
        Deconvolution options (see deconvolve() for details).

    Returns
    -------
    dict with keys:
        'channels': list of deconvolved numpy arrays
        'psfs': list of PSF numpy arrays used
        'metadata': microscopy metadata dict
        'source_channels': list of original (unprocessed) numpy arrays
    """
    data = load_image(
        path,
        na=na,
        refractive_index=refractive_index,
        microscope_type=microscope_type,
        pixel_size_xy=pixel_size_xy,
        pixel_size_z=pixel_size_z,
        emission_wavelengths=emission_wavelengths,
        excitation_wavelengths=excitation_wavelengths,
        pinhole_airy_units=pinhole_airy_units,
        sample_refractive_index=sample_refractive_index,
        overrule_metadata=overrule_metadata,
    )

    images = data["images"]
    metadata = data["metadata"]

    if channels is None:
        channels = list(range(len(images)))

    results: list[np.ndarray] = []
    psfs: list[np.ndarray] = []

    for ch_idx in channels:
        if ch_idx >= len(images):
            raise IndexError(
                f"Channel {ch_idx} requested but only {len(images)} available"
            )

        logger.info("Processing channel %d / %d ...", ch_idx + 1, len(channels))

        # Generate PSF for this channel's wavelength
        psf = generate_psf(
            metadata, channel_idx=ch_idx,
            psf_size_xy=psf_size_xy, n_pix_pupil=n_pix_pupil,
            ri_coverslip=ri_coverslip,
            ri_coverslip_design=ri_coverslip_design,
            ri_immersion_design=ri_immersion_design,
            t_g=t_g, t_g0=t_g0, t_i0=t_i0, z_p=z_p,
            two_d_mode=two_d_mode if method in ("ci_rl", "ci_rl_tv") else "legacy_2d",
        )
        psfs.append(psf)

        # Match PSF dimensionality to image
        img = images[ch_idx]
        keep_hidden_2d_psf = (
            img.ndim == 2
            and psf.ndim == 3
            and metadata.get("microscope_type", "widefield") == "widefield"
            and method in ("ci_rl", "ci_rl_tv")
            and str(two_d_mode).strip().lower() == "auto"
        )
        if img.ndim == 2 and psf.ndim == 3 and not keep_hidden_2d_psf:
            psf = psf[psf.shape[0] // 2]  # Take central slice
        elif img.ndim == 3 and psf.ndim == 2:
            # Expand 2D PSF into 3D (single-plane)
            psf = psf[np.newaxis, :, :]

        # Per-channel iteration count
        if isinstance(niter, list):
            ch_niter = niter[ch_idx] if ch_idx < len(niter) else niter[-1]
        else:
            ch_niter = niter

        # Deconvolve
        result = deconvolve(
            img, psf, method=method,
            niter=ch_niter, background=background, damping=damping, offset=offset,
            prefilter_sigma=prefilter_sigma, start=start,
            convergence=convergence, rel_threshold=rel_threshold,
            check_every=check_every, device=device,
            tv_lambda=tv_lambda,
            sparse_hessian_weight=sparse_hessian_weight,
            sparse_hessian_reg=sparse_hessian_reg,
            pixel_size_xy=metadata.get("pixel_size_x"),
            pixel_size_z=metadata.get("pixel_size_z"),
            microscope_type=metadata.get("microscope_type", "widefield"),
            two_d_mode=two_d_mode,
            two_d_wf_aggressiveness=two_d_wf_aggressiveness,
            two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
            two_d_wf_bg_scale=two_d_wf_bg_scale,
        )
        results.append(result)

    return {
        "channels": results,
        "psfs": psfs,
        "metadata": metadata,
        "source_channels": [images[i] for i in channels],
    }


# ===========================================================================
# Phase 6: Save results
# ===========================================================================

# -- Emission‑wavelength → RGB colour mapping for MIP PNGs ----------------

# Fallback palette when no emission wavelength is available
_FALLBACK_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (255, 0, 0),       # Red
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
]


def _emission_to_rgb(wavelength_nm: Optional[float]) -> tuple[int, int, int]:
    """Map an emission wavelength (nm) to an approximate RGB colour.

    Uses a piecewise‑linear visible‑spectrum approximation.  Returns a
    fallback green if the wavelength is *None* or outside 380‑780 nm.
    """
    if wavelength_nm is None:
        return (255, 255, 255)  # white → caller should use fallback palette
    wl = wavelength_nm
    r = g = b = 0.0
    if 380 <= wl < 440:
        r = -(wl - 440) / (440 - 380)
        b = 1.0
    elif 440 <= wl < 490:
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510:
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
    elif 645 <= wl <= 780:
        r = 1.0
    else:
        return (255, 255, 255)  # outside visible → white
    return (int(r * 255), int(g * 255), int(b * 255))


def _channel_color(metadata: dict[str, Any], ch_idx: int) -> tuple[int, int, int]:
    """Determine the display colour for a channel.

    Priority:
      1. Emission wavelength → spectral RGB
      2. Fallback palette (Green, Magenta, Cyan, Red, Blue, Yellow, …)
    """
    channels = metadata.get("channels", [])
    em = None
    if ch_idx < len(channels):
        em = channels[ch_idx].get("emission_wavelength")
    rgb = _emission_to_rgb(em)
    if rgb == (255, 255, 255):
        # Use fallback palette
        rgb = _FALLBACK_COLORS[ch_idx % len(_FALLBACK_COLORS)]
    return rgb


def save_mip_png(
    mip_data: np.ndarray,
    png_path: Union[str, Path],
    metadata: dict[str, Any],
    *,
    channel_indices: Optional[Sequence[int]] = None,
) -> Path:
    """Save a MIP array as a false‑colour PNG image.

    Parameters
    ----------
    mip_data : np.ndarray
        MIP image, shape ``(C, Y, X)`` or ``(Y, X)`` for single channel.
    png_path : str or Path
        Output path for the PNG file.
    metadata : dict
        Microscopy metadata (must contain ``channels`` with emission info).
    channel_indices : sequence of int, optional
        Which metadata channel indices map to each slice of *mip_data*.
        Defaults to ``[0, 1, …, C-1]``.

    Returns
    -------
    Path to saved PNG.
    """
    from PIL import Image

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise to (C, Y, X)
    if mip_data.ndim == 2:
        mip_data = mip_data[np.newaxis]
    n_ch, h, w = mip_data.shape

    if channel_indices is None:
        channel_indices = list(range(n_ch))

    # Resolve colours for each channel; if they are all identical fall back
    # to a Blue-Green-Red-Cyan-Yellow-Magenta cycle so channels are
    # distinguishable.
    _BGRCYM = [
        (0, 0, 255),      # Blue
        (0, 255, 0),      # Green
        (255, 0, 0),      # Red
        (0, 255, 255),    # Cyan
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
    ]
    colors = [_channel_color(metadata, channel_indices[i]) for i in range(n_ch)]
    if n_ch > 1 and len(set(colors)) == 1:
        colors = [_BGRCYM[i % len(_BGRCYM)] for i in range(n_ch)]

    # Build RGB canvas by additive blending of coloured channels
    canvas = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(n_ch):
        ch_img = mip_data[i].astype(np.float64)
        lo, hi = ch_img.min(), ch_img.max()
        if hi > lo:
            ch_img = (ch_img - lo) / (hi - lo)
        else:
            ch_img = np.zeros_like(ch_img)
        rgb = colors[i]
        for c_idx in range(3):
            canvas[:, :, c_idx] += ch_img * (rgb[c_idx] / 255.0)

    # Clip and convert to uint8
    canvas = np.clip(canvas, 0, 1)
    canvas = (canvas * 255).astype(np.uint8)

    img = Image.fromarray(canvas, mode="RGB")
    img.save(str(png_path))
    logger.info("Saved colour MIP PNG to %s", png_path)
    return png_path


def save_result(
    result: dict[str, Any],
    output_path: Union[str, Path],
    *,
    compress: bool = True,
    mip_only: bool = False,
) -> Path:
    """Save deconvolved images as OME-TIFF, preserving metadata.

    Parameters
    ----------
    result : dict
        Output from deconvolve_image().
    output_path : str or Path
        Output file path (.ome.tiff).
    compress : bool
        Whether to apply zlib compression (default: True).
    mip_only : bool
        If True, skip writing OME-TIFF files and only save MIP PNGs.
        Useful for benchmark mode where TIFFs are not needed.

    Returns
    -------
    Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = result["metadata"]
    channels_data = result["channels"]

    # Stack channels into CZYX or CYX
    if channels_data[0].ndim == 3:
        # 3D: stack to (C, Z, Y, X)
        stack = np.stack(channels_data, axis=0)
        axes = "CZYX"
    else:
        # 2D: stack to (C, Y, X)
        stack = np.stack(channels_data, axis=0)
        axes = "CYX"

    px_x = metadata.get("pixel_size_x")
    px_y = metadata.get("pixel_size_y")
    px_z = metadata.get("pixel_size_z")

    resolution = None
    if px_x and px_y:
        # tifffile resolution is in pixels per unit (inverse of pixel size)
        resolution = (1.0 / px_x, 1.0 / px_y)
    resolution_unit = 1  # No standard unit in basic TIFF; OME handles it

    if not mip_only:
        # Build rich OME metadata from the result metadata dict
        channel_names = metadata.get("channel_names") or [
            f"Ch{i}" for i in range(len(channels_data))
        ]
        # Ensure we have enough names for all channels
        while len(channel_names) < len(channels_data):
            channel_names.append(f"Ch{len(channel_names)}")

        ome_meta: dict[str, Any] = {
            "axes": axes,
            "PhysicalSizeX": px_x,
            "PhysicalSizeY": px_y,
            "PhysicalSizeZ": px_z,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZUnit": "µm",
            "Channel": {"Name": channel_names[:len(channels_data)]},
        }

        # Add per-channel emission/excitation wavelengths if available
        ch_info = metadata.get("channels", [])
        if ch_info:
            em_wavelengths = []
            ex_wavelengths = []
            for i, ch in enumerate(ch_info[:len(channels_data)]):
                em = ch.get("emission_wavelength")
                ex = ch.get("excitation_wavelength")
                if em is not None:
                    em_wavelengths.append(float(em))
                if ex is not None:
                    ex_wavelengths.append(float(ex))
            if em_wavelengths:
                ome_meta["Channel"]["EmissionWavelength"] = em_wavelengths
            if ex_wavelengths:
                ome_meta["Channel"]["ExcitationWavelength"] = ex_wavelengths

        # Add objective / instrument metadata as OME Description
        desc_parts = []
        if metadata.get("na") is not None:
            desc_parts.append(f"NA={metadata['na']}")
        if metadata.get("refractive_index") is not None:
            desc_parts.append(f"RI={metadata['refractive_index']}")
        if metadata.get("microscope_type"):
            desc_parts.append(f"Microscope={metadata['microscope_type']}")
        if metadata.get("magnification") is not None:
            desc_parts.append(f"Magnification={metadata['magnification']}x")
        if metadata.get("immersion"):
            desc_parts.append(f"Immersion={metadata['immersion']}")
        if desc_parts:
            ome_meta["Description"] = "; ".join(desc_parts)

        tifffile.imwrite(
            str(output_path),
            stack.astype(np.float32),
            ome=True,
            photometric="minisblack",
            compression="zlib" if compress else None,
            resolution=resolution,
            resolutionunit=resolution_unit,
            metadata=ome_meta,
        )
        logger.info("Saved deconvolved result to %s", output_path)

    # Save maximum intensity projection for 3D results
    # For 2D results, save the image directly with mip_ prefix for montage
    if axes == "CZYX":
        mip = stack.max(axis=1)  # Project along Z → (C, Y, X)
    else:
        mip = stack  # Already (C, Y, X) — use as-is for montage

    if mip is not None:
        mip_path = output_path.parent / ("mip_" + output_path.name)
        tifffile.imwrite(
            str(mip_path),
            mip.astype(np.float32),
            ome=True,
            photometric="minisblack",
            compression="zlib" if compress else None,
            resolution=resolution,
            resolutionunit=resolution_unit,
            metadata={
                "axes": "CYX",
                "PhysicalSizeX": px_x,
                "PhysicalSizeY": px_y,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeYUnit": "µm",
                "Channel": {
                    "Name": [
                        f"Ch{i}" for i in range(len(channels_data))
                    ]
                },
            },
        )
        logger.info("Saved MIP to %s", mip_path)

        # Save colour PNG of MIP (always — needed for montage)
        mip_png = mip_path.with_suffix(".png")
        save_mip_png(mip, mip_png, metadata)

    # Save maximum intensity projection of the source image for 3D data
    # For 2D, save the source directly with mip_ prefix for montage
    source_channels = result.get("source_channels")
    if source_channels:
        src_mip_path = output_path.parent / "mip_source.ome.tiff"
        src_stack = np.stack(source_channels, axis=0)
        if axes == "CZYX":
            src_mip = src_stack.max(axis=1)  # Project along Z → (C, Y, X)
        else:
            src_mip = src_stack  # Already (C, Y, X)
        tifffile.imwrite(
            str(src_mip_path),
            src_mip.astype(np.float32),
            ome=True,
            photometric="minisblack",
            compression="zlib" if compress else None,
            resolution=resolution,
            resolutionunit=resolution_unit,
            metadata={
                "axes": "CYX",
                "PhysicalSizeX": px_x,
                "PhysicalSizeY": px_y,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeYUnit": "µm",
                "Channel": {
                    "Name": [
                        f"Ch{i}" for i in range(len(source_channels))
                    ]
                },
            },
        )
        logger.info("Saved source MIP to %s", src_mip_path)

        # Save colour PNG of source MIP (always — needed for montage)
        src_mip_png = src_mip_path.with_suffix(".png")
        save_mip_png(src_mip, src_mip_png, metadata)

    return output_path
