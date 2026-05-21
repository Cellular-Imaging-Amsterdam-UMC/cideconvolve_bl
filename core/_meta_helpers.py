"""Lightweight OME metadata helpers shared by core and GUI modules.

Pure-Python with no heavy dependencies (no torch, no numpy) so they can be
imported at GUI startup without incurring the torch initialisation cost.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

_DEFAULT_PINHOLE_AIRY_UNITS = 1.0


_DYE_WAVELENGTH_TABLE: tuple[dict[str, Any], ...] = (
    {"dye": "DAPI", "aliases": ("dapi",), "excitation": 358.0, "emission": 461.0},
    {"dye": "Hoechst", "aliases": ("hoechst", "hoechst33342", "hoechst33258"), "excitation": 350.0, "emission": 461.0},
    {"dye": "CFP", "aliases": ("cfp", "cyan fluorescent protein"), "excitation": 433.0, "emission": 475.0},
    {"dye": "GFP", "aliases": ("gfp", "egfp"), "excitation": 488.0, "emission": 509.0},
    {"dye": "FITC", "aliases": ("fitc", "fluorescein"), "excitation": 495.0, "emission": 519.0},
    {"dye": "YFP", "aliases": ("yfp", "eyfp"), "excitation": 514.0, "emission": 527.0},
    {"dye": "Cy2", "aliases": ("cy2",), "excitation": 489.0, "emission": 506.0},
    {"dye": "Cy3", "aliases": ("cy3",), "excitation": 550.0, "emission": 570.0},
    {"dye": "TRITC", "aliases": ("tritc", "tetramethylrhodamine"), "excitation": 557.0, "emission": 576.0},
    {"dye": "Rhodamine", "aliases": ("rhodamine", "rhod", "tmr"), "excitation": 540.0, "emission": 625.0},
    {"dye": "mCherry", "aliases": ("mcherry",), "excitation": 587.0, "emission": 610.0},
    {"dye": "RFP", "aliases": ("rfp", "dsred", "tdtomato"), "excitation": 558.0, "emission": 583.0},
    {"dye": "Texas Red", "aliases": ("texasred", "texas red"), "excitation": 595.0, "emission": 615.0},
    {"dye": "Cy5", "aliases": ("cy5",), "excitation": 650.0, "emission": 670.0},
    {"dye": "Alexa Fluor 405", "aliases": ("alexa405", "af405", "a405"), "excitation": 401.0, "emission": 421.0},
    {"dye": "Alexa Fluor 488", "aliases": ("alexa488", "af488", "a488"), "excitation": 495.0, "emission": 519.0},
    {"dye": "Alexa Fluor 568", "aliases": ("alexa568", "af568", "a568"), "excitation": 578.0, "emission": 603.0},
    {"dye": "Alexa Fluor 594", "aliases": ("alexa594", "af594", "a594"), "excitation": 590.0, "emission": 617.0},
    {"dye": "Alexa Fluor 647", "aliases": ("alexa647", "af647", "a647"), "excitation": 650.0, "emission": 668.0},
)


def _normalise_dye_text(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _channel_name_for_dye_lookup(meta: dict[str, Any], channel: dict[str, Any], index: int) -> str:
    for key in ("name", "label", "dye", "fluorophore"):
        value = channel.get(key)
        if value not in (None, ""):
            return str(value)
    names = list(meta.get("channel_names") or [])
    if index < len(names) and names[index] not in (None, ""):
        return str(names[index])
    return ""


def dye_wavelengths_from_name(name: Any) -> Optional[dict[str, Any]]:
    """Return approximate excitation/emission wavelengths inferred from a dye name."""
    normalised = _normalise_dye_text(name)
    if not normalised:
        return None
    best: Optional[dict[str, Any]] = None
    best_len = 0
    for entry in _DYE_WAVELENGTH_TABLE:
        for alias in entry["aliases"]:
            alias_norm = _normalise_dye_text(alias)
            if alias_norm and alias_norm in normalised and len(alias_norm) > best_len:
                best = entry
                best_len = len(alias_norm)
    if best is None:
        return None
    return {
        "dye": best["dye"],
        "excitation_wavelength": float(best["excitation"]),
        "emission_wavelength": float(best["emission"]),
    }


def _missing_positive(value: Any) -> bool:
    try:
        return float(value) <= 0
    except (TypeError, ValueError):
        return True


def apply_dye_wavelength_fallbacks(
    meta: dict[str, Any],
    n_channels: Optional[int] = None,
) -> set[str]:
    """Fill missing channel wavelengths from known dye names.

    Real metadata is left untouched.  The function records inferred fields in
    ``meta["_inferred_keys"]`` and details in ``meta["_dye_wavelength_fallbacks"]``.
    """
    try:
        count = int(n_channels if n_channels is not None else meta.get("size_c") or meta.get("n_channels") or 0)
    except (TypeError, ValueError):
        count = 0
    channels = [dict(ch) if isinstance(ch, dict) else {} for ch in meta.get("channels", [])]
    if count <= 0:
        count = max(len(channels), len(meta.get("channel_names") or []))
    if count <= 0:
        return set()
    if len(channels) < count:
        channels.extend({} for _ in range(count - len(channels)))

    inferred_keys = set(meta.get("_inferred_keys") or [])
    details: list[dict[str, Any]] = list(meta.get("_dye_wavelength_fallbacks") or [])
    applied: set[str] = set()
    for idx in range(count):
        ch = channels[idx]
        name = _channel_name_for_dye_lookup(meta, ch, idx)
        inferred = dye_wavelengths_from_name(name)
        if inferred is None:
            continue
        detail: dict[str, Any] = {
            "channel": idx,
            "name": name,
            "dye": inferred["dye"],
            "fields": [],
        }
        if _missing_positive(ch.get("emission_wavelength")):
            ch["emission_wavelength"] = inferred["emission_wavelength"]
            ch["emission_wavelength_source"] = "dye_name"
            inferred_keys.add("emission_wavelength")
            applied.add("emission_wavelength")
            detail["fields"].append("emission_wavelength")
        if _missing_positive(ch.get("excitation_wavelength")):
            ch["excitation_wavelength"] = inferred["excitation_wavelength"]
            ch["excitation_wavelength_source"] = "dye_name"
            inferred_keys.add("excitation_wavelength")
            applied.add("excitation_wavelength")
            detail["fields"].append("excitation_wavelength")
        if detail["fields"]:
            details.append(detail)
    meta["channels"] = channels[:count]
    if applied:
        meta["_inferred_keys"] = inferred_keys
        meta["_dye_wavelength_fallbacks"] = details
    return applied


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


def _format_float_list(values: list[float]) -> str:
    return ", ".join(f"{value:g}" for value in values)
