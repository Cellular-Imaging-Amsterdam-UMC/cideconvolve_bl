"""Lightweight OME metadata helpers shared by core and GUI modules.

Pure-Python with no heavy dependencies (no torch, no numpy) so they can be
imported at GUI startup without incurring the torch initialisation cost.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

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
