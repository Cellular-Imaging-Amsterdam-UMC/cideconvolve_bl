"""Small OME-Zarr helpers shared by GUI, wrapper, and streaming code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def zarr_attrs(path: Path) -> dict[str, Any]:
    try:
        attrs_path = path / ".zattrs"
        if attrs_path.is_file():
            data = json.loads(attrs_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def is_ome_zarr_image_group(path: Path) -> bool:
    return isinstance(zarr_attrs(path).get("multiscales"), list)


def bioformats2raw_primary_series_path(path: Path) -> Optional[Path]:
    attrs = zarr_attrs(path)
    if "bioformats2raw.layout" not in attrs:
        return None

    series: list[str] = []
    raw_series = zarr_attrs(path / "OME").get("series")
    if isinstance(raw_series, list):
        series.extend(str(item) for item in raw_series if str(item))

    if not series:
        try:
            series.extend(
                child.name
                for child in sorted(path.iterdir(), key=lambda p: p.name)
                if child.is_dir() and is_ome_zarr_image_group(child)
            )
        except Exception:
            pass

    for series_name in series:
        candidate = path / series_name
        if is_ome_zarr_image_group(candidate):
            return candidate
    return None


def resolve_ome_zarr_image_path(path: str | Path) -> Path:
    path = Path(path)
    if is_ome_zarr_image_group(path):
        return path
    if path.is_dir() and path.suffix.lower() == ".zarr":
        series_path = bioformats2raw_primary_series_path(path)
        if series_path is not None:
            return series_path
    return path


def ome_zarr_format_for_path(path: str | Path):
    from ome_zarr.format import CurrentFormat, FormatV04

    attrs = zarr_attrs(resolve_ome_zarr_image_path(path))
    for multiscale in attrs.get("multiscales") or []:
        if str(multiscale.get("version", "")).startswith("0.4"):
            return FormatV04()
    return CurrentFormat()


def open_ome_zarr_image_node(path: str | Path):
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    image_path = resolve_ome_zarr_image_path(path)
    loc = parse_url(str(image_path), fmt=ome_zarr_format_for_path(image_path))
    if loc is None:
        raise ValueError(f"Could not open OME-Zarr path: {image_path}")
    for node in Reader(loc)():
        if isinstance(getattr(node, "data", None), list) and node.data:
            return image_path, node
    raise ValueError(f"No OME-Zarr image node found in {image_path}")
