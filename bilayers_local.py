"""
bilayers_local.py - Local Bilayers job helper for CIDeconvolve.

Provides a standalone BilayersJob class that reads parameter and path defaults
from a Bilayers config YAML.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

CLASS_SPTCNT = "LOCAL_CLASS_SPTCNT"

DEFAULT_SUFFIXES = (
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".npy",
)


def _str_to_bool(value: str) -> bool:
    """Convert a string to a boolean for argparse."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")

_DEFAULT_BILAYERS_CONFIG = "bilayers_config.yaml"
_DIR_TAG_TO_ATTR = {
    "--input-dir": "input_dir",
    "--infolder": "input_dir",
    "--output-dir": "output_dir",
    "--outfolder": "output_dir",
    "--gt-dir": "gt_dir",
    "--gtfolder": "gt_dir",
}


@dataclass
class ImageResource:
    """Minimal image representation compatible with wrapper_bl.py."""

    filename: str
    filename_original: str
    filepath: Path

    def __post_init__(self) -> None:
        self.filepath = Path(self.filepath)
        self.path = str(self.filepath)


def _collect_images(directory: Path, suffixes) -> List[ImageResource]:
    if not directory.exists():
        return []
    records: List[ImageResource] = []
    for entry in sorted(directory.iterdir()):
        if entry.is_dir() and entry.suffix.lower() == ".zarr":
            records.append(
                ImageResource(
                    filename=entry.name,
                    filename_original=entry.name,
                    filepath=entry,
                )
            )
            continue
        if not entry.is_file():
            continue
        if suffixes and entry.suffix.lower() not in suffixes:
            continue
        records.append(
            ImageResource(
                filename=entry.name,
                filename_original=entry.name,
                filepath=entry,
            )
        )
    return records


def prepare_data(
    discipline: str,
    job: "BilayersJob",
    *,
    is_2d: bool = True,
    **flags,
):
    """Prepare input/output directories and enumerate available images."""
    del discipline, is_2d, flags

    job.input_dir.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.temp_dir.mkdir(parents=True, exist_ok=True)

    in_imgs = _collect_images(job.input_dir, job.suffixes)
    gt_imgs = _collect_images(job.gt_dir, job.suffixes)

    return (
        in_imgs,
        gt_imgs,
        str(job.input_dir),
        str(job.gt_dir),
        str(job.output_dir),
        str(job.temp_dir),
    )


def get_discipline(job: "BilayersJob", default: Optional[str] = None) -> Optional[str]:
    """Return the requested default discipline (placeholder for compatibility)."""
    del job
    return default


def _load_bilayers_config(config_path: Path) -> Dict[str, Any]:
    """Load a Bilayers config YAML and return the parsed dictionary."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "BilayersJob requires PyYAML. Install pyyaml to read Bilayers config files."
        ) from exc

    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Bilayers config must be a YAML mapping, got {type(data).__name__}.")
    return data


def _cli_tag_to_dest(cli_tag: str) -> str:
    return cli_tag.strip().lstrip("-").replace("-", "_")


def _argparse_type_for_bilayers(param_type: str):
    t = str(param_type or "textbox").strip().lower()
    if t == "integer":
        return int
    if t == "float":
        return float
    return str


def _coerce_hidden_value(value: Any):
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "false", "1", "0", "yes", "no"}:
            return _str_to_bool(value)
    return value


def _first_folder_name(entries: List[dict], fallback: str) -> str:
    for entry in entries:
        folder = entry.get("folder_name")
        if isinstance(folder, str) and folder.strip():
            return folder
    return fallback


def _suffixes_from_bilayers_inputs(entries: List[dict]) -> List[str]:
    suffixes: List[str] = []
    for entry in entries:
        formats = entry.get("format")
        if not isinstance(formats, list):
            continue
        for fmt in formats:
            text = str(fmt).strip().lower()
            if not text:
                continue
            if text == "zarr":
                continue
            if not text.startswith("."):
                text = f".{text}"
            if text not in suffixes:
                suffixes.append(text)
    return suffixes or list(DEFAULT_SUFFIXES)


def _parse_bilayers_args(
    argv: Sequence[str],
    *,
    config_path=None,
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Parse CLI arguments using Bilayers YAML metadata."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    default_cfg = config_path or (Path(__file__).with_name(_DEFAULT_BILAYERS_CONFIG))
    bootstrap.add_argument("--bilayers-config", dest="bilayers_config", default=str(default_cfg))
    bootstrap_args, _ = bootstrap.parse_known_args(argv)

    config_file = Path(bootstrap_args.bilayers_config)
    config = _load_bilayers_config(config_file)

    parser = argparse.ArgumentParser(description="Local Bilayers runner for CIDeconvolve.")
    parser.add_argument("--bilayers-config", dest="bilayers_config", default=str(config_file))
    parser.add_argument("--temp-dir", dest="temp_dir", default=None)
    parser.add_argument(
        "--suffixes", nargs="*", default=None,
        help="File suffixes to process; defaults to Bilayers input formats.",
    )

    added_flags: Set[str] = set()

    for entry in config.get("inputs", []) or []:
        cli_tag = entry.get("cli_tag")
        if not cli_tag or cli_tag in added_flags:
            continue
        added_flags.add(cli_tag)
        dest = _DIR_TAG_TO_ATTR.get(cli_tag, entry.get("name") or _cli_tag_to_dest(cli_tag))
        parser.add_argument(
            cli_tag,
            dest=dest,
            default=entry.get("folder_name"),
            type=str,
            help=entry.get("description", ""),
        )

    for entry in config.get("outputs", []) or []:
        cli_tag = entry.get("cli_tag")
        if not cli_tag or cli_tag in added_flags:
            continue
        added_flags.add(cli_tag)
        dest = _DIR_TAG_TO_ATTR.get(cli_tag, entry.get("name") or _cli_tag_to_dest(cli_tag))
        parser.add_argument(
            cli_tag,
            dest=dest,
            default=entry.get("folder_name"),
            type=str,
            help=entry.get("description", ""),
        )

    for param in config.get("parameters", []) or []:
        cli_tag = param.get("cli_tag")
        name = param.get("name")
        if not cli_tag or not name or cli_tag in added_flags:
            continue
        added_flags.add(cli_tag)
        param_type = str(param.get("type", "textbox")).strip().lower()
        default = param.get("default")
        kwargs: Dict[str, Any] = {
            "dest": name,
            "default": default,
            "help": param.get("description", ""),
        }
        if param_type == "checkbox":
            kwargs["nargs"] = "?"
            kwargs["const"] = True
            kwargs["default"] = bool(default) if default is not None else False
            kwargs["type"] = _str_to_bool
            kwargs["metavar"] = "BOOL"
        else:
            kwargs["type"] = _argparse_type_for_bilayers(param_type)
        parser.add_argument(cli_tag, **kwargs)

    args, _ = parser.parse_known_args(argv)

    for hidden in (config.get("exec_function", {}) or {}).get("hidden_args", []) or []:
        cli_tag = hidden.get("cli_tag")
        if not cli_tag:
            continue
        dest = _DIR_TAG_TO_ATTR.get(cli_tag, _cli_tag_to_dest(cli_tag))
        existing = getattr(args, dest, None)
        if existing in (None, ""):
            setattr(args, dest, _coerce_hidden_value(hidden.get("value")))

    if not getattr(args, "input_dir", None):
        args.input_dir = _first_folder_name(config.get("inputs", []) or [], "/data/in")
    if not getattr(args, "output_dir", None):
        args.output_dir = _first_folder_name(config.get("outputs", []) or [], "/data/out")
    if not getattr(args, "gt_dir", None):
        args.gt_dir = "/data/gt"
    if not getattr(args, "suffixes", None):
        args.suffixes = _suffixes_from_bilayers_inputs(config.get("inputs", []) or [])

    param_values = {}
    for param in config.get("parameters", []) or []:
        name = param.get("name")
        if name:
            param_values[name] = getattr(args, name, param.get("default"))
    args.parameters = SimpleNamespace(**param_values)

    return args, config


class BilayersJob:
    """Bilayers-flavored local job initialized from a Bilayers config YAML."""

    @staticmethod
    def _normalise_suffixes(suffixes) -> List[str]:
        if not suffixes:
            return list(DEFAULT_SUFFIXES)
        normalised: List[str] = []
        for suffix in suffixes:
            clean = str(suffix).strip().lower()
            if not clean:
                continue
            if not clean.startswith("."):
                clean = f".{clean}"
            normalised.append(clean)
        return normalised or list(DEFAULT_SUFFIXES)

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        config=None,
        config_path: Optional[Path] = None,
        parameters: Optional[SimpleNamespace] = None,
    ) -> None:
        if parameters is None:
            parameters = getattr(args, "parameters", None)
        if parameters is None:
            parameters = SimpleNamespace()

        self.parameters = parameters
        self.flags = {}
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        self.gt_dir = Path(args.gt_dir)

        temp_dir_value = getattr(args, "temp_dir", None)
        if temp_dir_value is None:
            temp_dir_value = self.output_dir / "tmp"
        self.temp_dir = Path(temp_dir_value)
        self.suffixes = self._normalise_suffixes(getattr(args, "suffixes", None))

        self.config = config or {}
        self.config_path = Path(config_path) if config_path is not None else None

    def __enter__(self) -> "BilayersJob":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    @classmethod
    def from_cli(
        cls,
        argv: Sequence[str],
        *,
        config_path=None,
        **overrides,
    ) -> "BilayersJob":
        args, config = _parse_bilayers_args(argv, config_path=config_path)
        parameters = overrides.pop("parameters", getattr(args, "parameters", None))
        for key, value in overrides.items():
            setattr(args, key, value)
        if parameters is None:
            parameters = getattr(args, "parameters", None)
        return cls(
            args,
            config=config,
            config_path=Path(args.bilayers_config),
            parameters=parameters,
        )

