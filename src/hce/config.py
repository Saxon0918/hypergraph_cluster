"""YAML configuration loading, inheritance, validation, and hashing."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


def _merge(base: dict, override: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    extends = payload.pop("extends", None)
    if extends:
        base_path = (path.parent / extends).resolve()
        payload = _merge(load_config(base_path), payload)
    payload["_config_path"] = str(path)
    validate_config(payload)
    return payload


def project_root(config: Mapping[str, Any]) -> Path:
    """Return the repository root associated with a loaded configuration.

    Relative paths in research configs are repository-relative, not dependent
    on the shell's current working directory.  ``_config_path`` is deliberately
    excluded from config hashes and saved resolved configs, so run identifiers
    remain portable between a workstation and a server.
    """
    config_path = config.get("_config_path")
    if config_path:
        start = Path(config_path).resolve().parent
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
        return start
    return Path.cwd().resolve()


def resolve_project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve an absolute path or a repository-relative configured path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root(config) / path).resolve()


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("dtype", "float64") != "float64":
        raise ValueError("the validated numerical implementation requires float64")
    preprocess = config.get("preprocess", {})
    if preprocess.get("binary_incidence", True) is not True:
        raise ValueError("core Gram experiments require binary incidence")
    if preprocess.get("simplicial_closure", False):
        raise ValueError("simplicial closure is not part of the specified model")
    if int(preprocess.get("min_edge_size", 1)) < 1:
        raise ValueError("min_edge_size must be >= 1")
    spectrum = config.get("spectrum", {})
    if spectrum.get("full_eigenvalues", True) is not True:
        raise ValueError("top-k eigenvalues cannot be presented as exact spectra")
    if spectrum.get("preserve_zero_multiplicity", True) is not True:
        raise ValueError("all M spectral states, including zero modes, are required")


def config_hash(config: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}

    def encode_non_json(value: Any) -> str:
        # PyYAML intentionally resolves ISO-looking scalars such as 2009-10-01
        # to date objects. Normalize them deterministically instead of making
        # otherwise valid research configs fail during run-id generation.
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"unsupported config value type: {type(value).__name__}")

    canonical = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=encode_non_json,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

