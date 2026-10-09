"""Reproducibility, hashing, environment capture, and atomic result I/O."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


def stable_seed(root_seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(root_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    entropy = np.frombuffer(digest[:16], dtype=np.uint32)
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def write_json(path: str | Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def write_yaml(path: str | Path, payload: Any) -> None:
    atomic_text(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def write_table(table: pd.DataFrame, path: str | Path) -> Path:
    """Atomically write CSV or Parquet; Parquet absence is an explicit failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    if path.suffix == ".csv":
        table.to_csv(temp, index=False)
    elif path.suffix == ".parquet":
        try:
            table.to_parquet(temp, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet output requires the optional 'parquet' dependency (pyarrow)"
            ) from exc
    else:
        raise ValueError(f"unsupported table suffix {path.suffix}")
    os.replace(temp, path)
    return path


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def environment_record() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip() or None
    except OSError:
        commit = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "blas_threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "git_commit": commit,
    }

