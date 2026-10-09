"""Protocol for HIC/MoCHy/RethinkHGK adapters in separate environments."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from hce.utils import write_json


@dataclass
class ExternalBaselineResult:
    name: str
    status: str
    output_path: str | None = None
    commit: str | None = None
    returncode: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def validate_checkout(path: str | Path, expected_commit: str | None = None) -> str:
    path = Path(path)
    if not (path / ".git").exists():
        raise FileNotFoundError(f"external baseline checkout is absent: {path}")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    commit = result.stdout.strip()
    if expected_commit and commit != expected_commit:
        raise RuntimeError(f"checkout commit {commit} != frozen commit {expected_commit}")
    return commit


def run_external_adapter(
    name: str,
    checkout: str | Path,
    command: Sequence[str],
    protocol_input: Mapping[str, Any],
    work_dir: str | Path,
    expected_commit: str | None = None,
    timeout_seconds: int = 86_400,
) -> ExternalBaselineResult:
    """Run a pinned author-code adapter and require machine-readable output.

    The configured command uses literal ``{input}`` and ``{output}`` placeholders.
    The output JSON must contain ``predictions`` and ``fold_metrics``. This keeps old
    Torch stacks isolated without importing them into the numerical environment.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = validate_checkout(checkout, expected_commit)
    except Exception as exc:
        return ExternalBaselineResult(name, "not_run", diagnostics={"reason": str(exc)})
    input_path = work_dir / f"{name}_input.json"
    output_path = work_dir / f"{name}_output.json"
    write_json(input_path, protocol_input)
    resolved = [str(token).replace("{input}", str(input_path)).replace(
        "{output}", str(output_path)).replace("{checkout}", str(checkout)
    ) for token in command]
    try:
        process = subprocess.run(
            resolved, cwd=str(checkout), timeout=timeout_seconds,
            capture_output=True, text=True, check=False,
        )
        if process.returncode != 0:
            return ExternalBaselineResult(
                name, "failed", commit=commit, returncode=process.returncode,
                diagnostics={"stdout_tail": process.stdout[-4000:],
                             "stderr_tail": process.stderr[-4000:]},
            )
        if not output_path.exists():
            return ExternalBaselineResult(name, "failed", commit=commit,
                                          diagnostics={"reason": "adapter wrote no output"})
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        required = {"predictions", "fold_metrics"}
        if not required.issubset(payload):
            return ExternalBaselineResult(
                name, "failed", commit=commit,
                diagnostics={"reason": f"output missing {sorted(required - set(payload))}"},
            )
        return ExternalBaselineResult(name, "ok", str(output_path), commit,
                                      process.returncode)
    except Exception as exc:
        return ExternalBaselineResult(name, "failed", commit=commit,
                                      diagnostics={"reason": str(exc)})
