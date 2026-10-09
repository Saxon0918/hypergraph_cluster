"""Manifest-driven downloads with identity and HTML-response safeguards."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import urllib.request
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hce.exceptions import DataFormatError
from hce.utils import write_json


def download_source(
    entry: Mapping[str, Any], output_dir: str | Path, timeout: int = 120
) -> dict:
    """Download one direct URL and verify optional SHA256.

    Google Drive confirmation/HTML pages are rejected. They must be downloaded
    with a confirmation-aware client or supplied locally; an HTML page is never
    renamed and treated as a dataset archive.
    """
    url = str(entry["url"])
    name = str(entry.get("filename") or Path(urlparse(url).path).name)
    if not name:
        raise ValueError("manifest entry needs filename when URL has no basename")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / name
    temporary = destination.with_name(f".{destination.name}.download-{os.getpid()}")
    request = urllib.request.Request(url, headers={"User-Agent": "hce-research/0.1"})
    digest = hashlib.sha256()
    first_chunk = b""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
            content_type = response.headers.get_content_type()
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                if not first_chunk:
                    first_chunk = block[:512].lstrip().lower()
                digest.update(block)
                handle.write(block)
        if content_type in {"text/html", "application/xhtml+xml"} or first_chunk.startswith(
            (b"<!doctype html", b"<html")
        ):
            raise DataFormatError(
                "download returned HTML (often a Google Drive confirmation page), not data"
            )
        actual_hash = digest.hexdigest()
        expected_hash = entry.get("sha256")
        if expected_hash and actual_hash.lower() != str(expected_hash).lower():
            raise DataFormatError(
                f"SHA256 mismatch: expected {expected_hash}, obtained {actual_hash}"
            )
        os.replace(temporary, destination)
        return {
            **entry, "local_path": str(destination.resolve()), "sha256": actual_hash,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "status": "available",
        }
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def fetch_manifest(manifest_path: str | Path, output_dir: str | Path) -> list[dict]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest["sources"] if isinstance(manifest, dict) else manifest
    results = []
    for entry in entries:
        try:
            results.append(download_source(entry, output_dir))
        except Exception as exc:
            results.append({**entry, "status": "failed", "error_type": type(exc).__name__,
                            "reason": str(exc)})
    write_json(Path(output_dir) / "sources_manifest.json", results)
    return results
