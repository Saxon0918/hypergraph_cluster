"""Reader for timestamped-simplices ScHoLP files, including gzip inputs."""

from __future__ import annotations

import gzip
from datetime import datetime, timedelta, timezone as datetime_timezone
from pathlib import Path
from typing import Iterable, List, Sequence

from hce.exceptions import DataFormatError
from hce.models import EventTable, HypergraphSample


def _read_ints(path: Path) -> list[int]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        values = []
        for line_no, line in enumerate(handle, 1):
            for token in line.split():
                try:
                    values.append(int(token))
                except ValueError as exc:
                    raise DataFormatError(f"non-integer in {path}:{line_no}") from exc
    return values


def _unique_match(directory: Path, marker: str) -> Path:
    candidates = sorted(
        path for path in directory.rglob("*")
        if path.is_file()
        and marker in path.name
        and (path.name.endswith(".txt") or path.name.endswith(".txt.gz"))
    )
    if len(candidates) != 1:
        raise DataFormatError(
            f"expected one *{marker}*.txt[.gz] below {directory}, found "
            f"{[str(p) for p in candidates]}"
        )
    return candidates[0]


def parse_scholp(
    directory: str | Path,
    dataset_name: str,
    timestamp_unit: str,
    timestamp_origin: str | None = None,
    timezone: str | None = None,
) -> EventTable:
    directory = Path(directory)
    simplex_path = _unique_match(directory, "simplices")
    nverts_path = _unique_match(directory, "nverts")
    times_path = _unique_match(directory, "times")
    simplices = _read_ints(simplex_path)
    nverts = _read_ints(nverts_path)
    times = _read_ints(times_path)
    if len(nverts) != len(times):
        raise DataFormatError(
            f"nverts length {len(nverts)} != times length {len(times)}"
        )
    if any(size < 0 for size in nverts):
        raise DataFormatError("nverts cannot contain negative sizes")
    if sum(nverts) != len(simplices):
        raise DataFormatError(
            f"sum(nverts)={sum(nverts)} != len(simplices)={len(simplices)}"
        )
    events = []
    offset = 0
    for size in nverts:
        events.append(tuple(simplices[offset:offset + size]))
        offset += size
    order = sorted(range(len(events)), key=lambda idx: (times[idx], idx))
    sorted_events = tuple(events[idx] for idx in order)
    sorted_times = tuple(times[idx] for idx in order)
    vertex_ids = tuple(sorted(set(simplices)))
    return EventTable(
        dataset=dataset_name,
        vertex_ids=vertex_ids,
        events=sorted_events,
        times=sorted_times,
        original_event_indices=tuple(order),
        timestamp_unit=timestamp_unit,
        timestamp_origin=timestamp_origin,
        timezone=timezone,
        metadata={
            "simplices_path": str(simplex_path),
            "nverts_path": str(nverts_path),
            "times_path": str(times_path),
            "time_min": min(times) if times else None,
            "time_max": max(times) if times else None,
        },
    )


def event_table_to_sample(
    table: EventTable,
    sample_id: str,
    event_indices: Sequence[int] | None = None,
) -> HypergraphSample:
    indices = range(len(table.events)) if event_indices is None else event_indices
    selected_events = [table.events[idx] for idx in indices]
    selected_times = [table.times[idx] for idx in indices]
    vertex_to_local = {value: idx for idx, value in enumerate(table.vertex_ids)}
    edges = [tuple(vertex_to_local[v] for v in edge) for edge in selected_events]
    return HypergraphSample(
        sample_id=sample_id,
        dataset=table.dataset,
        num_vertices=len(table.vertex_ids),
        hyperedges=tuple(edges),
        original_vertex_ids=tuple(str(v) for v in table.vertex_ids),
        event_times=tuple(selected_times),
        metadata={"timestamp_unit": table.timestamp_unit,
                  "timestamp_origin": table.timestamp_origin,
                  "timezone": table.timezone},
    )


def resolved_datetimes(table: EventTable) -> list[datetime]:
    """Resolve timestamps only from explicit metadata, never numeric guessing."""
    if table.timestamp_origin is None:
        raise DataFormatError("timestamp_origin is unverified; calendar windows are unavailable")
    origin_text = table.timestamp_origin.strip().lower()
    if origin_text in {"unix", "unix_epoch", "1970-01-01t00:00:00+00:00"}:
        origin = datetime(1970, 1, 1, tzinfo=datetime_timezone.utc)
    else:
        try:
            origin = datetime.fromisoformat(table.timestamp_origin)
        except ValueError as exc:
            raise DataFormatError("timestamp_origin must be 'unix' or ISO-8601") from exc
        if origin.tzinfo is None:
            raise DataFormatError("timestamp_origin needs an explicit timezone offset")
    unit = table.timestamp_unit.lower()
    factors = {
        "seconds": 1.0, "second": 1.0, "s": 1.0,
        "milliseconds": 1e-3, "millisecond": 1e-3, "ms": 1e-3,
        "days": 86400.0, "day": 86400.0, "d": 86400.0,
    }
    if unit not in factors:
        raise DataFormatError(f"unsupported timestamp unit {table.timestamp_unit!r}")
    return [origin + timedelta(seconds=value * factors[unit]) for value in table.times]


def calendar_window_samples(
    table: EventTable,
    frequency: str,
    min_unique_edges: int = 0,
    merge_duplicate_edges: bool = True,
) -> list[HypergraphSample]:
    """Build verified calendar-year or calendar-quarter samples."""
    dates = resolved_datetimes(table)
    if frequency not in {"year", "quarter"}:
        raise ValueError("frequency must be 'year' or 'quarter'")
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, value in enumerate(dates):
        key = (value.year, 0 if frequency == "year" else (value.month - 1) // 3 + 1)
        buckets.setdefault(key, []).append(index)
    samples = []
    for (year, quarter), indices in sorted(buckets.items()):
        suffix = str(year) if frequency == "year" else f"{year}-Q{quarter}"
        sample = event_table_to_sample(table, f"{table.dataset}:{suffix}", indices)
        edges = tuple(sorted(set(sample.hyperedges))) if merge_duplicate_edges else sample.hyperedges
        if len(set(edges)) < min_unique_edges:
            continue
        samples.append(HypergraphSample(
            sample_id=sample.sample_id, dataset=sample.dataset,
            num_vertices=sample.num_vertices, hyperedges=edges,
            original_vertex_ids=sample.original_vertex_ids,
            event_times=sample.event_times,
            metadata={**sample.metadata, "calendar_window": suffix,
                      "frequency": frequency, "raw_event_count": len(indices)},
        ))
    return samples

