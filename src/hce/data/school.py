"""Primary-school contact parsing and simultaneous maximal-group construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import networkx as nx

from hce.exceptions import DataFormatError
from hce.models import EventTable, HypergraphSample


@dataclass(frozen=True)
class ContactRecord:
    time: int
    source: int
    target: int
    source_class: str | None = None
    target_class: str | None = None


def parse_contacts(path: str | Path) -> list[ContactRecord]:
    records = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) not in {3, 5}:
                raise DataFormatError(
                    f"{path}:{line_no}: expected 't i j [Ci Cj]'"
                )
            try:
                timestamp, left, right = map(int, fields[:3])
            except ValueError as exc:
                raise DataFormatError(f"non-integer contact at line {line_no}") from exc
            source, target = sorted((left, right))
            key = (timestamp, source, target)
            if key in seen:
                continue
            seen.add(key)
            source_class = fields[3] if len(fields) == 5 else None
            target_class = fields[4] if len(fields) == 5 else None
            if left > right:
                source_class, target_class = target_class, source_class
            records.append(ContactRecord(timestamp, source, target,
                                         source_class, target_class))
    return sorted(records, key=lambda row: (row.time, row.source, row.target))


def contacts_to_maximal_groups(
    contacts: Sequence[ContactRecord], dataset_name: str = "contact_primary_school"
) -> EventTable:
    """Enumerate all maximal cliques independently in each 20-second graph."""
    by_time: Dict[int, list[ContactRecord]] = defaultdict(list)
    vertices = set()
    for record in contacts:
        by_time[record.time].append(record)
        vertices.update((record.source, record.target))
    events: list[tuple[int, ...]] = []
    times: list[int] = []
    event_indices: list[int] = []
    for timestamp in sorted(by_time):
        graph = nx.Graph()
        graph.add_edges_from((row.source, row.target) for row in by_time[timestamp])
        # Only vertices incident to a measured contact are in graph, so no artificial
        # singleton groups are created from a fixed school-wide vertex list.
        cliques = sorted(
            (tuple(sorted(clique)) for clique in nx.find_cliques(graph)),
            key=lambda edge: (len(edge), edge),
        )
        for clique in cliques:
            if len(clique) >= 2:
                event_indices.append(len(event_indices))
                events.append(clique)
                times.append(timestamp)
    return EventTable(
        dataset=dataset_name,
        vertex_ids=tuple(sorted(vertices)),
        events=tuple(events),
        times=tuple(times),
        original_event_indices=tuple(event_indices),
        timestamp_unit="seconds",
        timestamp_origin=None,
        timezone=None,
        metadata={"native_contact_seconds": 20,
                  "construction": "maximal_cliques_per_timestamp"},
    )


def right_closed_window_index(timestamp: int, origin: int, width: int) -> int:
    """Index windows (origin+k*w, origin+(k+1)*w], including origin in k=0."""
    if width <= 0:
        raise ValueError("window width must be positive")
    delta = timestamp - origin
    if delta <= 0:
        return 0
    return (delta - 1) // width


def window_groups(
    table: EventTable,
    width_seconds: int,
    origin: int | None = None,
    merge_duplicate_edges: bool = True,
) -> list[HypergraphSample]:
    if not table.times:
        return []
    origin = int(min(table.times) if origin is None else origin)
    vertex_to_local = {value: idx for idx, value in enumerate(table.vertex_ids)}
    buckets: Dict[int, list[int]] = defaultdict(list)
    for event_idx, timestamp in enumerate(table.times):
        buckets[right_closed_window_index(timestamp, origin, width_seconds)].append(event_idx)
    max_bucket = right_closed_window_index(max(table.times), origin, width_seconds)
    samples = []
    for bucket in range(max_bucket + 1):
        event_indices = buckets.get(bucket, [])
        edges = [tuple(vertex_to_local[v] for v in table.events[idx]) for idx in event_indices]
        raw_count = len(edges)
        if merge_duplicate_edges:
            edges = sorted(set(edges))
        start = origin + bucket * width_seconds
        end = start + width_seconds
        samples.append(
            HypergraphSample(
                sample_id=f"{table.dataset}:{start}:{end}",
                dataset=table.dataset,
                num_vertices=len(table.vertex_ids),
                hyperedges=tuple(edges),
                original_vertex_ids=tuple(str(v) for v in table.vertex_ids),
                event_times=tuple(table.times[idx] for idx in event_indices),
                metadata={
                    "time_start": start,
                    "time_end": end,
                    "interval_convention": "right_closed",
                    "N_group_events": raw_count,
                    "M_unique": len(set(edges)),
                },
            )
        )
    return samples


def cross_class_contact_ratio(
    contacts: Sequence[ContactRecord], teacher_labels=("Teachers", "Teacher")
) -> dict:
    known = cross = unknown = teacher_involved = 0
    for row in contacts:
        if row.source_class is None or row.target_class is None:
            unknown += 1
            continue
        known += 1
        if row.source_class in teacher_labels or row.target_class in teacher_labels:
            teacher_involved += 1
        if row.source_class != row.target_class:
            cross += 1
    return {
        "R_cross": cross / known if known else None,
        "known_class_pair_records": known,
        "unknown_class_pair_records": unknown,
        "teacher_involved_records": teacher_involved,
    }
