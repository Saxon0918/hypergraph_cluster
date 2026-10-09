import numpy as np

from hce.data.preprocess import PreprocessPolicy, preprocess
from hce.data.school import (
    ContactRecord, contacts_to_maximal_groups, right_closed_window_index, window_groups,
)
from hce.models import HypergraphSample
from hce.nulls import run_fixed_margins_chain, sample_size_null


def degrees(sample):
    result = np.zeros(sample.num_vertices, dtype=int)
    for edge in sample.hyperedges:
        result[list(edge)] += 1
    return result


def test_preprocess_keeps_singletons_and_maps_duplicates():
    sample = HypergraphSample("x", "test", 5, ((), (2, 2), (1, 3), (3, 1)))
    cleaned, audit = preprocess(sample)
    assert cleaned.hyperedges == ((2,), (1, 3))
    assert audit.empty_edges_removed == 1
    assert audit.within_edge_duplicates_removed == 1
    assert audit.duplicate_edges_merged == 1
    assert audit.isolated_vertices == 2


def test_N1_preserves_sizes_and_simple_columns():
    sample = HypergraphSample("x", "test", 8, ((0,), (0, 1), (2, 3), (0, 4, 5)))
    null, diag = sample_size_null(sample, 42, simple=True)
    assert sorted(map(len, null.hyperedges)) == sorted(map(len, sample.hyperedges))
    assert len(set(null.hyperedges)) == len(null.hyperedges)
    assert diag.invariant_ok


def test_N2_preserves_degrees_sizes_binary_and_simple():
    sample = HypergraphSample(
        "x", "test", 6,
        ((0, 1, 2), (1, 3), (0, 4), (2, 3, 5), (1, 4, 5)),
    )
    initial_degrees = degrees(sample)
    initial_sizes = [len(edge) for edge in sample.hyperedges]
    snapshots = list(run_fixed_margins_chain(
        sample, seed=7, num_samples=3, burnin_sweeps=2,
        thinning_attempt_sweeps=1, simple=True,
    ))
    assert snapshots
    for current, diag in snapshots:
        assert np.array_equal(degrees(current), initial_degrees)
        assert [len(edge) for edge in current.hyperedges] == initial_sizes
        assert len(set(current.hyperedges)) == len(current.hyperedges)
        assert all(len(edge) == len(set(edge)) for edge in current.hyperedges)
        assert diag.invariant_ok


def test_school_cliques_are_per_timestamp_not_aggregated():
    # Edges (1,2) then (2,3) never coexist, so {1,2,3} must not appear.
    contacts = [ContactRecord(20, 1, 2), ContactRecord(40, 2, 3)]
    table = contacts_to_maximal_groups(contacts)
    assert table.events == ((1, 2), (2, 3))
    assert (1, 2, 3) not in table.events


def test_school_enumerates_all_maximal_cliques_and_right_closed_boundaries():
    contacts = [
        ContactRecord(20, 1, 2), ContactRecord(20, 2, 3), ContactRecord(20, 1, 3),
        ContactRecord(20, 3, 4),
    ]
    table = contacts_to_maximal_groups(contacts)
    assert set(table.events) == {(1, 2, 3), (3, 4)}
    assert right_closed_window_index(600, 0, 600) == 0
    assert right_closed_window_index(601, 0, 600) == 1

