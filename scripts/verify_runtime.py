"""Pytest-free verification for the prescribed PyTorch Conda interpreter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from hce.core import (
    build_incidence, compute_cumulants, compute_spectrum, compute_walk_moments,
    evaluate_thermodynamics, invert_U_exact,
)
from hce.data.hic import parse_hic
from hce.data.preprocess import preprocess
from hce.data.school import ContactRecord, contacts_to_maximal_groups
from hce.evaluation.grouping import find_isomorphism_groups
from hce.features import ScaleModel, build_graph_cache, make_features
from hce.models import HypergraphSample
from hce.nulls import run_fixed_margins_chain, sample_size_null


def degrees(sample):
    result = np.zeros(sample.num_vertices, dtype=int)
    for edge in sample.hyperedges:
        result[list(edge)] += 1
    return result


def main() -> None:
    golden = HypergraphSample("golden", "test", 4,
                              ((0, 1, 2), (1, 2, 3), (0, 3)))
    H = build_incidence(golden)
    moments = compute_walk_moments(H)
    assert (moments.W0, moments.W1, moments.W2, moments.W3) == (3, 8, 34, 176)
    assert (moments.A2, moments.O2, moments.A3, moments.B3, moments.T3) == (
        22, 12, 62, 102, 12
    )
    spectrum_edge = compute_spectrum(H, side="edge", compress_multiplicities=False)
    spectrum_vertex = compute_spectrum(H, side="vertex", compress_multiplicities=False)
    assert np.allclose(spectrum_edge.values, spectrum_vertex.values)
    curve = evaluate_thermodynamics(spectrum_edge, [0, 0.1, 1])
    assert np.all(np.diff(curve.U_struct) >= -1e-12)
    assert np.all(np.diff(curve.entropy) <= 1e-12)
    for beta in (0.01, 0.1, 1):
        target = evaluate_thermodynamics(spectrum_edge, [beta]).U_struct[0]
        inverse = invert_U_exact(spectrum_edge, target)
        assert inverse.status == "ok"
        assert np.isclose(inverse.beta, beta, rtol=1e-8, atol=1e-10)

    samples = parse_hic("tests/fixtures/tiny_hic.txt", "tiny")
    assert len(samples) == 12
    cleaned = [preprocess(sample)[0] for sample in samples]
    caches = [build_graph_cache(sample) for sample in cleaned]
    scale = ScaleModel().fit(caches[:8])
    assert all(sample.sample_id in scale.fit_sample_ids for sample in cleaned[:8])
    assert all(sample.sample_id not in scale.fit_sample_ids for sample in cleaned[8:])
    for method in ("basic", "cumulants", "low_order", "size_thermo",
                   "gram_thermo", "basic_gram", "netlsd_edge", "netlsd_vertex"):
        assert np.all(np.isfinite(make_features(caches[-1], scale, method)))

    n1, _ = sample_size_null(golden, 7)
    assert sorted(map(len, n1.hyperedges)) == sorted(map(len, golden.hyperedges))
    initial_degree = degrees(golden)
    for n2, diagnostics in run_fixed_margins_chain(
        golden, 9, num_samples=3, burnin_sweeps=2, thinning_attempt_sweeps=1
    ):
        assert np.array_equal(initial_degree, degrees(n2))
        assert diagnostics.invariant_ok

    contacts = [
        ContactRecord(20, 1, 2), ContactRecord(20, 2, 3), ContactRecord(20, 1, 3),
        ContactRecord(40, 3, 4),
    ]
    groups = contacts_to_maximal_groups(contacts)
    assert set(groups.events) == {(1, 2, 3), (3, 4)}

    a = HypergraphSample("a", "test", 4, ((0, 1), (1, 2, 3)))
    b = HypergraphSample("b", "test", 4, ((3, 2), (2, 1, 0)))
    assert find_isomorphism_groups([a, b])[0] == find_isomorphism_groups([a, b])[1]
    print("runtime verification passed")


if __name__ == "__main__":
    main()
