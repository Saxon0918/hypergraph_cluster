import numpy as np
from scipy.linalg import expm

from hce.core import (
    approximate_U,
    build_incidence,
    compute_cumulants,
    compute_spectrum,
    compute_spectrum_and_moments,
    compute_walk_moments,
    evaluate_thermodynamics,
)
from hce.models import HypergraphSample


def golden_sample():
    return HypergraphSample(
        sample_id="golden",
        dataset="test",
        num_vertices=4,
        hyperedges=((0, 1, 2), (1, 2, 3), (0, 3)),
    )


def test_hand_computed_gold_standard():
    H = build_incidence(golden_sample())
    result = compute_walk_moments(H)
    assert (result.W0, result.W1, result.W2, result.W3) == (3, 8, 34, 176)
    assert (result.A2, result.O2) == (22, 12)
    assert (result.A3, result.B3, result.T3) == (62, 102, 12)
    cumulants = compute_cumulants(result.W1, result.W2, result.W3, result.M)
    assert np.allclose(
        [cumulants.kappa1, cumulants.kappa2, cumulants.kappa3],
        [8 / 3, 38 / 9, 160 / 27], atol=1e-12,
    )


def test_joint_operator_reuse_matches_separate_computations():
    H = build_incidence(golden_sample())
    separate_spectrum = compute_spectrum(H, compress_multiplicities=False)
    separate_moments = compute_walk_moments(H)
    joint_spectrum, joint_moments = compute_spectrum_and_moments(
        H, compress_multiplicities=False
    )
    assert np.array_equal(joint_spectrum.values, separate_spectrum.values)
    assert joint_spectrum.side == separate_spectrum.side
    assert joint_moments == separate_moments
    spectrum = compute_spectrum(H, compress_multiplicities=False)
    expected = [1, (7 - np.sqrt(17)) / 2, (7 + np.sqrt(17)) / 2]
    assert np.allclose(spectrum.values, expected, atol=1e-12)
    curve = evaluate_thermodynamics(spectrum, [0.1])
    assert np.isclose(curve.logZ[0], 1.387251474974264)
    assert np.isclose(curve.U_struct[0], 3.113330829741711)
    assert np.isclose(curve.entropy[0], 1.075918392000093)
    assert np.isclose(curve.free_energy[0], -13.872514749742638)
    assert np.allclose(
        [approximate_U(cumulants, [0.1], 1)[0],
         approximate_U(cumulants, [0.1], 2)[0]],
        [3.088888888888889, 3.118518518518519],
    )


def test_small_matrix_exponential_trace_matches_spectrum():
    H = build_incidence(golden_sample())
    G = (H.T @ H).toarray()
    spectrum = compute_spectrum(H)
    for beta in (0.0, 0.1, 2.0):
        curve = evaluate_thermodynamics(spectrum, [beta])
        assert np.isclose(np.trace(expm(beta * G)), np.exp(curve.logZ[0]))


def test_empty_triple_intersection_can_have_positive_T3():
    edges = golden_sample().hyperedges
    assert set(edges[0]) & set(edges[1]) & set(edges[2]) == set()
    assert compute_walk_moments(build_incidence(golden_sample())).T3 == 12

