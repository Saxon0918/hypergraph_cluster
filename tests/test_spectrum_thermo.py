import numpy as np
import pytest

from hce.core import build_incidence, compute_spectrum, evaluate_thermodynamics
from hce.models import HypergraphSample, SpectrumResult


@pytest.mark.parametrize(
    "sample",
    [
        HypergraphSample("n_lt_M", "test", 2, ((0,), (1,), (0, 1), (0, 1))),
        HypergraphSample("rank_deficient", "test", 3, ((0, 1), (0, 1), (1, 2))),
        HypergraphSample("zero_row", "test", 5, ((0, 1), (1, 2), (0, 2), (0, 1))),
    ],
)
def test_vertex_and_edge_routes_preserve_complete_spectrum(sample):
    H = build_incidence(sample)
    edge = compute_spectrum(H, side="edge", compress_multiplicities=False)
    vertex = compute_spectrum(H, side="vertex", compress_multiplicities=False)
    assert edge.state_count == vertex.state_count == sample.M
    assert np.allclose(edge.values, vertex.values, atol=1e-10)
    betas = [0, 0.1, 3]
    a = evaluate_thermodynamics(edge, betas)
    b = evaluate_thermodynamics(vertex, betas)
    assert np.allclose(a.logZ, b.logZ)
    assert np.allclose(a.U_struct, b.U_struct)
    assert np.allclose(a.entropy, b.entropy)


def test_compressed_multiplicity_entropy_is_not_distinct_value_entropy():
    spectrum = SpectrumResult(
        values=np.array([0.0, 2.0]), multiplicities=np.array([3, 1]), M=4,
        n_active=2, side="manual", lambda_max=2.0, tolerance=0,
        raw_min_eigenvalue=0, operator_dimension=2,
    )
    curve = evaluate_thermodynamics(spectrum, [0])
    assert curve.logZ[0] == pytest.approx(np.log(4))
    assert curve.entropy[0] == pytest.approx(np.log(4))
    assert curve.U_struct[0] == pytest.approx(0.5)


def test_single_edge_and_equal_disjoint_edges_boundaries():
    single = HypergraphSample("one", "test", 3, ((0, 1, 2),))
    curve = evaluate_thermodynamics(compute_spectrum(build_incidence(single)), [0, 10])
    assert np.allclose(curve.U_struct, 3)
    assert np.allclose(curve.entropy, 0)
    assert np.allclose(curve.normalized_entropy, 0)

    disjoint = HypergraphSample("disjoint", "test", 6, ((0, 1), (2, 3), (4, 5)))
    curve = evaluate_thermodynamics(compute_spectrum(build_incidence(disjoint)), [0, 100])
    assert np.allclose(curve.U_struct, 2)
    assert np.allclose(curve.entropy, np.log(3))


def test_repeated_event_changes_zero_modes_and_curves():
    repeated = HypergraphSample("repeat", "test", 2, ((0, 1), (0, 1)))
    simple = HypergraphSample("simple", "test", 2, ((0, 1),))
    r = compute_spectrum(build_incidence(repeated), compress_multiplicities=False)
    s = compute_spectrum(build_incidence(simple), compress_multiplicities=False)
    assert np.allclose(r.values, [0, 4])
    assert np.allclose(s.values, [2])
    assert evaluate_thermodynamics(r, [1]).entropy[0] != pytest.approx(
        evaluate_thermodynamics(s, [1]).entropy[0]
    )


def test_monotonicity_and_derivative_identity():
    sample = HypergraphSample("a", "test", 4, ((0, 1, 2), (1, 2, 3), (0, 3)))
    spectrum = compute_spectrum(build_incidence(sample))
    betas = np.linspace(0, 4, 100)
    curve = evaluate_thermodynamics(spectrum, betas)
    assert np.all(np.diff(curve.U_struct) >= -1e-12)
    assert np.all(np.diff(curve.entropy) <= 1e-12)
    beta = 0.7
    h = 1e-5
    values = evaluate_thermodynamics(spectrum, [beta - h, beta, beta + h])
    d_logz = (values.logZ[2] - values.logZ[0]) / (2 * h)
    d_U = (values.U_struct[2] - values.U_struct[0]) / (2 * h)
    assert d_logz == pytest.approx(values.U_struct[1], rel=1e-8)
    assert d_U == pytest.approx(values.spectral_variance[1], rel=1e-7)

