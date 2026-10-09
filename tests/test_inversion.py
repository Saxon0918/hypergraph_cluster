import numpy as np

from hce.core import (
    build_incidence, compute_cumulants, compute_spectrum, compute_walk_moments,
    evaluate_thermodynamics, invert_U_approx, invert_U_exact,
)
from hce.models import HypergraphSample


def setup_values():
    sample = HypergraphSample("a", "test", 4, ((0, 1, 2), (1, 2, 3), (0, 3)))
    H = build_incidence(sample)
    moments = compute_walk_moments(H)
    return (
        compute_spectrum(H),
        compute_cumulants(moments.W1, moments.W2, moments.W3, moments.M),
    )


def test_exact_inversion_recovers_beta():
    spectrum, _ = setup_values()
    for beta in (0.01, 0.1, 1.0):
        target = evaluate_thermodynamics(spectrum, [beta]).U_struct[0]
        result = invert_U_exact(spectrum, target)
        assert result.status == "ok"
        assert np.isclose(result.beta, beta, rtol=1e-8, atol=1e-10)


def test_approximate_inverse_statuses_and_local_accuracy():
    spectrum, cumulants = setup_values()
    target = evaluate_thermodynamics(spectrum, [1e-4]).U_struct[0]
    result = invert_U_approx(cumulants, target, 2, spectrum.lambda_max)
    assert result.status == "ok"
    assert np.isclose(result.beta, 1e-4, rtol=1e-3)
    assert invert_U_approx(cumulants, -1, 2, spectrum.lambda_max).status == "target_out_of_range"


def test_degenerate_spectrum_is_unidentifiable():
    sample = HypergraphSample("d", "test", 4, ((0, 1), (2, 3)))
    H = build_incidence(sample)
    moments = compute_walk_moments(H)
    cumulants = compute_cumulants(moments.W1, moments.W2, moments.W3, moments.M)
    result = invert_U_approx(cumulants, 2.1, 2, 2.0)
    assert result.status in {"target_out_of_range", "unidentifiable"}

