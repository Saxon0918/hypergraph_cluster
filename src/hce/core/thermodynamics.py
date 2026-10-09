"""Stable exact spectral thermodynamics without materializing exp(beta G)."""

from __future__ import annotations

import numpy as np

from hce.models import SpectrumResult, ThermoCurves


def evaluate_thermodynamics(spectrum: SpectrumResult, betas) -> ThermoCurves:
    """Evaluate logZ, U, entropy and responses for non-negative beta values."""
    beta = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    if beta.ndim != 1 or not np.all(np.isfinite(beta)) or np.any(beta < 0):
        raise ValueError("betas must be a finite one-dimensional non-negative array")
    if spectrum.M <= 0 or spectrum.state_count != spectrum.M:
        raise ValueError("spectrum must preserve exactly M spectral states")

    values = np.asarray(spectrum.values, dtype=np.float64)
    weights = np.asarray(spectrum.multiplicities, dtype=np.float64)
    lmax = float(spectrum.lambda_max)
    logZ = np.empty_like(beta)
    U = np.empty_like(beta)
    entropy = np.empty_like(beta)
    variance = np.empty_like(beta)
    free_energy = np.full_like(beta, np.nan)
    free_energy_excess = np.empty_like(beta)
    status = []

    for idx, b in enumerate(beta):
        shifted = b * (values - lmax)
        masses = weights * np.exp(shifted)
        mass_sum = float(np.sum(masses))
        aggregate_probability = masses / mass_sum
        L = float(np.log(mass_sum))
        logZ[idx] = b * lmax + L
        U[idx] = float(np.dot(aggregate_probability, values))
        # This is the entropy over all M modes. Multiplicities are aggregate mass,
        # not a reduced set of equally weighted distinct eigenvalues.
        entropy[idx] = L - float(np.dot(aggregate_probability, shifted))
        centered = values - U[idx]
        variance[idx] = float(np.dot(aggregate_probability, centered * centered))
        if b > 0:
            free_energy[idx] = -logZ[idx] / b
            free_energy_excess[idx] = -(logZ[idx] - np.log(spectrum.M)) / b
            status.append("ok")
        else:
            free_energy_excess[idx] = -U[idx]
            status.append("free_energy_undefined_at_beta_zero")

    if spectrum.M == 1:
        normalized_entropy = np.zeros_like(entropy)
    else:
        normalized_entropy = entropy / np.log(spectrum.M)
    return ThermoCurves(
        beta=beta,
        logZ=logZ,
        U_struct=U,
        entropy=entropy,
        free_energy=free_energy,
        free_energy_excess=free_energy_excess,
        spectral_variance=variance,
        heat_capacity=beta * beta * variance,
        normalized_entropy=normalized_entropy,
        status=status,
    )

