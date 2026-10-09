"""Conservative scalar MCMC diagnostics for fixed-margin switch chains."""

from __future__ import annotations

import numpy as np


def autocorrelation(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        return np.asarray([1.0])
    centered = values - np.mean(values)
    variance = float(np.dot(centered, centered))
    if variance == 0:
        return np.concatenate([[1.0], np.zeros(values.size - 1)])
    result = np.correlate(centered, centered, mode="full")[values.size - 1:]
    overlap = np.arange(values.size, 0, -1)
    result = result / overlap
    return result / result[0]


def effective_sample_size(values) -> tuple[float, str]:
    """Initial-positive-sequence ESS; short/constant chains are explicit."""
    values = np.asarray(values, dtype=float)
    if values.size < 4:
        return float("nan"), "insufficient_samples"
    if np.var(values) == 0:
        return float("nan"), "degenerate_trace"
    rho = autocorrelation(values)
    positive_sum = 0.0
    # Geyer's paired initial positive sequence.
    for lag in range(1, len(rho) - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair <= 0:
            break
        positive_sum += pair
    tau = max(1.0, 1.0 + 2.0 * positive_sum)
    return float(min(values.size, values.size / tau)), "ok"


def rhat(chains) -> tuple[float, str]:
    """Classic between/within-chain R-hat for equal truncated chain lengths."""
    arrays = [np.asarray(chain, dtype=float) for chain in chains]
    if len(arrays) < 2:
        return float("nan"), "insufficient_chains"
    length = min(map(len, arrays))
    if length < 4:
        return float("nan"), "insufficient_samples"
    matrix = np.vstack([array[:length] for array in arrays])
    within = float(np.mean(np.var(matrix, axis=1, ddof=1)))
    between = float(length * np.var(np.mean(matrix, axis=1), ddof=1))
    if within == 0:
        return (1.0, "degenerate_identical_chains") if between == 0 else (
            float("inf"), "zero_within_variance"
        )
    variance_hat = ((length - 1) / length) * within + between / length
    return float(np.sqrt(variance_hat / within)), "ok"
