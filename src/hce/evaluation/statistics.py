"""Pre-specified scalar curve summaries and finite Monte Carlo tails."""

from __future__ import annotations

import numpy as np


def normalized_curve_integral(x, y, x_min: float = 0.01, x_max: float = 10.0) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x >= x_min) & (x <= x_max)
    if np.sum(mask) < 2:
        raise ValueError("curve grid must contain at least two points in the interval")
    logx = np.log(x[mask])
    return float(np.trapz(y[mask], logx) / np.log(x_max / x_min))


def monte_carlo_two_sided_p(real_value: float, null_values) -> float:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    upper = (1 + np.sum(values >= real_value)) / (values.size + 1)
    lower = (1 + np.sum(values <= real_value)) / (values.size + 1)
    return float(min(1.0, 2.0 * min(upper, lower)))


def standardized_difference(real_value: float, null_values) -> tuple[float, str]:
    values = np.asarray(null_values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if std == 0:
        return float("nan"), "degenerate_null_distribution"
    return (float(real_value) - mean) / std, "ok"

