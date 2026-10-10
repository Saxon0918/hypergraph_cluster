"""Joint exact spectrum and moment computation with shared preprocessing."""

from __future__ import annotations

from hce.core.moments import compute_walk_moments_from_operator
from hce.core.operators import build_dense_gram_operator
from hce.core.spectrum import compute_spectrum_from_operator


def compute_spectrum_and_moments(
    H,
    side: str = "auto",
    psd_relative_tolerance: float = 1e-10,
    memory_budget_gb: float = 8.0,
    compress_multiplicities: bool = True,
):
    """Return ``(spectrum, moments)`` after building one Gram operator."""

    prepared = build_dense_gram_operator(
        H,
        side=side,
        memory_budget_gb=memory_budget_gb,
        workspace_factor=4.0,
        operation="joint spectrum and moment computation",
    )
    moments = compute_walk_moments_from_operator(prepared)
    spectrum = compute_spectrum_from_operator(
        prepared,
        psd_relative_tolerance=psd_relative_tolerance,
        compress_multiplicities=compress_multiplicities,
    )
    return spectrum, moments
