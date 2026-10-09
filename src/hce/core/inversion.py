"""Approximate local and exact monotone inversion of U(beta)."""

from __future__ import annotations

import math

import numpy as np

from hce.core.thermodynamics import evaluate_thermodynamics
from hce.models import CumulantResult, InverseResult, SpectrumResult


def invert_U_approx(
    cumulants: CumulantResult,
    U_target: float,
    order: int,
    lambda_max: float,
    tolerance: float = 1e-12,
) -> InverseResult:
    """Invert U1/U2 on the branch continuous from beta=0.

    The calculation is dimensionless and the quadratic root is rationalized to
    avoid subtraction cancellation.
    """
    if order not in {1, 2}:
        raise ValueError("order must be 1 or 2")
    c = float(lambda_max)
    if not np.isfinite(c) or c <= 0:
        return InverseResult(None, "invalid_scale")
    u = float(U_target) / c
    k1 = cumulants.kappa1 / c
    k2 = cumulants.kappa2 / c**2
    k3 = cumulants.kappa3 / c**3
    delta = u - k1
    if delta < -tolerance or u > 1.0 + tolerance:
        return InverseResult(None, "target_out_of_range")
    if abs(delta) <= tolerance:
        return InverseResult(0.0, "ok", residual=delta * c, condition_slope=k2 * c**2)
    if abs(k2) <= tolerance:
        return InverseResult(None, "unidentifiable")

    if order == 1 or abs(k3) <= tolerance:
        x = delta / k2
        fallback = order == 2 and abs(k3) <= tolerance
    else:
        discriminant = k2 * k2 + 2.0 * k3 * delta
        if discriminant < -tolerance:
            return InverseResult(None, "negative_discriminant", diagnostics={"discriminant": discriminant})
        discriminant = max(0.0, discriminant)
        denominator = k2 + math.sqrt(discriminant)
        if abs(denominator) <= tolerance:
            return InverseResult(None, "ill_conditioned_root")
        x = 2.0 * delta / denominator
        fallback = False
    if not np.isfinite(x) or x < -tolerance:
        return InverseResult(None, "invalid_root", diagnostics={"x": x})
    x = max(0.0, x)
    slope_dimensionless = k2 + (k3 * x if order == 2 else 0.0)
    if abs(slope_dimensionless) <= tolerance:
        return InverseResult(None, "ill_conditioned_root", diagnostics={"x": x})
    beta = x / c
    predicted = (
        cumulants.kappa1
        + beta * cumulants.kappa2
        + (0.5 * beta * beta * cumulants.kappa3 if order == 2 else 0.0)
    )
    return InverseResult(
        beta=float(beta),
        status="ok_linear_fallback" if fallback else "ok",
        residual=float(predicted - U_target),
        condition_slope=float(c**2 * slope_dimensionless),
        diagnostics={"x": float(x)},
    )


def invert_U_exact(
    spectrum: SpectrumResult,
    U_target: float,
    x_max: float = 1e6,
    residual_tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> InverseResult:
    """Reference bisection using monotonicity; x=beta*lambda_max is bracketed."""
    lmax = float(spectrum.lambda_max)
    if lmax <= 0:
        return InverseResult(None, "invalid_scale")
    at_zero = evaluate_thermodynamics(spectrum, [0.0])
    mean = float(at_zero.U_struct[0])
    target = float(U_target)
    scale_tol = residual_tolerance * max(1.0, lmax)
    if target < mean - scale_tol or target > lmax + scale_tol:
        return InverseResult(None, "target_out_of_range")
    if abs(target - mean) <= scale_tol:
        return InverseResult(0.0, "ok", residual=mean - target,
                             condition_slope=float(at_zero.spectral_variance[0]))
    if abs(target - lmax) <= scale_tol:
        return InverseResult(None, "asymptotic_boundary")

    x_lo, x_hi = 0.0, 1.0
    upper = evaluate_thermodynamics(spectrum, [x_hi / lmax])
    while float(upper.U_struct[0]) < target and x_hi < x_max:
        x_lo = x_hi
        x_hi = min(x_max, 2.0 * x_hi)
        upper = evaluate_thermodynamics(spectrum, [x_hi / lmax])
    if float(upper.U_struct[0]) < target:
        return InverseResult(None, "unbracketed_saturation", diagnostics={"x_hi": x_hi})

    mid_curve = upper
    for iteration in range(max_iterations):
        x_mid = 0.5 * (x_lo + x_hi)
        mid_curve = evaluate_thermodynamics(spectrum, [x_mid / lmax])
        residual = float(mid_curve.U_struct[0] - target)
        if abs(residual) <= scale_tol or (x_hi - x_lo) <= residual_tolerance * max(1.0, x_mid):
            return InverseResult(
                beta=x_mid / lmax,
                status="ok",
                residual=residual,
                condition_slope=float(mid_curve.spectral_variance[0]),
                diagnostics={"iterations": iteration + 1, "x": x_mid},
            )
        if residual < 0:
            x_lo = x_mid
        else:
            x_hi = x_mid
    x_mid = 0.5 * (x_lo + x_hi)
    residual = float(mid_curve.U_struct[0] - target)
    return InverseResult(
        beta=x_mid / lmax,
        status="max_iterations",
        residual=residual,
        condition_slope=float(mid_curve.spectral_variance[0]),
        diagnostics={"iterations": max_iterations, "x": x_mid},
    )
