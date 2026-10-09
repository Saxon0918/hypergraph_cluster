"""Basic, low-order, thermodynamic, and NetLSD feature families."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from hce.core import (
    build_incidence,
    compute_cumulants,
    compute_spectrum,
    compute_walk_moments,
    evaluate_thermodynamics,
)
from hce.data.preprocess import sample_statistics
from hce.models import CumulantResult, HypergraphSample, MomentResult, SpectrumResult


@dataclass
class GraphCache:
    sample: HypergraphSample
    spectrum: SpectrumResult
    moments: MomentResult
    cumulants: CumulantResult
    statistics: dict


@dataclass
class ScaleModel:
    c_D: float | None = None
    fit_sample_ids_hash: str | None = None
    fit_sample_ids: tuple[str, ...] = ()

    def fit(self, caches: Sequence[GraphCache]) -> "ScaleModel":
        if not caches:
            raise ValueError("cannot fit scale on an empty training set")
        maxima = np.asarray([cache.spectrum.lambda_max for cache in caches])
        if not np.all(np.isfinite(maxima)) or np.any(maxima <= 0):
            raise ValueError("all training lambda_max values must be positive")
        ids = tuple(sorted(cache.sample.sample_id for cache in caches))
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        self.c_D = float(np.median(maxima))
        self.fit_sample_ids_hash = digest
        self.fit_sample_ids = ids
        return self

    def beta_grid(self, x_grid: Sequence[float]) -> np.ndarray:
        if self.c_D is None or self.c_D <= 0:
            raise RuntimeError("ScaleModel must be fit before transform")
        return np.asarray(x_grid, dtype=np.float64) / self.c_D


def build_graph_cache(sample: HypergraphSample, **spectrum_kwargs) -> GraphCache:
    H = build_incidence(sample)
    spectrum = compute_spectrum(H, **spectrum_kwargs)
    moments = compute_walk_moments(H)
    cumulants = compute_cumulants(moments.W1, moments.W2, moments.W3, moments.M)
    return GraphCache(sample, spectrum, moments, cumulants, sample_statistics(sample))


def basic_statistics_features(cache: GraphCache) -> np.ndarray:
    row = cache.statistics
    n = max(1, int(row["n_declared"]))
    M = max(1, int(row["M"]))
    density = row["I_H"] / (n * M)
    fields = [
        np.log1p(row["n_declared"]), np.log1p(row["n_active"]),
        np.log1p(row["M"]), np.log1p(row["I_H"]), density,
        row["isolated_ratio"],
    ]
    for prefix in ("size", "degree"):
        fields.extend(row[f"{prefix}_{name}"] for name in (
            "mean", "std", "median", "p25", "p75", "p90", "max"
        ))
    fields.extend(row[name] for name in (
        "fraction_size1", "fraction_size2", "fraction_size_gt2"
    ))
    return np.nan_to_num(np.asarray(fields, dtype=np.float64), nan=0.0)


def _cumulant_features(cache: GraphCache, c: float) -> np.ndarray:
    k = cache.cumulants
    return np.asarray([k.kappa1 / c, k.kappa2 / c**2, k.kappa3 / c**3])


def _low_order_features(cache: GraphCache, c: float) -> np.ndarray:
    m = cache.moments
    M = m.M
    return np.concatenate([
        _cumulant_features(cache, c),
        np.asarray([
            m.A2 / (M * c**2), m.O2 / (M * c**2),
            m.A3 / (M * c**3), m.B3 / (M * c**3), m.T3 / (M * c**3),
        ]),
    ])


def _thermo_features(
    cache: GraphCache,
    scale: ScaleModel,
    x_grid: Sequence[float],
    include_U: bool = True,
    include_entropy: bool = True,
    size_only: bool = False,
) -> np.ndarray:
    c = float(scale.c_D)
    betas = scale.beta_grid(x_grid)
    spectrum = cache.spectrum
    if size_only:
        sizes = np.asarray([len(edge) for edge in cache.sample.hyperedges], dtype=float)
        spectrum = SpectrumResult(
            values=sizes,
            multiplicities=np.ones(len(sizes), dtype=np.int64),
            M=len(sizes),
            n_active=cache.spectrum.n_active,
            side="diagonal_size",
            lambda_max=float(np.max(sizes)),
            tolerance=0.0,
            raw_min_eigenvalue=float(np.min(sizes)),
            operator_dimension=len(sizes),
        )
    curve = evaluate_thermodynamics(spectrum, betas)
    blocks = []
    if include_U:
        blocks.append(curve.U_struct / c)
    if include_entropy:
        blocks.append(curve.normalized_entropy)
    if not blocks:
        raise ValueError("thermodynamic feature set is empty")
    return np.concatenate(blocks)


def netlsd_features(
    cache: GraphCache,
    times: Sequence[float] | None = None,
    perspective: str = "edge",
) -> np.ndarray:
    """Normalized heat trace of a weighted projection's normalized Laplacian."""
    from scipy import sparse
    from scipy.linalg import eigvalsh

    H = build_incidence(cache.sample)
    if perspective == "edge":
        projection = (H.T @ H).toarray()
    elif perspective == "vertex":
        projection = (H @ H.T).toarray()
    else:
        raise ValueError("perspective must be edge or vertex")
    np.fill_diagonal(projection, 0.0)
    degree = np.sum(projection, axis=1)
    inverse_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inverse_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized_adjacency = inverse_sqrt[:, None] * projection * inverse_sqrt[None, :]
    laplacian = np.eye(len(degree)) - normalized_adjacency
    # Isolated-node rows/columns are zero by project convention.
    laplacian[~positive, :] = 0.0
    laplacian[:, ~positive] = 0.0
    eigenvalues = eigvalsh((laplacian + laplacian.T) * 0.5)
    grid = np.logspace(-2, 2, 250) if times is None else np.asarray(times)
    return np.mean(np.exp(-np.outer(grid, eigenvalues)), axis=1)


def make_features(
    cache: GraphCache,
    scale: ScaleModel,
    method: str,
    x_grid: Sequence[float] | None = None,
) -> np.ndarray:
    grid = np.logspace(-2, 1, 12) if x_grid is None else np.asarray(x_grid)
    c = float(scale.c_D)
    if method == "basic":
        return basic_statistics_features(cache)
    if method == "cumulants":
        return _cumulant_features(cache, c)
    if method == "low_order":
        return _low_order_features(cache, c)
    if method == "size_thermo":
        return _thermo_features(cache, scale, grid, size_only=True)
    if method == "gram_thermo":
        return _thermo_features(cache, scale, grid)
    if method == "gram_U":
        return _thermo_features(cache, scale, grid, include_entropy=False)
    if method == "gram_S":
        return _thermo_features(cache, scale, grid, include_U=False)
    if method == "basic_gram":
        return np.concatenate([basic_statistics_features(cache),
                               _thermo_features(cache, scale, grid)])
    if method == "netlsd_edge":
        return netlsd_features(cache, perspective="edge")
    if method == "netlsd_vertex":
        return netlsd_features(cache, perspective="vertex")
    raise ValueError(f"unknown feature method {method!r}")
