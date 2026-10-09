"""Nested CV that refits scale and standardization inside every inner fold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hce.features import GraphCache, ScaleModel, make_features


@dataclass(frozen=True)
class NestedCVConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    seeds: tuple[int, ...] = (11, 23, 37, 51, 71)
    C_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    class_weights: tuple[Any, ...] = (None, "balanced")
    primary_metric: str = "f1_macro"


def _splitter(y: np.ndarray, groups: np.ndarray | None, folds: int, seed: int):
    class_counts = np.bincount(np.unique(y, return_inverse=True)[1])
    effective = min(int(folds), int(np.min(class_counts)))
    if groups is not None:
        effective = min(effective, len(np.unique(groups)))
    if effective < 2:
        raise ValueError("at least two feasible folds are required")
    if groups is None:
        splitter = StratifiedKFold(n_splits=effective, shuffle=True, random_state=seed)
        return splitter.split(np.zeros(len(y)), y), effective
    splitter = StratifiedGroupKFold(n_splits=effective, shuffle=True, random_state=seed)
    return splitter.split(np.zeros(len(y)), y, groups), effective


def _matrix(
    caches: Sequence[GraphCache],
    indices: Sequence[int],
    scale: ScaleModel,
    method: str,
    x_grid,
) -> np.ndarray:
    return np.vstack([make_features(caches[idx], scale, method, x_grid) for idx in indices])


def _model(C: float, class_weight):
    return Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="linear", C=C, class_weight=class_weight)),
    ])


def _metrics(y_true, y_pred) -> dict:
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def run_nested_cv(
    caches: Sequence[GraphCache],
    method: str,
    config: NestedCVConfig = NestedCVConfig(),
    groups: Sequence[str] | None = None,
    x_grid: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return per-sample predictions, fold metrics, and split manifests.

    Every candidate/inner fold fits c_D using only inner_train. The outer test set
    is touched exactly once after selection and outer_train refitting.
    """
    if not caches:
        raise ValueError("no samples")
    y = np.asarray([cache.sample.graph_label for cache in caches])
    if any(label is None for label in y):
        raise ValueError("classification samples require graph labels")
    group_array = None if groups is None else np.asarray(groups)
    if group_array is not None and len(group_array) != len(caches):
        raise ValueError("groups length mismatch")
    grid = np.logspace(-2, 1, 12) if x_grid is None else np.asarray(x_grid)
    prediction_rows = []
    metric_rows = []
    manifests: list[dict] = []

    for repeat, seed in enumerate(config.seeds):
        outer_iter, actual_outer_folds = _splitter(y, group_array, config.outer_folds, seed)
        for outer_fold, (outer_train, outer_test) in enumerate(outer_iter):
            best = None
            candidates = [(1.0, None)] if method == "dummy" else [
                (float(C), weight) for C in config.C_values for weight in config.class_weights
            ]
            for C, class_weight in candidates:
                scores = []
                inner_groups = None if group_array is None else group_array[outer_train]
                inner_iter, actual_inner_folds = _splitter(
                    y[outer_train], inner_groups, config.inner_folds, seed + outer_fold + 1
                )
                for inner_train_local, inner_valid_local in inner_iter:
                    inner_train = outer_train[inner_train_local]
                    inner_valid = outer_train[inner_valid_local]
                    if method == "dummy":
                        model = DummyClassifier(strategy="most_frequent")
                        X_train = np.zeros((len(inner_train), 1))
                        X_valid = np.zeros((len(inner_valid), 1))
                    else:
                        scale = ScaleModel().fit([caches[idx] for idx in inner_train])
                        X_train = _matrix(caches, inner_train, scale, method, grid)
                        X_valid = _matrix(caches, inner_valid, scale, method, grid)
                        model = _model(C, class_weight)
                    model.fit(X_train, y[inner_train])
                    pred = model.predict(X_valid)
                    scores.append(f1_score(y[inner_valid], pred, average="macro", zero_division=0))
                candidate = (float(np.mean(scores)), -C, class_weight is None, C, class_weight)
                # score high, then lower C (via -C high), then None class weight.
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
            assert best is not None
            _, _, _, best_C, best_weight = best
            if method == "dummy":
                final_scale = None
                model = DummyClassifier(strategy="most_frequent")
                X_train = np.zeros((len(outer_train), 1))
                X_test = np.zeros((len(outer_test), 1))
                fit_hash = None
            else:
                final_scale = ScaleModel().fit([caches[idx] for idx in outer_train])
                X_train = _matrix(caches, outer_train, final_scale, method, grid)
                X_test = _matrix(caches, outer_test, final_scale, method, grid)
                model = _model(best_C, best_weight)
                fit_hash = final_scale.fit_sample_ids_hash
            model.fit(X_train, y[outer_train])
            predicted = model.predict(X_test)
            metrics = _metrics(y[outer_test], predicted)
            metric_rows.append({
                "method": method, "repeat": repeat, "seed": seed,
                "outer_fold": outer_fold, "outer_folds_actual": actual_outer_folds,
                "best_C": best_C, "best_class_weight": best_weight,
                "scale_fit_sample_ids_hash": fit_hash, **metrics,
            })
            for idx, prediction in zip(outer_test, predicted):
                prediction_rows.append({
                    "method": method, "repeat": repeat, "seed": seed,
                    "outer_fold": outer_fold,
                    "sample_id": caches[idx].sample.sample_id,
                    "y_true": y[idx], "y_pred": prediction,
                    "scale_fit_sample_ids_hash": fit_hash,
                })
            manifests.append({
                "repeat": repeat, "seed": seed, "outer_fold": outer_fold,
                "train_ids": [caches[idx].sample.sample_id for idx in outer_train],
                "test_ids": [caches[idx].sample.sample_id for idx in outer_test],
                "grouped": groups is not None,
            })
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows), manifests
