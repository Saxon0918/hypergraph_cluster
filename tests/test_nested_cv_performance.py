import numpy as np

import hce.evaluation.nested_cv as nested_cv_module
from hce.evaluation import NestedCVConfig, run_nested_cv
from hce.features import build_graph_cache
from hce.models import HypergraphSample


def _classification_samples():
    samples = []
    for index in range(12):
        if index % 2:
            edges = ((0, 1, 2), (1, 3), (2, 4))
            label = 1
        else:
            edges = ((0, 1), (2, 3), (3, 4))
            label = 0
        samples.append(HypergraphSample(
            f"sample-{index}", "test", 5, edges, graph_label=label,
        ))
    return samples


def test_nested_cv_reuses_fold_features_across_hyperparameters(monkeypatch):
    caches = [build_graph_cache(sample) for sample in _classification_samples()]
    original = nested_cv_module.make_features
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(nested_cv_module, "make_features", counted)
    config = NestedCVConfig(
        outer_folds=2,
        inner_folds=2,
        seeds=(11,),
        C_values=(0.1, 1.0),
        class_weights=(None, "balanced"),
    )
    predictions, metrics, manifests = run_nested_cv(
        caches, "basic", config, groups=None,
    )

    # Each outer fold computes two inner train/valid partitions and one final
    # train/test partition. Candidate count must not multiply feature calls.
    assert calls == 48
    assert len(predictions) == 12
    assert len(metrics) == len(manifests) == 2
    assert set(metrics["estimator"]) == {"LinearSVC"}
    assert np.all(metrics["fit_iterations"] > 0)
