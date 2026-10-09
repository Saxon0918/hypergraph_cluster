import numpy as np

from hce.evaluation.grouping import find_isomorphism_groups
from hce.features import ScaleModel, build_graph_cache, make_features
from hce.models import HypergraphSample


def test_isomorphism_groups_vertex_relabelings_but_not_nonisomorphic_graphs():
    a = HypergraphSample("a", "test", 4, ((0, 1), (1, 2, 3)))
    b = HypergraphSample("b", "test", 4, ((3, 2), (2, 1, 0)))
    c = HypergraphSample("c", "test", 4, ((0, 1), (2, 3)))
    groups = find_isomorphism_groups([a, b, c])
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_scale_records_only_training_ids_and_features_are_finite():
    samples = [
        HypergraphSample("train1", "test", 3, ((0, 1), (1, 2)), graph_label=0),
        HypergraphSample("train2", "test", 3, ((0, 1, 2),), graph_label=1),
        HypergraphSample("test", "test", 3, ((0,), (1,), (2,)), graph_label=1),
    ]
    caches = [build_graph_cache(sample) for sample in samples]
    scale = ScaleModel().fit(caches[:2])
    assert scale.fit_sample_ids == ("train1", "train2")
    assert "test" not in scale.fit_sample_ids
    for method in ("basic", "cumulants", "low_order", "size_thermo",
                   "gram_thermo", "basic_gram", "netlsd_edge"):
        feature = make_features(caches[2], scale, method)
        assert feature.ndim == 1
        assert np.all(np.isfinite(feature))

