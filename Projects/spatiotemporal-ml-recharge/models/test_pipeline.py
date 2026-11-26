import numpy as np
from pipeline import evaluate_spatial_cv, generate_synthetic_catchment


def test_dataset_shapes_and_blocks():
    ds = generate_synthetic_catchment(grid_size=20, block_size=5, seed=7)
    assert ds.features.shape == (400, 4)
    assert ds.labels.shape == (400,)
    assert ds.block_ids.shape == (400,)
    assert len(np.unique(ds.block_ids)) == 16


def test_spatial_cv_performance_and_determinism():
    ds = generate_synthetic_catchment(grid_size=40, block_size=10, seed=42)
    m1 = evaluate_spatial_cv(ds, n_splits=4, seed=42)
    m2 = evaluate_spatial_cv(ds, n_splits=4, seed=42)

    assert np.isclose(m1["macro_f1_mean"], m2["macro_f1_mean"])
    assert m1["macro_f1_mean"] > 0.75
    assert m1["balanced_accuracy_mean"] > 0.75