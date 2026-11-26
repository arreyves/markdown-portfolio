from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class SpatialDataset:
    features: np.ndarray       # Shape: (N, F)
    labels: np.ndarray         # Shape: (N,)
    block_ids: np.ndarray      # Shape: (N,) spatial block IDs to prevent leakage


def generate_synthetic_catchment(
    grid_size: int = 40,
    block_size: int = 10,
    seed: int = 42,
) -> SpatialDataset:
    """Generate spatially correlated multi-spectral + terrain features and LULC labels."""
    rng = np.random.default_rng(seed)
    rows, cols = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")

    # Spatial coordinates normalized to [0, 1]
    lat = rows / grid_size
    lon = cols / grid_size

    # Smooth spatial trend + sensor noise (simulates NDVI, NDWI, Elevation, Slope)
    ndvi = 0.6 * np.sin(2 * np.pi * lat) + 0.3 * np.cos(2 * np.pi * lon) + rng.normal(0, 0.15, (grid_size, grid_size))
    ndwi = -0.5 * ndvi + rng.normal(0, 0.12, (grid_size, grid_size))
    elevation = 500.0 * lat + 200.0 * lon + rng.normal(0, 20.0, (grid_size, grid_size))
    slope = np.hypot(np.gradient(elevation, axis=0), np.gradient(elevation, axis=1))

    # Assign spatial block IDs (prevents random-split spatial autocorrelation leakage)
    block_rows = rows // block_size
    block_cols = cols // block_size
    num_block_cols = (grid_size + block_size - 1) // block_size
    block_ids = (block_rows * num_block_cols + block_cols).ravel()

    # Non-linear class boundaries (0: Bare/Urban, 1: Grassland/Recharge Zone, 2: Dense Vegetation, 3: Water/Wetland)
    score = 1.8 * ndvi - 1.2 * ndwi - 0.002 * elevation + 0.01 * slope
    labels = np.digitize(score, bins=[-0.8, 0.0, 0.9]).ravel()

    features = np.column_stack([
        ndvi.ravel(),
        ndwi.ravel(),
        elevation.ravel(),
        slope.ravel(),
    ])
    return SpatialDataset(features=features, labels=labels, block_ids=block_ids)


def evaluate_spatial_cv(dataset: SpatialDataset, n_splits: int = 4, seed: int = 42) -> dict[str, float]:
    """Run spatially blocked GroupKFold cross-validation to prevent spatial leakage."""
    gkf = GroupKFold(n_splits=n_splits)
    f1_scores: list[float] = []
    bal_accs: list[float] = []

    for train_idx, test_idx in gkf.split(dataset.features, dataset.labels, groups=dataset.block_ids):
        # Guardrail: assert zero spatial block overlap
        overlap = np.intersect1d(dataset.block_ids[train_idx], dataset.block_ids[test_idx])
        if overlap.size > 0:
            raise RuntimeError(f"Spatial leakage detected across blocks: {overlap}")

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed)),
        ])
        clf.fit(dataset.features[train_idx], dataset.labels[train_idx])
        preds = clf.predict(dataset.features[test_idx])

        f1_scores.append(float(f1_score(dataset.labels[test_idx], preds, average="macro")))
        bal_accs.append(float(balanced_accuracy_score(dataset.labels[test_idx], preds)))

    return {
        "macro_f1_mean": float(np.mean(f1_scores)),
        "macro_f1_std": float(np.std(f1_scores)),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
    }


if __name__ == "__main__":
    ds = generate_synthetic_catchment()
    metrics = evaluate_spatial_cv(ds)
    print(f"Spatial CV Results: {metrics}")