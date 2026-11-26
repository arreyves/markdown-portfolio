import numpy as np
import pytest
import torch
from inversion_engine import (
    forward_infiltration_numpy,
    invert_parameters_logspace,
    train_surrogate,
)


def test_numpy_vectorized_broadcasting_and_validation():
    t = np.linspace(0.1, 10.0, 25)
    ks = np.array([1e-4, 1e-3, 1e-2])
    sorp = np.array([0.05, 0.1, 0.2])

    out = forward_infiltration_numpy(t, ks, sorp)
    assert out.shape == (3, 25)
    assert np.all(np.diff(out, axis=1) > 0)

    with pytest.raises(ValueError):
        forward_infiltration_numpy(np.array([-1.0, 1.0]), ks, sorp)

def test_scipy_logspace_inversion_recovers_ill_conditioned_params():
    rng = np.random.default_rng(123)
    t = np.linspace(0.5, 120.0, 60)
    true_ks = 1.5e-3
    true_sorp = 0.15
    noise_std = 0.005

    clean = forward_infiltration_numpy(t, true_ks, true_sorp)[0]
    noisy = clean + rng.normal(0, noise_std, size=t.shape)

    est = invert_parameters_logspace(t, noisy, noise_std=noise_std)
    assert np.isclose(est["ks_est"], true_ks, rtol=0.10)
    assert np.isclose(est["sorptivity_est"], true_sorp, rtol=0.05)

def test_pytorch_surrogate_deterministic_inference():
    t = np.linspace(0.5, 12.0, 20)
    model = train_surrogate(t, n_samples=400, epochs=250, seed=42)

    test_in = torch.tensor([[-3.5, -1.0], [-4.5, -0.5]], dtype=torch.float32)
    with torch.inference_mode():
        pred_1 = model(test_in)
        pred_2 = model(test_in)

    assert pred_1.shape == (2, 20)
    assert torch.allclose(pred_1, pred_2, atol=1e-7)

    expected = torch.tensor(
        forward_infiltration_numpy(t, 10.0 ** test_in[:, 0].numpy(), 10.0 ** test_in[:, 1].numpy()),
        dtype=torch.float32,
    )
    assert torch.allclose(pred_1, expected, atol=0.15)