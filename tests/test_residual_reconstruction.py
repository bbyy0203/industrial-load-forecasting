import numpy as np

from load15_residual_dataset import reconstruct_load_kw


def test_zero_residual_reconstructs_exact_last_week_baseline():
    last_week = np.array([[500.0, 620.0]])
    reconstructed = reconstruct_load_kw(last_week, np.zeros_like(last_week), train_std=432.0)
    np.testing.assert_array_equal(reconstructed, last_week)


def test_positive_residual_is_added_not_subtracted():
    last_week = np.array([500.0])
    actual = np.array([530.0])
    train_std = 10.0
    residual = actual - last_week
    residual_scaled = residual / train_std
    reconstructed = reconstruct_load_kw(last_week, residual_scaled, train_std)
    np.testing.assert_array_equal(residual, np.array([30.0]))
    np.testing.assert_array_equal(reconstructed, actual)
    assert reconstructed[0] == 530.0
    assert reconstructed[0] != 470.0
