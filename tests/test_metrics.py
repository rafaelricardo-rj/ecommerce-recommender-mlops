"""Testes unitários para a função de métricas em src/train.py.

A função `calculate_metrics` é parte do critério de avaliação do Tech
Challenge ("≥ 4 métricas") — testes garantem que ela sempre retorna as
4 métricas exigidas e que se comporta corretamente em casos extremos.
"""

import numpy as np
import pytest

from train import calculate_metrics


def test_perfect_predictions_yield_zero_error_and_r2_one() -> None:
    """Predições perfeitas: MSE=0, RMSE=0, MAE=0, R²=1."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = calculate_metrics(y_true, y_true)
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_constant_mean_prediction_yields_r2_zero() -> None:
    """Predizer sempre a média deve dar R² = 0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full_like(y_true, y_true.mean())
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["r2"] == pytest.approx(0.0, abs=1e-9)


def test_returns_all_four_required_metrics() -> None:
    """Sempre retorna exatamente as 4 métricas do Tech Challenge."""
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.9])
    metrics = calculate_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2"}


def test_rmse_is_square_root_of_mse() -> None:
    """RMSE = sqrt(MSE) por definição."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 2.5, 3.5])
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


def test_mae_smaller_than_or_equal_to_rmse() -> None:
    """Em qualquer caso, MAE ≤ RMSE (desigualdade de Jensen)."""
    y_true = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
    y_pred = np.array([1.5, 4.0, 4.0, 6.0, 2.5])
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["mae"] <= metrics["rmse"]


def test_r2_with_constant_target_is_zero_when_predictions_differ() -> None:
    """Quando y_true tem variância zero e predições não são exatas, sklearn
    retorna R² = 0.0 (não NaN), graças a `force_finite=True` no default.

    Documenta o comportamento real do sklearn em vez do que a definição
    matemática (SS_res / SS_tot com SS_tot=0) sugeriria. MSE/RMSE/MAE
    seguem definidos normalmente — apenas R² tem caso especial.
    """
    y_true = np.array([3.0, 3.0, 3.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0, 3.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["mse"] > 0
    assert metrics["mae"] > 0
    assert metrics["r2"] == 0.0
