"""Testes unitários para a lógica de decisão de promoção em src/registry.py.

`should_promote_model` é a função pura no centro do critério de avaliação
"MLflow + Registry" (10% da nota). Testes cobrem os 4 cenários de decisão
mais a opção `lower_is_better=False` para métricas onde maior é melhor.
"""

from registry import should_promote_model


def test_promote_when_no_production_exists() -> None:
    """Sem nada em Production, qualquer novo modelo deve ser promovido."""
    result = should_promote_model(
        new_metrics={"rmse": 1.0},
        current_production_metrics=None,
    )
    assert result is True


def test_promote_when_new_rmse_strictly_lower() -> None:
    """RMSE menor que o atual em Production: promove."""
    result = should_promote_model(
        new_metrics={"rmse": 0.5},
        current_production_metrics={"rmse": 1.0},
    )
    assert result is True


def test_do_not_promote_when_new_rmse_higher() -> None:
    """RMSE maior que o atual: não promove."""
    result = should_promote_model(
        new_metrics={"rmse": 1.5},
        current_production_metrics={"rmse": 1.0},
    )
    assert result is False


def test_do_not_promote_on_tie() -> None:
    """Empate exato em RMSE: não promove (regra 'estritamente menor')."""
    result = should_promote_model(
        new_metrics={"rmse": 1.0},
        current_production_metrics={"rmse": 1.0},
    )
    assert result is False


def test_promote_when_higher_is_better() -> None:
    """Com lower_is_better=False, R² maior é promovido."""
    result = should_promote_model(
        new_metrics={"r2": 0.9},
        current_production_metrics={"r2": 0.8},
        primary_metric="r2",
        lower_is_better=False,
    )
    assert result is True


def test_do_not_promote_when_higher_is_better_but_new_is_smaller() -> None:
    """Com lower_is_better=False, R² menor NÃO é promovido."""
    result = should_promote_model(
        new_metrics={"r2": 0.75},
        current_production_metrics={"r2": 0.80},
        primary_metric="r2",
        lower_is_better=False,
    )
    assert result is False
