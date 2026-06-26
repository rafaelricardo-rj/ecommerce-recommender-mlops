"""Testes unitários para src/models.py.

Cobertura:
    - RecommenderMLP: shape da saída, arquitetura customizável.
    - ModelFactory: cria cada tipo suportado, erro em tipo desconhecido.
"""

import pytest
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from models import ModelFactory, RecommenderMLP


class TestRecommenderMLP:
    """Testa a arquitetura e o forward da rede neural."""

    def test_forward_returns_correct_shape(self) -> None:
        """Forward de batch (N, F) deve retornar tensor (N, 1)."""
        model = RecommenderMLP(input_size=2)
        model.eval()
        x = torch.randn(8, 2)
        out = model(x)
        assert out.shape == (8, 1)

    def test_custom_hidden_sizes(self) -> None:
        """Permite arquitetura customizada via hidden_sizes."""
        model = RecommenderMLP(input_size=5, hidden_sizes=[10, 5])
        model.eval()
        x = torch.randn(3, 5)
        assert model(x).shape == (3, 1)

    def test_default_hidden_sizes_when_none(self) -> None:
        """Sem hidden_sizes, usa default [64, 32, 16] sem erro."""
        model = RecommenderMLP(input_size=2, hidden_sizes=None)
        model.eval()
        x = torch.randn(4, 2)
        assert model(x).shape == (4, 1)

    def test_dropout_active_in_train_mode_only(self) -> None:
        """Dropout deve estar ativo em train (saídas variam) e desligado em eval (constantes).

        Reescrito da versão anterior que dependia de seed manual e podia
        passar acidentalmente — agora compara explicitamente os dois modos.
        """
        model = RecommenderMLP(input_size=2, dropout_rate=0.5)
        x = torch.randn(16, 2)

        # eval: dropout desligado → forwards consecutivos são idênticos
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))

        # train: dropout ativo → forwards consecutivos diferem
        model.train()
        with torch.no_grad():
            assert not torch.allclose(model(x), model(x))


class TestModelFactory:
    """Testa o Factory Pattern de criação de modelos."""

    def test_creates_mlp(self) -> None:
        model = ModelFactory.create_model("MLP", input_size=2)
        assert isinstance(model, RecommenderMLP)

    def test_creates_linear_regression(self) -> None:
        model = ModelFactory.create_model("linear_regression")
        assert isinstance(model, LinearRegression)

    def test_creates_dummy_regressor(self) -> None:
        model = ModelFactory.create_model("dummy_regressor")
        assert isinstance(model, DummyRegressor)
        assert model.strategy == "mean"

    def test_creates_knn(self) -> None:
        model = ModelFactory.create_model("knn")
        assert isinstance(model, KNeighborsRegressor)
        assert model.n_neighbors == 5

    def test_creates_random_forest(self) -> None:
        model = ModelFactory.create_model("random_forest")
        assert isinstance(model, RandomForestRegressor)
        # random_state fixo garante reprodutibilidade
        assert model.random_state == 42

    def test_raises_value_error_on_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="desconhecido"):
            ModelFactory.create_model("xgboost")
