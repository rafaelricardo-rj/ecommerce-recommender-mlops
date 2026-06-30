"""Módulo de definição dos modelos de Machine Learning.

Contém a rede neural MLP para recomendação e a Factory responsável
por instanciar qualquer modelo do projeto de forma desacoplada.

Design Pattern utilizado:
    Factory Pattern — permite criar modelos diferentes (MLP + baselines
    sklearn) sem que o código de treinamento precise conhecer os detalhes
    internos de cada um.

Modelos suportados pelo Factory:
    - "MLP": rede neural RecommenderMLP (PyTorch) — modelo principal.
    - "linear_regression": Regressão Linear (baseline paramétrico).
    - "dummy_regressor": prediz a média do target (sanity check).
    - "knn": K-Nearest Neighbors (baseline não-paramétrico).
    - "random_forest": Random Forest (baseline ensemble).
"""

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


class RecommenderMLP(nn.Module):
    """Rede Neural MLP para predição de transações em e-commerce.

    Arquitetura:
        Para cada hidden layer configurada, aplica sequencialmente:
        Linear → BatchNorm → ReLU → Dropout.

        A última camada é um Linear simples que projeta para 1 saída.

    Args:
        input_size: Número de features de entrada (ex: 2 para view + addtocart).
        hidden_sizes: Lista com o tamanho de cada hidden layer.
        dropout_rate: Probabilidade de dropout (0.0 a 1.0).

    Example:
        >>> model = RecommenderMLP(input_size=2, hidden_sizes=[64, 32, 16])
        >>> x = torch.randn(32, 2)  # batch de 32 amostras, 2 features
        >>> output = model(x)       # shape: (32, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.3,
    ) -> None:
        """Constrói as camadas da rede neural.

        Args:
            input_size: Número de features de entrada.
            hidden_sizes: Tamanhos das hidden layers. Default: [64, 32, 16].
            dropout_rate: Taxa de dropout para regularização.
        """
        super().__init__()

        # Se não especificado, usa arquitetura padrão de 3 camadas
        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]

        # ── Monta as camadas dinamicamente ──
        # Começa com input_size e vai reduzindo conforme hidden_sizes
        layers: list[nn.Module] = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),  # ← conexão densa
                    nn.BatchNorm1d(hidden_size),  # ← estabiliza o gradiente
                    nn.ReLU(),  # ← ativação não-linear
                    nn.Dropout(dropout_rate),  # ← desliga neurônios aleatórios
                ]
            )
            previous_size = hidden_size

        # Camada final: projeta para 1 saída (valor de transação)
        layers.append(nn.Linear(previous_size, 1))

        # nn.Sequential executa as camadas em ordem automaticamente
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem dos dados pela rede.

        Args:
            x: Tensor de entrada com shape (batch_size, input_size).

        Returns:
            Tensor com shape (batch_size, 1) — predição de transações.
        """
        return self.network(x)


class ModelFactory:
    """Factory Pattern para criação de modelos.

    Centraliza a lógica de instanciação, permitindo que o código
    de treinamento crie modelos sem conhecer suas classes diretamente.

    Modelos disponíveis:
        - "MLP": rede neural RecommenderMLP (PyTorch) — modelo principal.
        - "linear_regression": Regressão Linear (sklearn) — baseline paramétrico.
        - "dummy_regressor": prediz a média do target (sklearn) — sanity check.
        - "knn": K-Nearest Neighbors (sklearn) — baseline não-paramétrico.
        - "random_forest": Random Forest (sklearn) — baseline ensemble.

    Os baselines sklearn são configurados com seeds fixadas onde aplicável,
    para preservar a reprodutibilidade bit-a-bit garantida pelo pipeline.

    Example:
        >>> model = ModelFactory.create_model("MLP", input_size=2)
        >>> baseline = ModelFactory.create_model("linear_regression")
        >>> dummy = ModelFactory.create_model("dummy_regressor")
    """

    @staticmethod
    def create_model(
        model_type: str,
        **kwargs: int | float | list[int],
    ) -> nn.Module | BaseEstimator:
        """Cria e retorna uma instância do modelo solicitado.

        Retorno é tipado como `nn.Module | BaseEstimator` — o caller sabe
        qual ramo pegou via o parâmetro `model_type` que ele mesmo passou.
        Todos os modelos sklearn aqui herdam de `sklearn.base.BaseEstimator`
        e expõem `.fit` / `.predict`.


        Args:
            model_type: Tipo do modelo. Um de:
                "MLP", "linear_regression", "dummy_regressor", "knn", "random_forest".
            **kwargs: Argumentos passados ao construtor do modelo.
                Para MLP: input_size, hidden_sizes, dropout_rate.

        Returns:
            Instância do modelo pronta para treinar.

        Raises:
            ValueError: Se o model_type não for reconhecido.
        """
        if model_type == "MLP":
            return RecommenderMLP(
                input_size=kwargs.get("input_size", 2),
                hidden_sizes=kwargs.get("hidden_sizes"),
                dropout_rate=kwargs.get("dropout_rate", 0.3),
            )

        if model_type == "linear_regression":
            return LinearRegression()

        if model_type == "dummy_regressor":
            return DummyRegressor(strategy="mean")

        if model_type == "knn":
            return KNeighborsRegressor(n_neighbors=5, n_jobs=1)

        if model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=1,
            )

        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
