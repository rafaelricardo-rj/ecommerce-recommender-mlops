"""Configurações centralizadas do projeto usando Pydantic Settings.

Este módulo carrega todas as variáveis de ambiente necessárias para o
treinamento dos modelos. Os valores são lidos do arquivo `.env` na raiz
do projeto, com valores padrão (defaults) caso não estejam definidos.

Exemplo de uso:
    from settings import get_settings
    settings = get_settings()
    print(settings.LEARNING_RATE)  # 0.001
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingSettings(BaseSettings):
    """Centraliza hiperparâmetros e caminhos do projeto.

    Attributes:
        DATA_DIR: Diretório raiz dos dados.
        MODEL_DIR: Diretório onde os modelos treinados são salvos.
        MLFLOW_TRACKING_URI: URL do servidor MLflow.
        MLFLOW_EXPERIMENT_NAME: Nome do experimento no MLflow.
        FEATURES_PATH: Caminho do CSV de features.
        FEATURE_COLUMNS: Colunas de entrada (features).
        TARGET_COLUMN: Coluna alvo (target).
        TEST_SIZE: Proporção dos dados usada para validação.
        RANDOM_SEED: Semente para reprodutibilidade.
        LEARNING_RATE: Taxa de aprendizado inicial do otimizador.
        EPOCHS: Número máximo de épocas de treinamento.
        PATIENCE: Épocas sem melhora antes do Early Stopping.
        MIN_DELTA: Melhoria mínima para considerar progresso.
        HIDDEN_SIZES: Tamanho de cada hidden layer da MLP.
        DROPOUT_RATE: Taxa de dropout para regularização.
        LR_SCHEDULER_FACTOR: Fator de redução do Learning Rate.
        LR_SCHEDULER_PATIENCE: Épocas sem melhora para reduzir LR.
    """

    # --- Diretórios e caminhos ---
    DATA_DIR: str = "data/"
    MODEL_DIR: str = "models/"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "ecommerce_recommender"
    FEATURES_PATH: str = "data/features/user_features.csv"

    # --- Colunas dos dados ---
    FEATURE_COLUMNS: list[str] = ["view", "addtocart"]
    TARGET_COLUMN: str = "transaction"

    # --- Split de dados ---
    TEST_SIZE: float = 0.2
    RANDOM_SEED: int = 42

    # --- Hiperparâmetros da MLP ---
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 300
    PATIENCE: int = 15
    MIN_DELTA: float = 1e-4
    HIDDEN_SIZES: list[int] = [64, 32, 16]
    DROPOUT_RATE: float = 0.3

    # --- Learning Rate Scheduler ---
    LR_SCHEDULER_FACTOR: float = 0.5
    LR_SCHEDULER_PATIENCE: int = 5

    # --- Model Registry ---
    REGISTRY_MLP_NAME: str = "ecommerce-recommender-mlp"
    REGISTRY_LR_NAME: str = "ecommerce-recommender-lr"
    REGISTRY_DUMMY_NAME: str = "ecommerce-recommender-dummy"
    REGISTRY_KNN_NAME: str = "ecommerce-recommender-knn"
    REGISTRY_RF_NAME: str = "ecommerce-recommender-rf"
    REGISTRY_MLP_DESCRIPTION: str = (
        "Rede Neural MLP para predição de transações em e-commerce. "
        "Arquitetura: Linear → BatchNorm → ReLU → Dropout com Early Stopping."
    )
    REGISTRY_LR_DESCRIPTION: str = (
        "Regressão Linear (baseline) para predição de transações. "
        "Modelo simples usado como referência de comparação."
    )
    REGISTRY_DUMMY_DESCRIPTION: str = (
        "DummyRegressor (sanity check) que sempre prediz a média do target. "
        "Limite inferior absoluto — qualquer modelo útil deve superá-lo."
    )
    REGISTRY_KNN_DESCRIPTION: str = (
        "K-Nearest Neighbors Regressor (k=5) — baseline não-paramétrico "
        "baseado em vizinhança local no espaço de features padronizado."
    )
    REGISTRY_RF_DESCRIPTION: str = (
        "Random Forest Regressor (100 árvores, random_state=42) — baseline "
        "ensemble que captura não-linearidades sem treinamento iterativo."
    )
    REGISTRY_MODEL_TAGS: dict[str, str] = {
        "team": "mlops-fiap",
        "domain": "e-commerce",
        "task": "transaction-prediction",
        "data_source": "retailrocket",
    }

    # Configuração interna do Pydantic para ler o arquivo .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> TrainingSettings:
    """Cria e retorna uma instância das configurações.

    Returns:
        TrainingSettings: Objeto com todos os hiperparâmetros carregados.
    """
    return TrainingSettings()
