"""Módulo principal de treinamento dos modelos de recomendação.

Orquestra o treinamento de dois modelos:
    1. Regressão Linear (baseline) — modelo simples para comparação.
    2. MLP (rede neural) — modelo principal com Early Stopping e LR Scheduler.

Todos os hiperparâmetros são carregados via Pydantic Settings (arquivo .env),
e todos os experimentos são registrados automaticamente no MLflow.

Fluxo de execução:
    main() → prepare_data() → train_baseline() → train_neural_network() → log

Uso:
    uv run python src/train.py
"""

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from models import ModelFactory
from settings import get_settings
from utils import load_data

from registry import register_and_promote

# ═══════════════════════════════════════════════════════════
#  1. PREPARAÇÃO DE DADOS
# ═══════════════════════════════════════════════════════════


def prepare_data(
    features_path: str,
    feature_columns: list[str],
    target_column: str,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carrega, escala e divide os dados em treino/validação.

    Etapas realizadas:
        1. Lê o CSV com as features dos usuários.
        2. Aplica StandardScaler para normalizar as features.
        3. Divide em conjuntos de treino e validação.

    Args:
        features_path: Caminho do arquivo CSV com as features.
        feature_columns: Lista com nomes das colunas de entrada.
        target_column: Nome da coluna alvo.
        test_size: Proporção reservada para validação (ex: 0.2 = 20%).
        random_seed: Semente para reprodutibilidade do split.

    Returns:
        Tupla com (X_train, X_val, y_train, y_val) como arrays NumPy.
    """
    dataframe = load_data(features_path)

    # StandardScaler: transforma cada coluna para média=0 e desvio=1
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(dataframe[feature_columns].values)
    target = dataframe[target_column].values

    return train_test_split(
        features_scaled,
        target,
        test_size=test_size,
        random_state=random_seed,
    )


def convert_to_tensors(
    x_array: np.ndarray,
    y_array: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converte arrays NumPy em tensores PyTorch float32.

    A rede neural PyTorch só aceita tensores como entrada.
    O target é remodelado para (N, 1) para compatibilidade com MSELoss.

    Args:
        x_array: Features no formato NumPy (N, num_features).
        y_array: Target no formato NumPy (N,).

    Returns:
        Tupla (X_tensor, y_tensor) prontos para o modelo.
    """
    x_tensor = torch.tensor(x_array, dtype=torch.float32)
    # .view(-1, 1) transforma [1, 0, 3] em [[1], [0], [3]]
    y_tensor = torch.tensor(y_array, dtype=torch.float32).view(-1, 1)
    return x_tensor, y_tensor


# ═══════════════════════════════════════════════════════════
#  2. MÉTRICAS DE AVALIAÇÃO
# ═══════════════════════════════════════════════════════════


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calcula 4 métricas de avaliação exigidas pelo Tech Challenge.

    Métricas calculadas:
        - MSE: Erro Quadrático Médio — penaliza erros grandes.
        - RMSE: Raiz do MSE — na mesma unidade do target.
        - MAE: Erro Absoluto Médio — média dos erros em módulo.
        - R²: Coeficiente de determinação — quanto o modelo explica.

    Args:
        y_true: Valores reais do target.
        y_pred: Valores preditos pelo modelo.

    Returns:
        Dicionário com as 4 métricas: {"mse", "rmse", "mae", "r2"}.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# ═══════════════════════════════════════════════════════════
#  3. PERSISTÊNCIA DE MODELOS
# ═══════════════════════════════════════════════════════════


def save_model(
    model: nn.Module | object,
    filename: str,
    output_dir: str = "models",
) -> Path:
    """Salva o modelo treinado em disco.

    Detecta automaticamente se é PyTorch (salva state_dict) ou
    Scikit-Learn (salva com joblib).

    Args:
        model: Modelo treinado (PyTorch ou Scikit-Learn).
        filename: Nome do arquivo de saída.
        output_dir: Diretório de destino.

    Returns:
        Path completo onde o modelo foi salvo.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    model_path = output_path / filename

    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), model_path)
    else:
        joblib.dump(model, model_path)

    print(f"Modelo salvo em: {model_path}")
    return model_path


# ═══════════════════════════════════════════════════════════
#  4. TREINAMENTO DA REGRESSÃO LINEAR (BASELINE)
# ═══════════════════════════════════════════════════════════


def train_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """Treina Regressão Linear e avalia no conjunto de VALIDAÇÃO.

    Importante: o modelo é treinado em x_train/y_train mas as métricas
    são calculadas em x_val/y_val, evitando data leakage.

    Args:
        x_train: Features de treino (escaladas).
        y_train: Target de treino.
        x_val: Features de validação (escaladas).
        y_val: Target de validação.

    Returns:
        Dicionário com as métricas calculadas na validação.
    """
    baseline = ModelFactory.create_model("linear_regression")
    baseline.fit(x_train, y_train)

    # Avalia na VALIDAÇÃO (não no treino) para métricas honestas
    predictions = baseline.predict(x_val)
    metrics = calculate_metrics(y_val, predictions)

    # Cria schema de entrada/saída para o registro
    signature = infer_signature(x_val, predictions)
    feature_columns = get_settings().FEATURE_COLUMNS
    input_example = pd.DataFrame(x_val[:2], columns=feature_columns)

    # Registra no MLflow (com e sem prefixo para identificação + canônico para Registry)
    mlflow.log_metrics({f"lr_{k}": v for k, v in metrics.items()})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        baseline,
        "linear_regression_model",
        signature=signature,
        input_example=input_example,
    )
    save_model(baseline, "LR_recommender_model.joblib")

    _print_metrics("Linear Regression (validação)", metrics)
    return metrics


# ═══════════════════════════════════════════════════════════
#  5. TREINAMENTO DA REDE NEURAL (MLP)
# ═══════════════════════════════════════════════════════════


def run_training_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
) -> float:
    """Executa UMA época de treinamento.

    Sequência: forward pass → calcula loss → backward → atualiza pesos.

    Args:
        model: Rede neural em modo treino.
        criterion: Função de perda (MSELoss).
        optimizer: Otimizador (Adam).
        x_train: Tensor de features de treino.
        y_train: Tensor de target de treino.

    Returns:
        Valor da loss de treino (float).
    """
    model.train()  # ← ativa dropout e batchnorm
    optimizer.zero_grad()  # ← zera gradientes acumulados
    train_loss = criterion(model(x_train), y_train)  # ← forward + calcula erro
    train_loss.backward()  # ← calcula gradientes
    optimizer.step()  # ← atualiza pesos
    return train_loss.item()


def evaluate_validation(
    model: nn.Module,
    criterion: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> float:
    """Avalia o modelo no conjunto de validação (sem gradientes).

    Args:
        model: Rede neural a ser avaliada.
        criterion: Função de perda.
        x_val: Tensor de features de validação.
        y_val: Tensor de target de validação.

    Returns:
        Valor da loss de validação (float).
    """
    model.eval()  # ← desativa dropout e batchnorm
    with torch.no_grad():  # ← economiza memória, sem gradientes
        val_loss = criterion(model(x_val), y_val)
    return val_loss.item()


def check_early_stopping(
    val_loss: float,
    best_loss: float,
    epochs_no_improve: int,
    min_delta: float,
) -> tuple[float, int, bool]:
    """Verifica se o treinamento deve parar (Early Stopping).

    Lógica: se a val_loss não melhorar por `patience` épocas
    consecutivas (com melhoria mínima de min_delta), para o treino.

    Args:
        val_loss: Loss de validação da época atual.
        best_loss: Melhor loss de validação registrada até agora.
        epochs_no_improve: Contador de épocas sem melhora.
        min_delta: Melhoria mínima para considerar progresso.

    Returns:
        Tupla (novo_best_loss, novo_contador, houve_melhora).
    """
    improved = val_loss < (best_loss - min_delta)
    if improved:
        return val_loss, 0, True
    return best_loss, epochs_no_improve + 1, False


def train_neural_network(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> tuple[nn.Module, dict[str, float]]:
    """Treina a MLP com Early Stopping e LR Scheduler.

    Args:
        x_train: Tensor de features de treino.
        y_train: Tensor de target de treino.
        x_val: Tensor de features de validação.
        y_val: Tensor de target de validação.

    Returns:
        Tupla (modelo treinado, métricas de validação).
    """
    settings = get_settings()
    input_size = len(settings.FEATURE_COLUMNS)

    # ── Cria modelo, loss e otimizador ──
    model = ModelFactory.create_model(
        "MLP",
        input_size=input_size,
        hidden_sizes=settings.HIDDEN_SIZES,
        dropout_rate=settings.DROPOUT_RATE,
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    # ── LR Scheduler: reduz LR quando val_loss estagna ──
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings.LR_SCHEDULER_FACTOR,
        patience=settings.LR_SCHEDULER_PATIENCE,
    )

    # ── Loop de treinamento ──
    best_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(settings.EPOCHS):
        train_loss = run_training_epoch(
            model,
            criterion,
            optimizer,
            x_train,
            y_train,
        )
        val_loss = evaluate_validation(model, criterion, x_val, y_val)
        current_lr = optimizer.param_groups[0]["lr"]

        # Registra métricas de cada época no MLflow
        mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss, "lr": current_lr},
            step=epoch,
        )

        # Atualiza LR com base na val_loss
        scheduler.step(val_loss)

        # Verifica Early Stopping
        best_loss, epochs_no_improve, improved = check_early_stopping(
            val_loss,
            best_loss,
            epochs_no_improve,
            settings.MIN_DELTA,
        )
        if improved:
            best_model_state = model.state_dict().copy()

        if epochs_no_improve >= settings.PATIENCE:
            print(
                f"Early Stopping na época {epoch + 1} (best val_loss: {best_loss:.6f})"
            )
            break

    # ── Restaura melhor modelo e calcula métricas finais ──
    model.load_state_dict(best_model_state)
    metrics = _evaluate_model_on_validation(model, x_val, y_val)

    _print_metrics("MLP (validação)", metrics)
    return model, metrics


def _evaluate_model_on_validation(
    model: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> dict[str, float]:
    """Avalia o melhor modelo no conjunto de validação.

    Args:
        model: Rede neural com os melhores pesos carregados.
        x_val: Tensor de features de validação.
        y_val: Tensor de target de validação.

    Returns:
        Dicionário com métricas de avaliação.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(x_val).numpy()
    return calculate_metrics(y_val.numpy(), predictions)


def _log_mlp_hyperparams() -> None:
    """Loga os hiperparâmetros da MLP no MLflow."""
    settings = get_settings()
    mlflow.log_params(
        {
            "input_size": len(settings.FEATURE_COLUMNS),
            "hidden_sizes": str(settings.HIDDEN_SIZES),
            "dropout_rate": settings.DROPOUT_RATE,
            "learning_rate": settings.LEARNING_RATE,
            "epochs_max": settings.EPOCHS,
            "patience": settings.PATIENCE,
            "min_delta": settings.MIN_DELTA,
            "lr_scheduler_factor": settings.LR_SCHEDULER_FACTOR,
            "lr_scheduler_patience": settings.LR_SCHEDULER_PATIENCE,
        }
    )


def log_nn_to_mlflow(
    model: nn.Module,
    metrics: dict[str, float],
    x_val: torch.Tensor,
) -> None:
    """Registra o modelo MLP e suas métricas no MLflow.

    Args:
        model: Modelo treinado a ser registrado.
        metrics: Dicionário de métricas a serem logadas.
        x_val: Tensor de features de validação para criar a signature.
    """
    _log_mlp_hyperparams()

    # Loga métricas finais (com e sem prefixo para identificação + canônico para Registry)
    mlflow.log_metrics({f"mlp_{k}": v for k, v in metrics.items()})
    mlflow.log_metrics(metrics)

    # Cria schema de entrada/saída para o registro
    model.eval()
    with torch.no_grad():
        sample_predictions = model(x_val[:2]).numpy()
    signature = infer_signature(x_val[:2].numpy(), sample_predictions)
    input_example = x_val[:2].numpy()

    # Salva e registra o modelo com signature
    model_path = save_model(model, "MLP_recommender_model.pth")
    mlflow.pytorch.log_model(
        model,
        "MLP_model",
        signature=signature,
        input_example=input_example,
    )
    mlflow.log_artifact(str(model_path))


# ═══════════════════════════════════════════════════════════
#  6. UTILITÁRIOS
# ═══════════════════════════════════════════════════════════


def _print_metrics(model_name: str, metrics: dict[str, float]) -> None:
    """Exibe métricas formatadas no console.

    Args:
        model_name: Nome do modelo para o cabeçalho.
        metrics: Dicionário com as métricas.
    """
    print(f"\n[METRICAS] {model_name}:")
    for name, value in metrics.items():
        print(f"   {name.upper():>6}: {value:.6f}")


def _log_model_card_artifact() -> None:
    """Anexa docs/model_card.md ao run ativo do MLflow, se o arquivo existir.

    O caminho é resolvido a partir da localização deste módulo (src/train.py),
    para funcionar independente do diretório de execução.
    """
    model_card_path = Path(__file__).resolve().parent.parent / "docs" / "model_card.md"
    if model_card_path.exists():
        mlflow.log_artifact(str(model_card_path))


def _set_deterministic(seed: int) -> None:
    """Fixa todas as fontes de aleatoriedade para reprodutibilidade entre runs.

    A seed em `settings.RANDOM_SEED` cobre apenas o split do sklearn por padrão.
    Para garantir que treinos consecutivos do PyTorch gerem o mesmo modelo
    (requisito do critério de promoção do Registry baseado em comparação de
    métricas), é necessário fixar adicionalmente Python `random`, NumPy, PyTorch
    CPU/CUDA e o backend cuDNN.

    Args:
        seed: Semente inteira aplicada a todos os RNGs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════
#  7. ORQUESTRADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════


def main() -> None:
    """Executa o pipeline completo de treinamento.

    Fluxo:
        1. Fixa seeds para reprodutibilidade.
        2. Carrega e prepara os dados.
        3. Treina Regressão Linear (baseline) com MLflow tracking.
        4. Treina MLP (rede neural) com MLflow tracking.
        5. Exibe resumo final.
    """
    settings = get_settings()
    _set_deterministic(settings.RANDOM_SEED)

    # ── Preparação dos dados ──
    x_train, x_val, y_train, y_val = prepare_data(
        features_path=settings.FEATURES_PATH,
        feature_columns=settings.FEATURE_COLUMNS,
        target_column=settings.TARGET_COLUMN,
        test_size=settings.TEST_SIZE,
        random_seed=settings.RANDOM_SEED,
    )

    # Converte para tensores PyTorch (usado apenas pela MLP)
    x_train_tensor, y_train_tensor = convert_to_tensors(x_train, y_train)
    x_val_tensor, y_val_tensor = convert_to_tensors(x_val, y_val)

    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # ── 1. Regressão Linear (Baseline) ──
    with mlflow.start_run(run_name="linear_regression_v2") as run:
        print(">> Treinando Linear Regression...")
        lr_metrics = train_baseline(x_train, y_train, x_val, y_val)
        _log_model_card_artifact()

        register_and_promote(
            run_id=run.info.run_id,
            artifact_path="linear_regression_model",
            registry_name=settings.REGISTRY_LR_NAME,
            new_metrics=lr_metrics,
            description=settings.REGISTRY_LR_DESCRIPTION,
            tags={
                "model_type": "LinearRegression",
                "framework": "sklearn",
                "features": ", ".join(settings.FEATURE_COLUMNS),
                "target": settings.TARGET_COLUMN,
                "scaler": "StandardScaler",
            },
            model_tags=settings.REGISTRY_MODEL_TAGS,
        )

    # ── 2. Rede Neural MLP ──
    with mlflow.start_run(run_name="MLP_v2") as run:
        print("\n>> Treinando MLP...")
        model, metrics = train_neural_network(
            x_train_tensor,
            y_train_tensor,
            x_val_tensor,
            y_val_tensor,
        )
        log_nn_to_mlflow(model, metrics, x_val_tensor)
        _log_model_card_artifact()

        register_and_promote(
            run_id=run.info.run_id,
            artifact_path="MLP_model",
            registry_name=settings.REGISTRY_MLP_NAME,
            new_metrics=metrics,
            description=settings.REGISTRY_MLP_DESCRIPTION,
            tags={
                "model_type": "MLP",
                "framework": "pytorch",
                "features": ", ".join(settings.FEATURE_COLUMNS),
                "target": settings.TARGET_COLUMN,
                "scaler": "StandardScaler",
                "architecture": f"[{', '.join(map(str, settings.HIDDEN_SIZES))}]\u21921",
                "early_stopping": f"patience={settings.PATIENCE}",
                "optimizer": "Adam",
                "loss_function": "MSELoss",
            },
            model_tags=settings.REGISTRY_MODEL_TAGS,
        )

    print("\n[OK] Treinamento concluido com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
