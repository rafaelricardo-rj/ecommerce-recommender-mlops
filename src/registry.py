"""Módulo de gerenciamento do MLflow Model Registry.

Responsável por registrar modelos treinados no catálogo oficial
e promover as melhores versões através do ciclo de vida:
    None → Staging → Production.

Uso:
    uv run python src/registry.py
"""

import mlflow
from mlflow import MlflowClient


def register_and_promote(
    run_id: str,
    artifact_path: str,
    registry_name: str,
    new_metrics: dict[str, float],
    description: str = "",
    tags: dict[str, str] | None = None,
    model_tags: dict[str, str] | None = None,
    primary_metric: str = "rmse",
) -> None:
    """Registra o modelo, adiciona metadados e promove se for melhor.

    Fluxo completo: registra → adiciona descrição/tags → compara → promove.

    Args:
        run_id: ID do run do MLflow.
        artifact_path: Caminho do artefato dentro do run.
        registry_name: Nome do modelo no Registry.
        new_metrics: Métricas do modelo recém-treinado.
        description: Descrição do modelo registrado.
        tags: Tags da versão (detalhes técnicos específicos).
        model_tags: Tags do modelo registrado (contexto de negócio).
        primary_metric: Métrica de comparação (default: rmse).
    """
    model_info = register_model(
        run_id,
        artifact_path,
        registry_name,
        description,
        tags,
        model_tags,
    )

    current_metrics = get_production_metrics(registry_name)

    if should_promote_model(new_metrics, current_metrics, primary_metric):
        stage = "Production"
    else:
        stage = "Staging"

    promote_model(registry_name, model_info["version"], stage)


def register_model(
    run_id: str,
    artifact_path: str,
    registry_name: str,
    description: str = "",
    tags: dict[str, str] | None = None,
    model_tags: dict[str, str] | None = None,
) -> dict[str, str]:
    """Registra um modelo do Tracking no Model Registry.

    Cria uma nova versão do modelo no catálogo oficial.
    Se o modelo com esse nome ainda não existir, será criado.
    Após o registro, adiciona descrição e tags de metadados.

    Args:
        run_id: ID do run do MLflow onde o modelo foi logado.
        artifact_path: Caminho do artefato dentro do run.
        registry_name: Nome do modelo no Registry.
        description: Descrição do modelo registrado.
        tags: Tags da versão (ex: {"model_type": "MLP"}).
        model_tags: Tags do modelo registrado (ex: {"team": "mlops"}).

    Returns:
        Dicionário com name, version e status do modelo registrado.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registry_name,
    )

    client = MlflowClient()
    _set_model_metadata(
        client, registry_name, model_version.version, description, tags, model_tags
    )

    print(f"Modelo registrado: {registry_name} v{model_version.version}")
    return {
        "name": model_version.name,
        "version": model_version.version,
        "status": model_version.status,
    }


def _set_model_metadata(
    client: MlflowClient,
    registry_name: str,
    version: str,
    description: str,
    tags: dict[str, str] | None,
    model_tags: dict[str, str] | None = None,
) -> None:
    """Adiciona descrição e tags ao modelo e à versão registrada.

    Args:
        client: Instância do MlflowClient.
        registry_name: Nome do modelo no Registry.
        version: Versão do modelo.
        description: Descrição do modelo.
        tags: Tags da versão (detalhes técnicos específicos).
        model_tags: Tags do modelo registrado (contexto de negócio).
    """
    if description:
        client.update_registered_model(registry_name, description=description)
        client.update_model_version(
            registry_name,
            version,
            description=description,
        )

    for key, value in (tags or {}).items():
        client.set_model_version_tag(registry_name, version, key, value)

    for key, value in (model_tags or {}).items():
        client.set_registered_model_tag(registry_name, key, value)


def promote_model(
    registry_name: str,
    version: str,
    stage: str,
) -> None:
    """Promove uma versão do modelo para um estágio do ciclo de vida.

    Estágios possíveis: "Staging", "Production", "Archived".

    Versões anteriores no mesmo estágio são automaticamente arquivadas via
    `archive_existing_versions=True`, garantindo que cada estágio (Staging /
    Production) referencie no máximo uma versão do modelo de cada vez.

    Args:
        registry_name: Nome do modelo no Registry.
        version: Número da versão a ser promovida.
        stage: Estágio destino ("Staging" ou "Production").
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registry_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    print(f"Modelo {registry_name} v{version} → {stage}")


def get_production_model_version(
    registry_name: str,
) -> str | None:
    """Busca a versão do modelo atualmente em Production.

    Args:
        registry_name: Nome do modelo no Registry.

    Returns:
        Número da versão em Production, ou None se não houver.
    """
    client = MlflowClient()
    versions = client.get_latest_versions(
        name=registry_name,
        stages=["Production"],
    )
    if versions:
        return versions[0].version
    return None


def should_promote_model(
    new_metrics: dict[str, float],
    current_production_metrics: dict[str, float] | None,
    primary_metric: str = "rmse",
    lower_is_better: bool = True,
) -> bool:
    """Decide se o novo modelo deve substituir o atual em Production.
    Critério: o novo modelo só é promovido se a métrica principal
    for MELHOR que a do modelo atualmente em Production.
    Args:
        new_metrics: Métricas do modelo recém-treinado.
        current_production_metrics: Métricas do modelo atual em Production.
            Se None, significa que não há modelo em Production ainda.
        primary_metric: Nome da métrica usada como critério (ex: "rmse").
        lower_is_better: Se True, valores menores são melhores (RMSE, MAE).
    Returns:
        True se o novo modelo deve ser promovido, False caso contrário.
    """
    # Se não existe modelo em Production, o primeiro sempre entra
    if current_production_metrics is None:
        print("Nenhum modelo em Production. Promovendo automaticamente.")
        return True

    new_value = new_metrics[primary_metric]
    current_value = current_production_metrics[primary_metric]

    if lower_is_better:
        is_better = new_value < current_value
    else:
        is_better = new_value > current_value
    print(
        f"Comparação ({primary_metric}): novo={new_value:.6f} vs atual={current_value:.6f}"
    )
    print(f"Resultado: {'✅ Novo é melhor' if is_better else '❌ Atual permanece'}")
    return is_better


def get_production_metrics(
    registry_name: str,
) -> dict[str, float] | None:
    """Busca as métricas do modelo atualmente em Production.

    Acessa o run de origem da versão em Production e retorna
    as métricas canônicas (sem prefixo) logadas no run.

    Args:
        registry_name: Nome do modelo no Registry.

    Returns:
        Dicionário com métricas, ou None se não houver modelo em Production.
    """
    client = MlflowClient()
    versions = client.get_latest_versions(
        name=registry_name,
        stages=["Production"],
    )

    if not versions:
        return None

    # Busca as métricas canônicas (sem prefixo) do run em Production
    run = client.get_run(versions[0].run_id)
    return {
        "rmse": run.data.metrics.get("rmse"),
        "mae": run.data.metrics.get("mae"),
        "r2": run.data.metrics.get("r2"),
        "mse": run.data.metrics.get("mse"),
    }
