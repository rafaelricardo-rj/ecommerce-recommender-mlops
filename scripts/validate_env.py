"""Valida as variáveis de ambiente obrigatórias do projeto e informa se
o ambiente local está pronto para execução.
"""

import sys

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentSettings(BaseSettings):
    """Configurações obrigatórias para execução do projeto."""

    data_dir: str = Field(alias="DATA_DIR")
    model_dir: str = Field(alias="MODEL_DIR")
    mlflow_tracking_uri: str = Field(alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(alias="MLFLOW_EXPERIMENT_NAME")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def validate_environment() -> EnvironmentSettings:
    """Carrega e valida as configurações obrigatórias do ambiente.

    Returns:
        EnvironmentSettings: Configurações carregadas do arquivo `.env`.

    Raises:
        ValidationError: Se alguma variável obrigatória estiver ausente.
    """
    return EnvironmentSettings()


def main() -> None:
    """Executa a validação das variáveis de ambiente."""
    try:
        settings = validate_environment()
    except ValidationError as error:
        print("Erro ao validar as variáveis de ambiente.")
        print("Verifique se o arquivo .env existe na raiz do projeto.")
        print(error)
        sys.exit(1)

    print("Ambiente validado com sucesso.")
    print(f"DATA_DIR={settings.data_dir}")
    print(f"MODEL_DIR={settings.model_dir}")
    print(f"MLFLOW_TRACKING_URI={settings.mlflow_tracking_uri}")
    print(f"MLFLOW_EXPERIMENT_NAME={settings.mlflow_experiment_name}")


if __name__ == "__main__":
    main()