from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Define as variáveis de ambiente requisitadas para o projeto
    """

    MODEL_DIR: str
    DATA_DIR: str
    MLFLOW_TRACKING_URI: str
    MLFLOW_EXPERIMENT_NAME: str

    # define a configuração interna do Pydantic para ler o arquivo .env
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


def main() -> None:
    try:
        settings = Settings()
        print("Configurações carregadas com sucesso:")
        print(f"DATA_DIR: {settings.DATA_DIR}")
        print(f"MODEL_DIR: {settings.MODEL_DIR}")
        print(f"MLFLOW_TRACKING_URI: {settings.MLFLOW_TRACKING_URI}")
        print(f"MLFLOW_EXPERIMENT_NAME: {settings.MLFLOW_EXPERIMENT_NAME}")

    except Exception as e:
        print("Erro ao carregar as configurações:")
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
