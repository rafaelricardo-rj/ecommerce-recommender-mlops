import pandas as pd
from pathlib import Path

from utils import load_data


def filter_active_users(df: pd.DataFrame, min_interactions: int = 3) -> pd.DataFrame:
    """Remove usuários com interações abaixo do mínimo estabelecido."""
    user_counts = df["visitorid"].value_counts()
    active_users = user_counts[user_counts >= min_interactions].index
    return df[df["visitorid"].isin(active_users)]


def main() -> None:
    """Executa o pipeline de pré-processamento."""
    input_path = Path("data/raw/events.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================")
    print("Carregando dados brutos...")
    df = load_data(input_path)

    print("Removendo usuários inativos...")
    df_clean = filter_active_users(df)

    output_path = output_dir / "events_clean.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"Dados limpos salvos em: {output_path}")


if __name__ == "__main__":
    main()
