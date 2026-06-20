import pandas as pd
from pathlib import Path

from utils import load_data


def create_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gera features agregadas por comportamento do usuário."""
    # Conta a quantidade de cada tipo de evento por usuário
    features = df.groupby("visitorid")["event"].value_counts().unstack(fill_value=0)
    features.columns.name = None
    features = features.reset_index()

    # Preenchendo possíveis valores nulos caso algum evento não exista
    for col in ["view", "addtocart", "transaction"]:
        if col not in features.columns:
            features[col] = 0

    # Calcula o total de interações para cada usuário
    event_cols = [
        col for col in ["view", "addtocart", "transaction"] if col in features.columns
    ]
    features["total_interactions"] = features[event_cols].sum(axis=1)

    return features


def main():
    """Executa a etapa de engenharia de features."""
    input_path = Path("data/processed/events_clean.csv")
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando dados limpos...")
    df_clean = load_data(input_path)

    print("Criando features de consumo dos usuários...")
    df_features = create_user_features(df_clean)

    output_path = output_dir / "user_features.csv"
    df_features.to_csv(output_path, index=False)
    print(f"Features geradas e salvas em: {output_path}")
    print("=============================================================")


if __name__ == "__main__":
    main()
