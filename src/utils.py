import pandas as pd
from pathlib import Path


def load_data(filepath: Path) -> pd.DataFrame:
    """Carrega os dados de um arquivo CSV e retorna um DataFrame."""
    return pd.read_csv(filepath)
