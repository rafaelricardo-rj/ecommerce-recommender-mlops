# ecommerce-recommender-mlops

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/DVC-945DD6?logo=dvc&logoColor=white" alt="DVC">
  <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintained">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License MIT">
</p>

## 📌 Visão Geral

Pipeline MLOps end-to-end para recomendação de produtos em e-commerce. Integra **DVC** para versionamento de dados, **MLflow** para tracking de experimentos e Model Registry, e **PyTorch** para o modelo principal (MLP). O pipeline treina e compara 5 modelos (1 MLP + 4 baselines Scikit-Learn) avaliados em 4 métricas.

## 🛠 Tech Stack

| Categoria | Tecnologia |
|---|---|
| **Linguagem** | Python 3.14+ |
| **Deep Learning** | PyTorch (MLP com Early Stopping) |
| **ML Baselines** | Scikit-Learn (Dummy, LR, KNN, RF) |
| **Gerenciamento de Deps** | uv + `pyproject.toml` |
| **Versionamento de Dados** | DVC (Data Version Control) |
| **Experiment Tracking** | MLflow (Tracking + Model Registry) |
| **Configuração** | Pydantic Settings (`.env`) |
| **Qualidade de Código** | Ruff + pre-commit |
| **Containerização** | Docker (multi-stage) |

## 📁 Estrutura do Projeto

```
ecommerce-recommender-mlops/
├── configs/                  # Arquivos de configuração
├── data/
│   ├── raw/                  # Dataset bruto (RetailRocket, versionado via DVC)
│   ├── processed/            # Dados limpos (gerado pelo pipeline)
│   └── features/             # Features agregadas (gerado pelo pipeline)
├── docs/
│   └── model_card.md         # Model Card detalhado
├── models/                   # Modelos treinados (versionados via DVC)
├── scripts/
│   └── validate_env.py       # Validação de variáveis de ambiente
├── src/
│   ├── __init__.py
│   ├── feature_eng.py        # Feature engineering (Stage 2)
│   ├── models.py             # Definição dos modelos (Factory Pattern)
│   ├── preprocess.py         # Pré-processamento (Stage 1)
│   ├── registry.py           # MLflow Model Registry
│   ├── settings.py           # Configuração com Pydantic Settings
│   ├── train.py              # Treinamento e avaliação (Stage 3)
│   └── utils.py              # Utilitários compartilhados
├── tests/
│   ├── test_metrics.py       # Testes de métricas
│   ├── test_models.py        # Testes de modelos
│   └── test_registry.py      # Testes do registry
├── .env                      # Variáveis de ambiente (não versionado)
├── .env.example              # Template das variáveis de ambiente
├── .pre-commit-config.yaml   # Hooks de pre-commit (Ruff)
├── dvc.yaml                  # Pipeline DVC (3 stages)
├── dvc.lock                  # Lock do pipeline (hashes dos dados)
└── pyproject.toml            # Dependências (prod + dev separadas)
```

## 🚀 Getting Started

### Pré-requisitos

- **Python 3.14+** — [Download](https://www.python.org/downloads/)
- **uv** — [Instalação](https://github.com/astral-sh/uv): `pip install uv`

### 1. Clonar o repositório

```bash
git clone https://github.com/rafaelricardo-rj/ecommerce-recommender-mlops.git
cd ecommerce-recommender-mlops
git checkout daniel
```

### 2. Instalar dependências

```bash
uv sync
```

### 3. Configurar variáveis de ambiente

```bash
# Copiar o template (se o .env não existir)
cp .env.example .env
```

### 4. Validar o ambiente

```bash
uv run python scripts/validate_env.py
```

### 5. Baixar dados e modelos (DVC)

Os dados (~1 GB) e modelos treinados são versionados com **DVC** e armazenados no [DagsHub](https://dagshub.com/danielbispo3015/ecommerce-recomender-mlops).

```bash
# Configurar credenciais de leitura (token read-only)
uv run dvc remote modify storage --local auth basic
uv run dvc remote modify storage --local user danielbispo3015
uv run dvc remote modify storage --local password 7854e6af9d7b3cff41b29f5aa897e369b5c00d0d

# Baixar dados e modelos
uv run dvc pull
```

> **Nota:** O token acima é **somente leitura** (read-only). As credenciais são armazenadas em `.dvc/config.local`, que **não é versionado** pelo Git.

### 6. Executar o pipeline completo

```bash
# No Windows, definir encoding UTF-8 (necessário para caracteres especiais nos logs)
# PowerShell:
$env:PYTHONIOENCODING = "utf-8"

# Executar os 3 stages do pipeline DVC
uv run dvc repro
```

### 7. Rodar os testes

```bash
uv run pytest
```

## 🔄 Pipeline DVC

O pipeline é definido no [`dvc.yaml`](dvc.yaml) com 3 stages reprodutíveis:

```
preprocess → feature_eng → train
```

| Stage | Script | Entrada | Saída |
|---|---|---|---|
| `preprocess` | `src/preprocess.py` | `data/raw/events.csv` | `data/processed/events_clean.csv` |
| `feature_eng` | `src/feature_eng.py` | `data/processed/events_clean.csv` | `data/features/user_features.csv` |
| `train` | `src/train.py` | `data/features/user_features.csv` | `models/*.joblib`, `models/*.pth` |

**Comandos úteis:**

```bash
uv run dvc repro          # Executar pipeline completo (pula stages sem mudanças)
uv run dvc status         # Verificar se algum stage precisa ser re-executado
uv run dvc dag            # Visualizar o DAG do pipeline
```

## 📊 Dados

**Fonte:** [RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — ~2,75M interações reais de um e-commerce (2015).

| Atributo | Valor |
|---|---|
| Eventos totais | ~2,75M |
| Visitantes únicos | ~1,4M |
| Itens únicos | ~235K |
| Tipos de evento | `view`, `addtocart`, `transaction` |
| Filtro aplicado | Usuários com ≥ 3 interações |

**Features geradas** (por usuário):

| Feature | Tipo | Descrição |
|---|---|---|
| `view` | int | Contagem de visualizações |
| `addtocart` | int | Contagem de adições ao carrinho |
| `transaction` | int | **Target** — contagem de transações |

## 🤖 Modelos e Resultados

5 modelos treinados e comparados no conjunto de validação (20%, `random_state=42`):

| Modelo | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|---|
| `DummyRegressor` (sanity) | 2,3864 | 1,5448 | 0,2017 | -0,0000 |
| `LinearRegression` | 0,6039 | 0,7771 | 0,2170 | 0,7469 |
| **`MLP` (PyTorch)** | 0,4913 | 0,7009 | 0,1143 | 0,7941 |
| `RandomForestRegressor` | 0,3726 | 0,6104 | 0,1004 | 0,8439 |
| `KNeighborsRegressor` | 0,3470 | 0,5891 | 0,1060 | 0,8546 |

> ↓ menor é melhor · ↑ maior é melhor

Para detalhes completos sobre arquitetura, limitações, vieses e próximos passos, consulte o [Model Card](docs/model_card.md).

## 📈 MLflow Tracking

O pipeline loga automaticamente parâmetros, métricas e artefatos no MLflow.

```bash
# Iniciar o MLflow UI (após executar o pipeline)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# Acessar: http://localhost:5000
```

**Modelos registrados no Model Registry:**

| Nome no Registry | Algoritmo | Stage |
|---|---|---|
| `ecommerce-recommender-dummy` | DummyRegressor | Production |
| `ecommerce-recommender-lr` | LinearRegression | Production |
| `ecommerce-recommender-knn` | KNeighborsRegressor | Production |
| `ecommerce-recommender-rf` | RandomForestRegressor | Production |
| `ecommerce-recommender-mlp` | MLP (PyTorch) | Production |

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
