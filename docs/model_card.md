# Model Card — Ecommerce Recommender

> Documento descritivo dos modelos treinados no pipeline `ecommerce-recommender-mlops`,
> seguindo o formato proposto por Mitchell et al. (2019) — *Model Cards for Model Reporting*.

## 1. Detalhes do modelo

Dois modelos são treinados e registrados no MLflow Model Registry como parte do mesmo pipeline:

| Atributo | `ecommerce-recommender-mlp` | `ecommerce-recommender-lr` |
|---|---|---|
| Tipo | Rede Neural Multilayer Perceptron | Regressão Linear (baseline) |
| Framework | PyTorch 2.x | scikit-learn 1.x |
| Arquitetura | `Linear(2→64) → BatchNorm → ReLU → Dropout(0.3) → Linear(64→32) → ... → Linear(16→1)` | `LinearRegression(fit_intercept=True)` |
| Otimizador / Loss | Adam (lr=1e-3), MSELoss | OLS analítico |
| Regularização | Dropout 0.3, BatchNorm, Early Stopping (patience=15), LR Scheduler (ReduceLROnPlateau) | — |
| Estágio atual no Registry | Production (v1) | Production (v1) |
| Repositório | <https://github.com/rafaelricardo-rj/ecommerce-recommender-mlops> | idem |
| Autores | Equipe Tech Challenge Fase 02 — POS TECH | idem |

Tags de negócio aplicadas a ambos: `team=mlops-fiap`, `domain=e-commerce`, `task=transaction-prediction`, `data_source=retailrocket`.

## 2. Uso pretendido

### Caso de uso principal
Estimar o **número esperado de transações** por usuário a partir do seu comportamento de navegação agregado (visualizações e adições ao carrinho). A saída é um valor contínuo que pode ser usado como **score de propensão de conversão**.

### Casos de uso fora do escopo
- **Ranking de itens** — o modelo não recomenda produtos específicos; recebe agregados por usuário e produz um único score, não uma lista ranqueada.
- **Cold start** — usuários com menos de 3 interações totais são removidos no pré-processamento (`src/preprocess.py`), portanto o modelo não generaliza para visitantes novos.
- **Recomendação personalizada baseada em conteúdo** — nenhuma feature de item (categoria, preço, propriedades) é utilizada na configuração atual.
- **Decisões críticas** — não usar para classificação de risco, crédito, ou qualquer decisão com impacto financeiro direto sobre o usuário.

### Usuários previstos
Cientistas de dados e engenheiros de ML do time, no contexto do Tech Challenge Fase 02. Não recomendado para produção sem revisão das limitações da Seção 5.

## 3. Dados de treino e avaliação

### Fonte
[RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — interações reais coletadas em um e-commerce real durante 4,5 meses (2015). Quatro arquivos: `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv`. Apenas `events.csv` é utilizado nesta versão.

### Estatísticas do dataset bruto
- ~2,75M eventos
- ~1,4M visitantes únicos
- ~235K itens únicos
- Eventos: `view`, `addtocart`, `transaction`

### Pipeline de preparação (versionado via DVC)
1. **`preprocess`** (`src/preprocess.py`) — filtra usuários com **≥ 3 interações** (qualquer evento). Salva em `data/processed/events_clean.csv`.
2. **`feature_eng`** (`src/feature_eng.py`) — agrega por `visitorid` a contagem de cada tipo de evento e o total de interações. Salva em `data/features/user_features.csv`.
3. **`train`** (`src/train.py`) — aplica `StandardScaler`, faz split 80/20 (estratificação ausente — ver limitação 5.3) com `random_state=42`, treina baseline e MLP.

### Features e target
| Coluna | Tipo | Descrição |
|---|---|---|
| `view` | int | Quantas visualizações o usuário fez |
| `addtocart` | int | Quantas adições ao carrinho o usuário fez |
| `transaction` (**target**) | int | Quantas transações o usuário concluiu |

## 4. Métricas de performance

Os números abaixo referem-se às versões atualmente em **Production** no MLflow Registry, obtidos no conjunto de **validação** (20% dos usuários filtrados). Execução em **2026-06-22**, `RANDOM_SEED=42`. **Reprodutível bit-a-bit** via `uv run dvc repro` graças ao fix de determinismo aplicado em `src/train.py:_set_deterministic()` (cobre Python `random`, NumPy, PyTorch CPU/CUDA, cuDNN).

| Métrica | Linear Regression (baseline) | MLP (PyTorch) | Vencedor |
|---|---|---|---|
| **MSE** ↓ | 0,6039 | **0,4913** | MLP |
| **RMSE** ↓ | 0,7771 | **0,7009** | MLP |
| **MAE** ↓ | 0,2170 | **0,1143** | MLP |
| **R²** ↑ | 0,7469 | **0,7941** | MLP |

(↓ menor é melhor, ↑ maior é melhor.)

### Interpretação

A MLP venceu o baseline em todas as 4 métricas. O ganho mais expressivo está no **MAE** (47% menor: 0,114 vs 0,217), indicando que a rede produz predições mais alinhadas com a mediana do erro. Em RMSE/MSE o ganho é menor (~10% e ~19%, respectivamente), porque essas métricas penalizam fortemente outliers — e o target (`transaction` count) tem cauda longa com poucos usuários de altíssima conversão que ambos os modelos erram.

O ganho em R² (0,794 vs 0,747) é moderado e mostra que a MLP captura ~4,7 pontos percentuais a mais da variância do que a regressão linear.

Apesar do resultado favorável à MLP nessa configuração, **não interpretar como evidência de superioridade arquitetural geral**. Com apenas 2 features de entrada (`view`, `addtocart`) a relação com o target é aproximadamente linear, e parte do ganho da MLP pode ser devida ao Early Stopping baseado em validação atuando como regularização forte. Comparação mais rigorosa exigiria múltiplas seeds independentes e teste estatístico — não feito nesta versão (ver Seção 9).

### Hyperparâmetros usados na MLP (v1 em Production)
- `hidden_sizes=[64, 32, 16]`, `dropout_rate=0.3`, `learning_rate=1e-3`, `epochs_max=300`, `patience=15`, `min_delta=1e-4`, `lr_scheduler_factor=0.5`, `lr_scheduler_patience=5`
- Otimizador: Adam. Loss: MSELoss. Treinamento encerrado por **Early Stopping na época 127** (best `val_loss=0,4598`).

## 5. Limitações

### 5.1 Modelagem como regressão de contagem
O target `transaction` é uma **contagem inteira não-negativa** com distribuição altamente assimétrica (a maioria dos usuários tem 0 transações). Modelar isso como regressão contínua com `MSELoss` é matematicamente subótimo — alternativas mais adequadas seriam regressão Poisson, classificação binária (converter > 0 sim/não), ou regressão zero-inflada.

### 5.2 Sem informação de itens, sessões ou tempo
A feature engineering atual colapsa todo o histórico do usuário em 2 contagens agregadas (`view`, `addtocart`). Toda informação sequencial, temporal e de quais produtos foram interagidos é perdida. O modelo, portanto, **não é um sistema de recomendação no sentido clássico** — não pode rankear produtos para um usuário. É um **estimador de propensão agregada à conversão**.

### 5.3 Data leakage potencial entre features e target
As features (`view`, `addtocart`) e o target (`transaction`) são contagens calculadas **na mesma janela temporal**. Um usuário que já realizou várias transações no histórico provavelmente também tem muitas views e addtocarts no mesmo intervalo. Parte do desempenho da regressão linear (R² 0,75) pode refletir essa correlação espúria, não um sinal preditivo genuíno. Recomenda-se redesenhar o split com **corte temporal** (treinar em janela `t`, avaliar em janela `t+1`).

### 5.4 Ausência de conjunto de teste real
O pipeline usa apenas split treino/validação (80/20). Não há holdout independente para avaliação final, o que torna as métricas vulneráveis a overfitting de hiperparâmetros caso futuras iterações ajustem o modelo olhando a validação.

### 5.5 Sem validação cruzada
Single split — variância das métricas não é estimada. Pequenas mudanças na seed podem alterar o ranking entre modelos.

### 5.6 Filtragem de usuários pouco ativos
O `preprocess.py` remove usuários com `<3` interações. Isso elimina a maior parte do dataset original e impede que o modelo aprenda padrões de **cold start**, que são justamente o caso mais difícil em recomendação real.

### 5.7 Ambiente Windows
O código atual contém caracteres Unicode em `print()` (ex.: `→` em `src/registry.py` e `src/train.py`) que causam `UnicodeEncodeError` em terminais Windows com codepage `cp1252` padrão. Workaround atual: setar `$env:PYTHONIOENCODING="utf-8"` antes de executar. Fix permanente sugerido na Seção 9.

### 5.8 Validação estatística ausente
A comparação MLP vs LR (Seção 4) é baseada em **uma única seed** (`RANDOM_SEED=42`). Embora o pipeline agora seja determinístico (mesma seed → mesmo resultado bit-a-bit), isso não substitui validação estatística com múltiplas seeds independentes. Diferenças observadas podem refletir uma seed afortunada para um dos modelos. Recomendado: rodar k-fold cross-validation ou múltiplas seeds e reportar média ± desvio.

## 6. Vieses e considerações éticas

### Vieses identificados
- **Viés de sobrevivência / atividade**: ao filtrar usuários com `<3` interações, o modelo é treinado apenas em uma população já engajada. Aplicar a recém-chegados extrapola para fora da distribuição de treino.
- **Viés temporal do dataset**: RetailRocket é de 2015. Comportamento de e-commerce mudou significativamente desde então (mobile-first, social commerce, marketplaces). Modelo treinado neste dataset não reflete dinâmicas atuais.
- **Viés geográfico e setorial**: o dataset cobre **um único e-commerce**, em **uma única região**, em **uma única vertical**. Generalização para outros nichos não é garantida.

### Considerações éticas
- **Privacidade**: o dataset é anonimizado (apenas IDs numéricos para visitor/item), sem PII. Ainda assim, IDs poderiam em tese ser correlacionados com cookies ou sessões em ambiente de produção — não fazer.
- **Risco de manipulação**: como o modelo prevê propensão a transação, usá-lo para priorizar ofertas pode reforçar bolhas de consumo (recomendar mais a quem já compra muito). Mitigação fora do escopo deste modelo.
- **Decisões automatizadas sobre usuários**: o modelo **não deve** ser usado para decisões adversariais (negar produto, limitar acesso) sem revisão humana.

## 7. MLflow Tracking e Registry

### Experimentos rastreados
Experimento `ecommerce_recommender` no MLflow Tracking (4 runs `FINISHED`, satisfazendo o critério "≥ 3 runs rastreados"):

| Run name | Status | Tipo | RMSE | Artefatos |
|---|---|---|---|---|
| `linear_regression_v2` (run 1) | FINISHED | LR | 0,7771 | `model_card.md` |
| `MLP_v2` (run 1) | FINISHED | MLP | 0,7009 | `MLP_recommender_model.pth`, `model_card.md` |
| `linear_regression_v2` (run 2) | FINISHED | LR | 0,7771 | `model_card.md` |
| `MLP_v2` (run 2) | FINISHED | MLP | 0,7009 | `MLP_recommender_model.pth`, `model_card.md` |

Runs com mesmo nome têm métricas idênticas — a função `_set_deterministic()` em `src/train.py` garante reprodutibilidade bit-a-bit. O Model Card vai anexado a cada run como artefato versionado.

### Modelos registrados

```
ecommerce-recommender-lr
├── v1  →  Production   (primeira versão, promovida automaticamente)
└── v2  →  Staging      (RMSE idêntico a v1, regra "estritamente menor" não promove)

ecommerce-recommender-mlp
├── v1  →  Production   (primeira versão, promovida automaticamente)
└── v2  →  Staging      (RMSE idêntico a v1, regra "estritamente menor" não promove)
```

### Critério de promoção
`src/registry.py:register_and_promote()` promove a nova versão para `Production` apenas se o **RMSE for estritamente menor** (`<`) que o RMSE da versão atual em Production (`lower_is_better=True`). Caso contrário, a versão fica em `Staging`. Comparação ocorre por nome de modelo registrado — LR competindo só contra LR, MLP só contra MLP. Quando uma promoção ocorre, `archive_existing_versions=True` arquiva automaticamente a versão anterior do mesmo estágio, garantindo que cada estágio referencie no máximo uma versão de cada modelo.

## 8. Como reproduzir

```powershell
# 1. Pré-requisitos: Python 3.13+, uv, DVC instalado
# 2. Copiar .env.example → .env
Copy-Item .env.example .env

# 3. Instalar deps
uv sync

# 4. Baixar dataset RetailRocket do Kaggle e extrair em data/raw/
#    Esperado: events.csv, item_properties_part1.csv, item_properties_part2.csv, category_tree.csv

# 5. Executar pipeline completo (encoding UTF-8 obrigatório no Windows)
$env:PYTHONIOENCODING = "utf-8"
uv run dvc repro

# 6. Abrir UI do MLflow para inspecionar runs e registry
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 9. Próximos passos recomendados

Em ordem de impacto esperado:

1. **Redesenhar a feature engineering** para preservar granularidade de item e tempo — base para um recomendador real (ranking).
2. **Mudar o problema para classificação binária** (`converted = transaction > 0`) e usar `BCEWithLogitsLoss`. Isso resolve a Seção 5.1 e permite métricas mais informativas (precision, recall, F1, ROC-AUC).
3. **Corte temporal no split** treino/teste para mitigar o data leakage da Seção 5.3.
4. **Adicionar baselines clássicos de RecSys**: popularity, item-KNN, ALS (matrix factorization). Comparação mais justa para um "sistema de recomendação".
5. **Validação estatística com múltiplas seeds** (Seção 5.8): rodar o treino com seeds variadas e reportar média ± desvio em vez de uma única run.
6. **Filtrar runs `status=FAILED`** na lógica de promoção (`src/registry.py`) — defesa em profundidade caso runs venham a falhar parcialmente.
7. **Normalizar prints** removendo caracteres Unicode em `src/registry.py`/`src/train.py` ou definir `PYTHONIOENCODING=utf-8` no `.env` carregado pelo pipeline.
8. **Migrar de Stages para Aliases** no Model Registry — Stages (`Staging`/`Production`) estão deprecadas a partir do MLflow 2.9; a API recomendada agora é `set_registered_model_alias()`.
