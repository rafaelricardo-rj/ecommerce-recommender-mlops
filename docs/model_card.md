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

Os números abaixo referem-se às versões atualmente em **Production** no MLflow Registry, obtidos no conjunto de **validação** (20% dos usuários filtrados). Execução em **2026-06-22**, `RANDOM_SEED=42`. Reprodutível via `uv run dvc repro` (mas ver Seção 5.8 sobre não-determinismo).

| Métrica | Linear Regression (baseline) | MLP (PyTorch) | Vencedor |
|---|---|---|---|
| **MSE** ↓ | 0,6039 | **0,5359** | MLP |
| **RMSE** ↓ | 0,7771 | **0,7321** | MLP |
| **MAE** ↓ | 0,2170 | **0,1234** | MLP |
| **R²** ↑ | 0,7469 | **0,7754** | MLP |

(↓ menor é melhor, ↑ maior é melhor.)

### Interpretação honesta

A MLP venceu o baseline em todas as 4 métricas, com ganho expressivo no MAE (43% menor: 0,123 vs 0,217) e R² superior (0,775 vs 0,747). A vantagem em RMSE/MSE é mais modesta (~6%), sugerindo que a MLP captura padrões úteis para predições típicas (refletido no MAE), mas tem pouco a oferecer onde a relação já é aproximadamente linear (RMSE penaliza erros grandes ao quadrado).

Importante: **a magnitude dessa vantagem depende de qual run da MLP é considerado**. Em uma execução anterior do mesmo pipeline, com seed e hyperparâmetros idênticos, a MLP **perdeu** para a regressão linear em MSE/RMSE/R² (ver Seção 5.8). Treinos individuais não são reprodutíveis bit-a-bit, e qualquer afirmação categórica sobre superioridade da MLP precisa de **múltiplas runs com seeds variadas** e teste estatístico — não feito nesta versão.

### Hyperparâmetros usados na MLP (run em Production)
- `hidden_sizes=[64, 32, 16]`, `dropout_rate=0.3`, `learning_rate=1e-3`, `epochs_max=300`, `patience=15`, `min_delta=1e-4`, `lr_scheduler_factor=0.5`, `lr_scheduler_patience=5`
- Otimizador: Adam. Loss: MSELoss. Treinamento encerrado por Early Stopping antes do máximo de épocas.

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
O código atual contém caracteres Unicode em `print()` (ex.: `→` em `src/registry.py:155` e `src/train.py:562`) que causam `UnicodeEncodeError` em terminais Windows com codepage `cp1252` padrão. Workaround: setar `$env:PYTHONIOENCODING="utf-8"` antes de executar.

### 5.8 Não-determinismo do PyTorch
Mesmo com `RANDOM_SEED=42` em `settings.py`, o treinamento da MLP **não é reprodutível bit-a-bit entre execuções**. Em duas execuções consecutivas do mesmo `dvc repro train`, os RMSEs da MLP foram 0,8548 e 0,7321 (variação de ~17%). Causas prováveis:
- A seed configurada via `train_test_split(random_state=...)` afeta apenas o split do sklearn, não o treinamento do PyTorch.
- Falta de `torch.manual_seed(...)`, `torch.cuda.manual_seed_all(...)`, `numpy.random.seed(...)` e `torch.use_deterministic_algorithms(True)`.
- Operações não-determinísticas em cuDNN/MKL não estão desabilitadas.

**Implicação prática**: o critério automático de promoção do Registry (compara RMSE da nova versão com a atual em Production) pode promover ou rebaixar modelos por puro acaso, não por mérito. Recomenda-se fixar todas as seeds e executar múltiplas runs antes de comparar.

### 5.9 Bug no critério de promoção: versões não-arquivadas
Em `src/registry.py:148-154`, `promote_model()` chama `client.transition_model_version_stage(...)` sem o argumento `archive_existing_versions=True`. Resultado: quando uma nova versão é promovida para `Production`, a versão anterior **permanece em `Production`** em paralelo, em vez de ser arquivada. Estado atual observado no Registry:

```
ecommerce-recommender-mlp v1 -> Production   (run antigo)
ecommerce-recommender-mlp v2 -> Production   (run novo, deveria ter arquivado v1)
```

Isso quebra a expectativa de que `Production` referencie um único modelo de cada nome. Fix: passar `archive_existing_versions=True` na chamada acima.

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
Experimento `ecommerce_recommender` no MLflow Tracking (5 runs ao todo, satisfazendo com folga o critério "≥ 3 runs rastreados"):

| Run name | Status | Tipo | RMSE | Observação |
|---|---|---|---|---|
| `linear_regression_v2` (1ª tentativa) | FAILED | LR | 0,7771 | Crash Unicode após log de métricas/artefatos (Seção 5.7) |
| `linear_regression_v2` | FINISHED | LR | 0,7771 | Re-run |
| `MLP_v2` | FINISHED | MLP | 0,8548 | Run inicial; perdia da LR em 3/4 métricas |
| `linear_regression_v2` | FINISHED | LR | 0,7771 | Re-run após edição de `train.py` para anexar Model Card |
| `MLP_v2` | FINISHED | MLP | **0,7321** | Run atual em Production; vence a LR em todas as métricas (não-determinismo: Seção 5.8) |

### Modelos registrados

```
ecommerce-recommender-lr
├── v1  →  Production   (run FAILED; promovida automaticamente por ser a primeira)
├── v2  →  Staging      (RMSE empatado com v1, regra "estritamente menor" impede promoção)
└── v3  →  Staging      (idem)

ecommerce-recommender-mlp
├── v1  →  Production   (run inicial; deveria ter sido arquivada — ver Seção 5.9)
└── v2  →  Production   (versão mais recente, melhor RMSE)
```

> **Observações operacionais:**
> - A v1 do `ecommerce-recommender-lr` em Production aponta para um run com `status=FAILED`. O crash de Unicode ocorreu **após** o registro/promoção, e a lógica de promoção não distingue runs com falha. Recomenda-se filtrar por `status=FINISHED` antes de promover.
> - O Registry da MLP tem **duas versões em Production simultaneamente** devido ao bug descrito na Seção 5.9.

### Critério de promoção
`src/registry.py:register_and_promote()` promove a nova versão para `Production` apenas se o **RMSE for estritamente menor** (`<`) que o RMSE da versão atual em Production (`lower_is_better=True`). Caso contrário, a versão fica em `Staging`. Comparação ocorre por nome de modelo registrado — LR competindo só contra LR, MLP só contra MLP. Como discutido na Seção 5.8, esse critério é frágil em face do não-determinismo do PyTorch.

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
5. **Fixar determinismo do PyTorch** (Seção 5.8): `torch.manual_seed`, `np.random.seed`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
6. **Passar `archive_existing_versions=True`** em `client.transition_model_version_stage(...)` para corrigir o bug da Seção 5.9.
7. **Filtrar runs `status=FAILED`** na lógica de promoção (`src/registry.py`).
8. **Normalizar prints** removendo caracteres Unicode ou definir `PYTHONIOENCODING=utf-8` no `.env` carregado pelo pipeline.
