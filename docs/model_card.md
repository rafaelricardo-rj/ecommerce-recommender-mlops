# Model Card — Ecommerce Recommender

> Documento descritivo dos modelos treinados no pipeline `ecommerce-recommender-mlops`,
> seguindo o formato proposto por Mitchell et al. (2019) — *Model Cards for Model Reporting*.

## 1. Detalhes dos modelos

Cinco modelos são treinados e registrados no MLflow Model Registry como parte do mesmo pipeline (`src/train.py`). O **MLP é o modelo principal** exigido pelo Tech Challenge (rede neural PyTorch); os outros quatro são baselines sklearn de complexidades crescentes para comparação.

### Modelo principal

| Atributo | `ecommerce-recommender-mlp` |
|---|---|
| Tipo | Rede Neural Multilayer Perceptron |
| Framework | PyTorch 2.x |
| Arquitetura | `Linear(2→64) → BatchNorm → ReLU → Dropout(0.3) → Linear(64→32) → ... → Linear(16→1)` |
| Otimizador / Loss | Adam (lr=1e-3), MSELoss |
| Regularização | Dropout 0.3, BatchNorm, Early Stopping (patience=15), LR Scheduler (ReduceLROnPlateau) |
| Estágio atual no Registry | Production (v1) |

### Baselines (Scikit-Learn) registrados

| Nome no Registry | Algoritmo | Configuração | Propósito da comparação |
|---|---|---|---|
| `ecommerce-recommender-dummy` | `DummyRegressor` | `strategy="mean"` | Sanity check — prediz sempre a média. Limite inferior absoluto. |
| `ecommerce-recommender-lr` | `LinearRegression` | OLS analítico | Baseline paramétrico linear. |
| `ecommerce-recommender-knn` | `KNeighborsRegressor` | `n_neighbors=5`, distância Euclidiana | Baseline não-paramétrico (vizinhança local). |
| `ecommerce-recommender-rf` | `RandomForestRegressor` | 100 árvores, `random_state=42`, `n_jobs=1` | Baseline ensemble com decision trees. |

A escolha cobre **quatro abordagens diferentes** (constante / linear paramétrico / não-paramétrico / ensemble de árvores), permitindo isolar onde o ganho de cada classe de modelo vem.

Repositório: <https://github.com/rafaelricardo-rj/ecommerce-recommender-mlops>. Tags de negócio aplicadas a todos os modelos registrados: `team=mlops-fiap`, `domain=e-commerce`, `task=transaction-prediction`, `data_source=retailrocket`.

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

### Escopo realizado vs escopo do problema de negócio
O enunciado descreve um "sistema de recomendação de produtos baseado em comportamento de navegação", o que tipicamente implica **ranking de itens por usuário**. A versão atual entrega um passo anterior: um **estimador de propensão agregada à conversão** (output: valor contínuo por usuário, não lista ranqueada de produtos). Isso é uma simplificação consciente — o problema completo de ranking exige feature engineering por item/sessão (ver limitação 5.2 e próximo passo 1 na Seção 9), trabalho fora do escopo da Etapa 4 do Tech Challenge (Cientista de Dados, foco em treinar e comparar modelos sobre features já existentes).

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

Resultados no conjunto de **validação** (20% dos usuários filtrados), execução em **2026-06-22** com `RANDOM_SEED=42`. Reprodutível na **mesma plataforma** (Windows, Python 3.14, PyTorch 2.x em CPU) com a mesma seed graças a `src/train.py:_set_deterministic()`. Plataformas diferentes (Linux vs Windows, GPU vs CPU, arquiteturas distintas) podem gerar variações numéricas pequenas devido a diferenças de arredondamento em ponto flutuante.

Tabela ordenada **da pior para a melhor RMSE**:

| Modelo | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|---|
| `DummyRegressor` (sanity) | 2,3864 | 1,5448 | 0,2017 | -0,0000 |
| `LinearRegression` | 0,6039 | 0,7771 | 0,2170 | 0,7469 |
| **`MLP` (PyTorch)** | 0,4913 | 0,7009 | 0,1143 | 0,7941 |
| `RandomForestRegressor` | 0,3726 | 0,6104 | **0,1004** | 0,8439 |
| `KNeighborsRegressor` (k=5) | **0,3470** | **0,5891** | 0,1060 | **0,8546** |

(↓ menor é melhor, ↑ maior é melhor. **Negrito**: melhor valor por métrica.)

> **Sobre precisão das diferenças:** as métricas vêm de **uma única seed** (`RANDOM_SEED=42`) e **um único split** treino/validação. Sem k-fold cross-validation, diferenças observadas (por exemplo KNN 0,589 vs RF 0,610) podem refletir variância amostral, não diferenças estruturais entre os modelos. Ver limitação 5.8 para recomendação de validação com múltiplas seeds.

### Interpretação honesta

**Sanity check passa**: todos os modelos preditivos batem fortemente o `DummyRegressor` em MSE/RMSE/R². O sinal nas features está sendo capturado.

**Mas a MLP não é o melhor modelo nessa configuração**: ela ficou em terceiro lugar (RMSE 0,7009), atrás do KNN (0,5891) e do Random Forest (0,6104). O ordenamento final por RMSE é:

```
KNN (0,589) < RF (0,610) < MLP (0,701) < LR (0,777) < Dummy (1,545)
```

A MLP só vence o `LinearRegression` e o `DummyRegressor`. Hipóteses para esse resultado:

1. **Espaço de features muito pequeno (2 dimensões)**: com input bidimensional, baselines não-paramétricos (KNN, RF) capturam a estrutura local do problema de forma mais direta do que uma MLP densa, que precisaria aprender representações abstratas que não trazem ganho real aqui (ver limitação 5.2).
2. **MLP superdimensionada para o problema**: 3 camadas ocultas (64→32→16) sobre 2 features é arquitetura complexa demais por construção — uma decisão de hyperparâmetros que não foi reavaliada após observar que mais profundidade não compra mais expressividade nesta dimensionalidade. Não foi feito ablation comparando 1, 2, 3 camadas. Dropout 0,3 ajuda mas não resolve o desalinhamento entre capacidade e sinal disponível.
3. **Relação não-linear local**: o KNN aproxima localmente sem assumir forma funcional, e o RF particiona o espaço em retângulos — ambas estratégias eficientes quando a superfície de decisão tem padrões não-monotônicos.

**Esse resultado é o ponto principal da comparação**, e não uma falha do exercício: o Tech Challenge pede "MLP funcional + comparação com baselines", e a comparação revelou um insight metodológico relevante (rede neural não é sempre o melhor caminho em problemas de baixa dimensionalidade) — exatamente o tipo de observação que o Model Card existe para documentar. A arquitetura da MLP foi documentada e fixada antes da comparação; o ranking final é consequência honesta da experimentação, não foi escolhido retrospectivamente.

### Detalhes do treino da MLP (v1 em Production)
- Hyperparams: `hidden_sizes=[64, 32, 16]`, `dropout_rate=0.3`, `learning_rate=1e-3`, `epochs_max=300`, `patience=15`, `min_delta=1e-4`, `lr_scheduler_factor=0.5`, `lr_scheduler_patience=5`
- Otimizador: Adam. Loss: MSELoss. Treinamento encerrado por **Early Stopping na época 127** (best `val_loss=0,4598`).

## 5. Limitações

### 5.1 Modelagem como regressão de contagem
O target `transaction` é uma **contagem inteira não-negativa** com distribuição altamente assimétrica (a maioria dos usuários tem 0 transações). Modelar isso como regressão contínua com `MSELoss` é matematicamente subótimo — alternativas mais adequadas seriam regressão Poisson, classificação binária (converter > 0 sim/não), ou regressão zero-inflada.

### 5.2 Sem informação de itens, sessões ou tempo
A feature engineering atual colapsa todo o histórico do usuário em 2 contagens agregadas (`view`, `addtocart`). Toda informação sequencial, temporal e de quais produtos foram interagidos é perdida. O modelo, portanto, **não é um sistema de recomendação no sentido clássico** — não pode rankear produtos para um usuário. É um **estimador de propensão agregada à conversão**.

### 5.3 Correlação temporal entre features e target (covariate shift, não leakage clássico)
As features (`view`, `addtocart`) e o target (`transaction`) são agregações do **mesmo período temporal** — todas contagens do histórico completo do usuário no dataset. **Não é data leakage clássico** (o modelo não está "vendo o futuro" para prever o passado), mas é covariate shift natural: o comportamento agregado de navegação está correlacionado com o agregado de conversão simplesmente porque um usuário ativo tende a estar mais ativo em todos os eventos. Parte do R² alto da regressão linear (0,747) reflete essa correlação estrutural, não um sinal preditivo de propensão.

Para uma avaliação preditiva mais rigorosa, recomenda-se split com **corte temporal**: treinar com features calculadas até instante `t` e avaliar contra transações observadas em janela `t+1` (próximo passo 3 na Seção 9).

### 5.4 Ausência de conjunto de teste real
O pipeline usa apenas split treino/validação (80/20). Não há holdout independente para avaliação final, o que torna as métricas vulneráveis a overfitting de hiperparâmetros caso futuras iterações ajustem o modelo olhando a validação.

### 5.5 Sem validação cruzada
Single split — variância das métricas não é estimada. Pequenas mudanças na seed podem alterar o ranking entre modelos.

### 5.6 Filtragem de usuários pouco ativos
O `preprocess.py` remove usuários com `<3` interações. Isso elimina a maior parte do dataset original e impede que o modelo aprenda padrões de **cold start**, que são justamente o caso mais difícil em recomendação real.

### 5.7 Ambiente Windows
O código atual contém caracteres Unicode em `print()` (ex.: `→` em `src/registry.py` e `src/train.py`) que causam `UnicodeEncodeError` em terminais Windows com codepage `cp1252` padrão. Workaround atual: setar `$env:PYTHONIOENCODING="utf-8"` antes de executar. Fix permanente sugerido na Seção 9.

### 5.8 Validação estatística ausente
A comparação dos 5 modelos (Seção 4) é baseada em **uma única seed** (`RANDOM_SEED=42`) e **um único split** treino/validação. O pipeline é determinístico (mesma seed e plataforma → mesmo resultado), mas isso não substitui validação estatística com múltiplas seeds independentes. Diferenças observadas podem refletir variância amostral. Recomendado: rodar k-fold cross-validation ou múltiplas seeds e reportar média ± desvio para cada métrica.

### 5.9 Early Stopping da MLP usa o conjunto de validação que também serve à comparação
O Early Stopping da MLP (`patience=15`, monitorando `val_loss`) é parametrizado **no mesmo conjunto de validação** usado para reportar as métricas finais da Seção 4. Isso dá à MLP uma forma de **regularização adicional baseada em validação** — ela escolhe quando parar olhando para o conjunto contra o qual será avaliada. Os baselines sklearn (LR, KNN, RF, Dummy) não têm acesso equivalente: eles treinam apenas em `x_train`/`y_train` e são avaliados em `x_val` sem nenhuma decisão de hyperparâmetros ajustada nesse conjunto.

Apesar dessa vantagem metodológica, a MLP ficou em terceiro lugar (atrás de KNN e RF), o que reforça a interpretação da Seção 4 — o ganho não-paramétrico em baixa dimensionalidade supera o ganho que a MLP obtém pelo early stopping. Comparação mais justa exigiria aplicar regularização baseada em validação também aos baselines (ex.: validation curve para escolher `n_neighbors` no KNN, `max_depth`/`n_estimators` no RF), ou usar k-fold cross-validation em ambos os lados.

## 6. Vieses e considerações éticas

### Vieses identificados
- **Viés de sobrevivência / atividade**: ao filtrar usuários com `<3` interações, o modelo é treinado apenas em uma população já engajada. Aplicar a recém-chegados extrapola para fora da distribuição de treino.
- **Viés de desbalanceamento + split não-estratificado**: o target (`transaction`) tem distribuição extremamente assimétrica — a maioria dos usuários tem 0 transações, com cauda longa de poucos compradores frequentes. O `train_test_split` atual usa apenas `random_state=42` sem `stratify=y`, o que pode resultar em treino e validação com proporções de transações diferentes (especialmente para os bins de altíssima conversão). Mitigação sugerida: estratificar por bins (0, 1–2, 3+ transações) ou reportar a distribuição observada em cada split.
- **Viés temporal do dataset**: RetailRocket é de 2015. Comportamento de e-commerce mudou significativamente desde então (mobile-first, social commerce, marketplaces). Modelo treinado neste dataset não reflete dinâmicas atuais.
- **Viés geográfico e setorial**: o dataset cobre **um único e-commerce**, em **uma única região**, em **uma única vertical**. Generalização para outros nichos não é garantida.

### Considerações éticas
- **Privacidade**: o dataset é anonimizado (apenas IDs numéricos para visitor/item), sem PII. Ainda assim, IDs poderiam em tese ser correlacionados com cookies ou sessões em ambiente de produção — não fazer.
- **Risco de manipulação**: como o modelo prevê propensão a transação, usá-lo para priorizar ofertas pode reforçar bolhas de consumo (recomendar mais a quem já compra muito). Mitigação fora do escopo deste modelo.
- **Decisões automatizadas sobre usuários**: o modelo **não deve** ser usado para decisões adversariais (negar produto, limitar acesso) sem revisão humana.

## 7. MLflow Tracking e Registry

### Experimentos rastreados
Experimento `ecommerce_recommender` no MLflow Tracking — **5 runs `FINISHED`** (atende com folga o critério "≥ 3 runs rastreados"):

| Run name | Status | Modelo | RMSE | Artefatos |
|---|---|---|---|---|
| `dummy_regressor_v1` | FINISHED | DummyRegressor | 1,5448 | `model_card.md` |
| `linear_regression_v1` | FINISHED | LinearRegression | 0,7771 | `model_card.md` |
| `knn_v1` | FINISHED | KNeighborsRegressor | 0,5891 | `model_card.md` |
| `random_forest_v1` | FINISHED | RandomForestRegressor | 0,6104 | `model_card.md` |
| `MLP_v2` | FINISHED | MLP (PyTorch) | 0,7009 | `MLP_recommender_model.pth`, `model_card.md` |

> Convenção de versionamento dos run names: os baselines sklearn (incluindo LR) recebem `_v1` por serem a primeira execução do pipeline com os 4 baselines em loop. O sufixo `_v2` da MLP é herdado da execução anterior (commit `5a5b2e1`, primeira versão pós-fix de determinismo) e preservado por compatibilidade com runs históricos do mesmo nome no Tracking.

Todos os runs anexam `docs/model_card.md` como artefato (via `_log_model_card_artifact()` em `src/train.py`), garantindo que o documento descritivo esteja versionado junto com cada execução do pipeline.

### Modelos registrados

```
ecommerce-recommender-dummy  v1  →  Production   (sanity check)
ecommerce-recommender-lr     v1  →  Production   (baseline paramétrico linear)
ecommerce-recommender-knn    v1  →  Production   (baseline não-paramétrico)
ecommerce-recommender-rf     v1  →  Production   (baseline ensemble)
ecommerce-recommender-mlp    v1  →  Production   (modelo principal — rede neural)
```

Cada algoritmo tem sua **própria linhagem de versões** no Registry. A versão v1 de cada um foi promovida automaticamente a Production por ser a primeira (sem nada para comparar).

### Critério de promoção
`src/registry.py:register_and_promote()` promove a nova versão para `Production` apenas se o **RMSE for estritamente menor** (`<`) que o RMSE da versão atual em Production (`lower_is_better=True`). Caso contrário, a versão fica em `Staging`. A comparação ocorre **por nome de modelo registrado** — cada algoritmo concorre apenas consigo mesmo, garantindo que a evolução de cada baseline seja rastreada independentemente. Quando uma promoção ocorre, `archive_existing_versions=True` arquiva automaticamente a versão anterior do mesmo estágio, garantindo que cada estágio referencie no máximo uma versão de cada modelo.

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

1. **Redesenhar a feature engineering** para preservar granularidade de item e tempo — base para um recomendador real (ranking). Com mais features, a MLP pode passar a fazer sentido em vez de perder para KNN/RF.
2. **Mudar o problema para classificação binária** (`converted = transaction > 0`) e usar `BCEWithLogitsLoss`. Isso resolve a Seção 5.1 e permite métricas mais informativas (precision, recall, F1, ROC-AUC).
3. **Corte temporal no split** treino/teste para mitigar o data leakage da Seção 5.3.
4. **Adicionar baselines clássicos de RecSys**: popularity, item-KNN colaborativo, ALS (matrix factorization). Os 4 baselines atuais (Dummy, LR, KNN, RF) são genéricos de regressão — falta comparação com baselines específicos de recomendação.
5. **Validação estatística com múltiplas seeds** (Seção 5.8): rodar o treino com seeds variadas e reportar média ± desvio em vez de uma única run.
6. **Filtrar runs `status=FAILED`** na lógica de promoção (`src/registry.py`) — defesa em profundidade caso runs venham a falhar parcialmente.
7. **Normalizar prints** removendo caracteres Unicode em `src/registry.py`/`src/train.py` ou definir `PYTHONIOENCODING=utf-8` no `.env` carregado pelo pipeline.
8. **Migrar de Stages para Aliases** no Model Registry — Stages (`Staging`/`Production`) estão deprecadas a partir do MLflow 2.9; a API recomendada agora é `set_registered_model_alias()`.
