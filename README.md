# Detecção de Fraudes em Transações Bancárias

Projeto de análise de dados e machine learning para identificar transações fraudulentas em
cartão de crédito, construído a partir de um projeto do bootcamp da Accenture - Python para Análise e
Automação de Dados (DIO) e aprofundado como peça de portfólio.

## 🎯 Sobre o projeto

Bancos e operadoras de cartão processam milhões de transações por dia, e apenas uma fração
mínima delas é fraudulenta — no dataset usado aqui, **apenas 0,17%** das transações são
fraude. Esse desbalanceamento extremo é o principal desafio técnico do projeto: um modelo
"ingênuo" pode ter 99,8% de acurácia simplesmente dizendo que nenhuma transação é fraude, e
ainda assim ser completamente inútil na prática.

**Pergunta de negócio:** dado o histórico de transações, é possível identificar quais são
fraudulentas, minimizando o prejuízo financeiro de fraudes não detectadas sem gerar excesso
de alarmes falsos?

**Métrica de sucesso:** o projeto prioriza **recall** (capturar o máximo de fraudes reais
possível), pois o custo de deixar passar uma fraude é consideravelmente maior do que o custo
de investigar uma transação normal à toa — essa suposição foi formalizada com uma análise de
custo (falso negativo = 10x o custo de um falso positivo), guiando as decisões técnicas de
balanceamento e ponto de corte do modelo.

## 📊 Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284.807 transações de cartão de crédito, das quais 492 são fraudulentas. As variáveis
`V1` a `V28` já vêm transformadas por PCA (por questões de privacidade dos dados originais),
enquanto `Time` e `Amount` são as únicas variáveis originais interpretáveis.

## 🛠️ Tecnologias e o papel de cada uma

**Linguagem:** Python 3.13

### Manipulação e análise de dados

| Framework | Para que serve, em geral | Como foi usado neste projeto |
|---|---|---|
| **pandas** | Biblioteca para manipular dados em formato de tabela (DataFrames) — leitura, filtragem, agregação, limpeza | Leitura do CSV, checagem de nulos/duplicados, criação de novas colunas (feature engineering), organização dos resultados em tabelas comparativas |
| **numpy** | Biblioteca de computação numérica, base matemática para quase todo o ecossistema de dados em Python | Transformação logarítmica de `Amount` (`np.log1p`), para suavizar a assimetria da distribuição |

### Visualização

| Framework | Para que serve, em geral | Como foi usado neste projeto |
|---|---|---|
| **matplotlib** | Biblioteca de visualização de baixo nível — a base sobre a qual outras bibliotecas de gráficos são construídas | Controle de título, exibição e formatação geral dos gráficos gerados |
| **seaborn** | Biblioteca de visualização construída sobre o matplotlib, com gráficos estatísticos prontos e visual mais elaborado por padrão | Geração do boxenplot de `Amount`, para investigar outliers na distribuição das transações |

### Machine Learning

| Framework | Para que serve, em geral | Como foi usado neste projeto |
|---|---|---|
| **scikit-learn** | Principal biblioteca de machine learning "clássico" em Python — modelos, pré-processamento, validação e métricas | Split treino/teste (`train_test_split`), padronização (`StandardScaler`), modelo (`RandomForestClassifier`), ajuste de hiperparâmetros (`GridSearchCV`), métricas de avaliação (recall, precision, F1, matriz de confusão) |
| **imbalanced-learn (imblearn)** | Extensão do scikit-learn especializada em lidar com datasets desbalanceados | Técnicas de balanceamento (`SMOTE`, `RandomUnderSampler`) e `Pipeline` que aplica o balanceamento corretamente dentro de cada fold da validação cruzada, evitando vazamento de dados |

### Interpretabilidade

| Framework | Para que serve, em geral | Como foi usado neste projeto |
|---|---|---|
| **SHAP** | Biblioteca de explicabilidade de modelos de machine learning, baseada em teoria dos jogos — explica o quanto cada variável contribuiu para uma previsão | Geração de três tipos de gráfico: importância global das variáveis (bar plot), direção do impacto de cada variável (beeswarm), e explicação de casos individuais — incluindo diagnóstico de erros do modelo (falso positivo e falso negativo) |

### Persistência

| Framework | Para que serve, em geral | Como foi usado neste projeto |
|---|---|---|
| **joblib** | Biblioteca para salvar e carregar objetos Python complexos em disco (como modelos treinados), de forma eficiente | Salvar o modelo final treinado (junto com o `scaler` usado) em um arquivo `.pkl`, evitando precisar retreinar o modelo (~20 minutos via GridSearchCV) toda vez que uma nova etapa do projeto precisa dele |

## 🔄 Pipeline do projeto

O projeto segue um fluxo estruturado de análise de dados, documentado em detalhe em
[`docs/plano_de_estudo/`](./docs/plano_de_estudo/):

1. **Definição do problema** — pergunta de negócio e métrica prioritária (recall)
2. **Coleta dos dados** — leitura do CSV público
3. **Exploração inicial (EDA)** — estrutura dos dados, distribuição da variável alvo, outliers
4. **Limpeza dos dados** — remoção de 1.081 linhas duplicadas, verificação de nulos e tipos
5. **Feature engineering** — `Amount_log` (suaviza assimetria) e `Hour` (hora do dia extraída de `Time`)
6. **Split treino/teste** — separação estratificada (70/30), preservando a proporção de fraude
7. **Balanceamento das classes** — comparação formal entre SMOTE, undersampling e `class_weight="balanced"`
8. **Treinamento dos modelos** — Random Forest, comparado em cada versão dos dados
9. **Avaliação** — recall, precision, F1 e matriz de confusão
10. **Ajuste fino** — threshold de decisão e hiperparâmetros, ambos otimizados por uma análise de custo (não apenas por recall isolado)
11. **Interpretação (SHAP)** — explicação global e de casos individuais
12. **Comunicação dos resultados** — *(em andamento — dashboard Power BI)*

## 🏆 Resultados principais

- **Configuração final do modelo:** SMOTE (balanceamento) + Random Forest + threshold de
  decisão em 0,3
- **Desempenho no conjunto de teste:** recall de 82% na classe fraude, com 26 falsos
  negativos e 25 falsos positivos em 85.118 transações testadas
- **Validação formal:** a configuração foi confirmada como ótima (ou empatada) por um
  `GridSearchCV` com scorer customizado, otimizando diretamente pelo custo de negócio
  (falso negativo = 10x o custo de um falso positivo) em vez de uma métrica isolada
- **Maior limitação identificada (via SHAP):** o modelo depende fortemente da variável `V14`
  estar em valores muito negativos para identificar fraude com confiança — o que causa tanto
  falsos negativos (fraudes "sutis", com `V14` pouco extremo) quanto falsos positivos
  (transações normais que coincidentemente têm `V14` muito negativo)

## 📁 Estrutura do repositório

```
├── src/
│   ├── deteccao_fraude.py         # pipeline completo: EDA até o modelo final salvo
│   └── interpretacao_shap.py      # carrega o modelo salvo e gera as explicações SHAP
├── notebooks/
│   └── aula_deteccao_fraude.py    # código original da aula (DIO), mantido como referência
├── models/
│   └── modelo_fraude_final.pkl    # modelo treinado + scaler, prontos para uso
├── outputs/
│   └── figures/                   # gráficos gerados pelo projeto
├── docs/
│   └── plano_de_estudo/
│       ├── processo-de-estudo.md         # fluxo geral de análise de dados (genérico)
│       ├── notas_aplicadas_ao_projeto.md # decisões e resultados específicos deste projeto
│       ├── como_ler_graficos.md          # guia de leitura de cada gráfico usado
│       └── imagens/
└── README.md
```

## ▶️ Como reproduzir

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap joblib

python src/deteccao_fraude.py       # roda o pipeline completo (~20 min, por causa do GridSearchCV)
python src/interpretacao_shap.py    # gera as explicações SHAP a partir do modelo já salvo
```

## 📚 Principais aprendizados

- Por que **recall** é a métrica certa para priorizar em problemas de fraude, e como formalizar
  esse critério com uma análise de custo, em vez de uma escolha só qualitativa
- A diferença prática entre balancear os **dados** (SMOTE, undersampling) e balancear o
  **algoritmo** (`class_weight`), e por que nem sempre a técnica mais "popular" é a melhor
  para um dataset específico
- Como evitar vazamento de dados (data leakage) em cada etapa sensível do pipeline: split antes
  de padronizar, balanceamento só no treino, SMOTE recalculado por fold dentro do
  `GridSearchCV`
- Como usar SHAP não só para "confirmar que o modelo funciona", mas para **diagnosticar
  especificamente por que ele erra** em casos individuais

---

*Projeto desenvolvido por Danielly Toledo, a partir do bootcamp de Accenture - Python para Análise e
Automação de Dados da [DIO](https://www.dio.me/).*