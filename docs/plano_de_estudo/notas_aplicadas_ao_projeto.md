# Notas Aplicadas ao Projeto — Detecção de Fraudes

> Este documento registra como cada etapa do [fluxo geral de análise de dados](./processo-de-estudo.md)
> foi aplicada especificamente neste projeto. Enquanto o outro arquivo é genérico e reutilizável
> em qualquer projeto futuro, este aqui é o meu diário de bordo: decisões, contexto e raciocínio
> aplicados de fato à detecção de fraudes em transações.

---

## 1. Definição do problema

Antes do código, preciso saber: **o que estou tentando responder ou prever?**

> Neste caso: *"Dado o histórico de transações, consigo identificar quais são fraudulentas?"*

**Métrica que importa:** durante a aula sobre o projeto, analisamos recall e acurácia — mas o
**recall** é o que mais importa aqui, porque **deixar passar uma fraude é pior do que investigar
uma transação normal à toa**.

---

## 2. Coleta dos dados

Neste projeto, os dados foram coletados via URL, usando:

```python
df = pd.read_csv(url)
```

---

## 3. Exploração inicial (EDA)

Conhecer os dados antes de mexer neles. Perguntas guiando essa etapa:

- Quantas linhas/colunas? Quais os tipos de dados?
- Tem valores nulos, duplicados?
- Qual a distribuição da variável alvo (`Class`)?
- Como as variáveis se distribuem? Tem outliers?

**Resultados obtidos:**

| Checagem | Resultado |
|---|---|
| Dimensão do dataset | 284.807 linhas × 31 colunas |
| Tipos de dados | `float64` (30 colunas) + `int64` (`Class`) — todos corretos |
| Valores nulos | Nenhum, em nenhuma coluna |
| Distribuição de `Class` (antes da limpeza) | 99,83% normais / 0,17% fraude |

- A distribuição de `Amount` foi verificada com um **boxenplot** (variação do boxplot, mais
  indicada para datasets grandes) — revelou forte assimetria à direita, com a maioria das
  transações concentrada perto de zero e alguns outliers isolados entre ~12.000 e ~26.000
  (ver [como_ler_graficos.md](./como_ler_graficos.md) para a explicação de como ler esse
  gráfico)

---

## 4. Limpeza dos dados (Data Cleaning)

Com base no que a EDA mostrou, o que fazer:

- Tratar valores nulos (remover, preencher com média/mediana, etc.)
- Remover duplicados
- Corrigir tipos de dados errados
- Tratar outliers — decidir se são erro ou informação real

> ⚠️ **Cuidado específico deste projeto:** em fraude, muitas vezes o outlier **é** a fraude —
> por isso, remover outliers sem pensar pode apagar justamente os casos que o modelo precisa
> aprender a identificar.

**Checklist de limpeza — status final:**

| Checagem | Resultado |
|---|---|
| Nulos | Nenhum encontrado — nenhuma ação necessária |
| Tipos de dados | Todos corretos — nenhuma ação necessária |
| Duplicados | 1.081 linhas encontradas (`df.duplicated().sum()`) e removidas com `drop_duplicates()` |
| Dataset após remoção | 283.726 linhas × 31 colunas |
| Impacto na proporção de fraude | Mínimo — de 0,1727% para 0,1667% (≈19 das 1.081 duplicatas eram fraude) |
| Impacto na distribuição de `Amount` | Nenhum perceptível visualmente (boxenplot antes e depois praticamente idênticos) |
| Outliers em `Amount` | **Mantidos** — decisão consciente, sem remoção |

**Decisão sobre os outliers:** em detecção de fraude, valores atípicos podem representar
justamente o padrão que o modelo precisa aprender a reconhecer, não necessariamente um erro
de coleta. Por isso, nenhuma remoção de outliers foi aplicada nesta etapa.

> ✅ Etapa de limpeza concluída — cada decisão foi verificada com evidência no código, não
> assumida por padrão.

---

## 5. Feature Engineering

Variáveis criadas/transformadas nesta etapa:

| Variável | O que faz |
|---|---|
| `Amount_log` | Log da variável `Amount` — suaviza a assimetria da distribuição (usado na aula) |
| `Amount_scaled` | Padronização de `Amount` |

**Ideias para acrescentar ao projeto:**
- Extrair hora do dia a partir de `Time`
- Criar variáveis de razão/proporção entre colunas existentes

---

## 6. Separação treino/teste

Depois de preparar os dados, separo uma parte para treinar o modelo e outra para testar se
ele realmente aprendeu (e não decorou) — feito na aula com:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)
```

> ⚠️ Sempre **antes** de treinar qualquer modelo, nunca depois — senão o teste fica contaminado.

---

## 7. Balanceamento das classes

Como fraude é rara, o modelo tende a "preguiça": aprender a dizer sempre *"não é fraude"* e já
acertar 99%+. Por isso, entram aqui técnicas como:

- `SMOTE`
- Undersampling
- `class_weight="balanced"`

> ⚠️ Sempre **depois do split**, e aplicadas **só no treino** — nunca no teste, senão é como
> estar "trapaceando" na avaliação.

**Comparação das 4 técnicas (Random Forest, threshold padrão 0,5):**

| Versão | Recall | Precision | F1 |
|---|---|---|---|
| Sem balanceamento | 0,768 | 0,965 | 0,855 |
| SMOTE | 0,782 | 0,902 | 0,838 |
| Undersampling | 0,866 | 0,051 | 0,095 |
| `class_weight="balanced"` | 0,760 | 0,930 | 0,840 |

- **Undersampling foi descartado**: reduziu o treino de 198.608 para apenas 662 linhas,
  descartando >99% dos exemplos normais reais — resultado foi um colapso de precision (5,1%),
  inutilizável na prática.
- **`class_weight="balanced"` não trouxe ganho** em relação a "sem balanceamento" — ficou
  praticamente empatado, sem justificar a complexidade extra.
- **SMOTE e "sem balanceamento" seguiram como candidatos finais**, decididos junto com o
  ajuste de threshold (ver seção 10).

---

## 8. Treinamento do modelo

Algoritmos testados: Regressão Logística, Random Forest, XGBoost.

> Testar mais de um modelo permite comparar qual performa melhor para este problema específico.

---

## 9. Avaliação do modelo

Depois de treinado, testo com dados que o modelo nunca viu e meço o desempenho com:

- `classification_report` (precision, recall, F1)
- Curva ROC / AUC
- Curva Precision-Recall

> Neste projeto, a curva Precision-Recall é **mais informativa que a ROC**, já que as classes
> são bastante desbalanceadas.

---

## 10. Ajuste fino (Threshold e Hiperparâmetros)

**Threshold:** por padrão, o modelo classifica como fraude se a probabilidade > 0.5, mas esse
corte pode mudar para equilibrar melhor precision vs. recall.

```python
threshold = 0.3  # usado na aula
y_pred_custom = (y_probs > threshold).astype(int)
```

**Hiperparâmetros:** uso do `GridSearchCV` para testar várias configurações do modelo e achar
a que performa melhor.

### Threshold variando — Sem balanceamento vs. SMOTE

Testamos os dois melhores candidatos de balanceamento em vários thresholds (0,5 a 0,1):

- **"Sem balanceamento" bate um teto por volta de recall 0,80** — baixar o threshold além de
  0,2 não aumenta o recall de forma relevante, só piora a precision (o modelo não tem mais
  informação nova para dar)
- **SMOTE consegue ultrapassar esse teto**, alcançando recall de até 0,852 (com custo de
  precision mais baixa em thresholds extremos)

### Análise de custo — decisão final

Suposição documentada: **custo de um Falso Negativo (fraude não detectada) = 10× o custo de
um Falso Positivo** (investigar uma transação normal à toa).

```python
CUSTO_FN = 10
CUSTO_FP = 1
custo_total = (fn * CUSTO_FN) + (fp * CUSTO_FP)
```

**Resultado (top 5 menores custos):**

| Versão | Threshold | FN | FP | Custo Total |
|---|---|---|---|---|
| **SMOTE** | **0,3** | 26 | 25 | **285** ✅ |
| SMOTE | 0,4 | 29 | 16 | 306 |
| Sem balanceamento | 0,3 | 30 | 10 | 310 |
| Sem balanceamento | 0,1 | 28 | 33 | 313 |
| Sem balanceamento | 0,4 | 31 | 5 | 315 |

> ✅ **Decisão final do projeto:** SMOTE + threshold 0,3 — menor custo total entre todas as
> configurações testadas, equilibrando a redução de falsos negativos (caros) sem deixar os
> falsos positivos explodirem.

### Ajuste de hiperparâmetros (GridSearchCV)

**Erro metodológico identificado e corrigido:** a primeira tentativa usou
`scoring="recall"` no `GridSearchCV`. Como recall isolado não penaliza falsos positivos, o
grid encontrou hiperparâmetros que maximizavam recall às custas de uma explosão de falsos
positivos (FN: 19, FP: 1.450 — custo total de 1.640, muito pior que o baseline de 285).

**Correção:** criado um `scorer` customizado que otimiza diretamente pelo **custo total**
(a métrica que realmente importa para o projeto), usando `make_scorer` com a mesma fórmula
de custo definida acima.

```python
def custo_scorer(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    custo = (fn * CUSTO_FN) + (fp * CUSTO_FP)
    return -custo  # negativo pois GridSearchCV maximiza o score
```

O SMOTE foi aplicado dentro de um `imblearn.Pipeline`, para ser recalculado separadamente em
cada fold do cross-validation (evita vazamento entre exemplos sintéticos e reais).

**Resultado final:**

| | Custo Total | FN | FP |
|---|---|---|---|
| Baseline (RF padrão, sem grid search) | 285 | 26 | 25 |
| Grid search (`max_depth=None`, `n_estimators=100`) | **285** | 26 | 25 |

> ✅ O `GridSearchCV` confirmou que a configuração já usada (RF padrão) já era ótima (ou
> empatada) entre as combinações testadas — validando formalmente a decisão anterior, em vez
> de apenas assumi-la.

### Boas práticas de engenharia aplicadas nesta etapa

Como o `GridSearchCV` demora ~20 minutos para rodar, duas práticas foram adotadas para evitar
retrabalho desnecessário:

- **Persistência do modelo (`joblib`)**: o `melhor_modelo` (pipeline SMOTE + RF) e o `scaler`
  já treinados são salvos juntos em `models/modelo_fraude_final.pkl`, como um dicionário
  (`{"modelo": ..., "scaler": ...}`). Isso permite que a etapa de interpretação (SHAP) carregue
  o modelo pronto, sem precisar retreinar nada.
- **Células interativas no VS Code (`# %%`)**: o script `deteccao_fraude.py` foi organizado em
  células executáveis individualmente, permitindo rodar o `GridSearchCV` uma única vez e manter
  o resultado (`melhor_modelo`) vivo na sessão, sem reiniciar o script inteiro a cada ajuste.

---

## 11. Interpretação / Explicabilidade

Um modelo "caixa preta" que só cospe previsões não é suficiente em ambientes profissionais —
especialmente em fraude, onde é preciso **justificar por que uma transação foi marcada**.

> É aqui que entra o **SHAP**: mostra quais variáveis mais pesaram em cada decisão do modelo.

**Importância global (bar plot):** `V14`, `V12` e `V4` lideram como as variáveis mais
influentes nas decisões do modelo, seguidas por `V3`, `V10`, `V11`, `V17`, `V7` e `V16`. As
demais 24 variáveis, somadas, ainda representam uma fatia relevante — a decisão do modelo é
distribuída, não concentrada numa única variável.

**Direção do impacto (beeswarm):** valores **baixos** de `V14` e `V12` empurram fortemente a
previsão para fraude; em `V4`, o padrão se inverte (valores altos empurram para fraude).

**Casos individuais (waterfall) — comparando acerto, falso negativo e falso positivo:**

| Caso | V14 | Resultado |
|---|---|---|
| Acerto de fraude | -5,208 | Detectado corretamente (f(x) = 0,83) |
| Falso negativo | -0,932 | Não detectado (f(x) ≈ 0) — valor pouco extremo |
| Falso positivo | -5,554 | Marcado como fraude (f(x) = 0,6) — era transação normal |

**Insight principal:** o modelo depende fortemente de `V14` estar muito negativo para
"confiar" que uma transação é fraude. Isso funciona bem quando a fraude segue esse padrão
extremo (acerto), mas gera dois tipos de erro: fraudes mais "sutis" passam despercebidas
(falso negativo), e transações normais que coincidentemente têm `V14` muito negativo são
marcadas por engano (falso positivo).

> ⚠️ Como `V1`-`V28` são componentes de PCA, a leitura é apenas estatística — sabemos que
> "V14 baixo" está associado a fraude segundo o modelo, mas não sabemos o que V14 representa
> no mundo real.

(Ver [como_ler_graficos.md](./como_ler_graficos.md) para a explicação de como ler os
gráficos de bar plot, beeswarm e waterfall.)

---

## 12. Comunicação dos resultados

Depois de tudo pronto, o último passo é traduzir isso para quem não é técnico — dashboards,
relatórios, apresentações.

> De nada adianta um modelo excelente se ninguém no negócio entende o impacto dele.

---

## Próximos passos / ideias para evoluir o projeto

- [x] Ajuste de hiperparâmetros (GridSearchCV) na configuração final (SMOTE + RF)
- [ ] Expandir análise SHAP com `beeswarm` e `waterfall` para casos individuais
- [ ] Construir dashboard no Power BI com os resultados do modelo
- [ ] *(ideia para projeto futuro separado)* Extrair as partes genéricas deste pipeline
      (comparação de balanceamento, análise de threshold/custo) para um módulo reutilizável,
      aplicável a outros problemas de classificação binária desbalanceada — não só fraude