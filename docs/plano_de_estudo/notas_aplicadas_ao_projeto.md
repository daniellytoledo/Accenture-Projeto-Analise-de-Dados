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

---

## 11. Interpretação / Explicabilidade

Um modelo "caixa preta" que só cospe previsões não é suficiente em ambientes profissionais —
especialmente em fraude, onde é preciso **justificar por que uma transação foi marcada**.

> É aqui que entra o **SHAP**: mostra quais variáveis mais pesaram em cada decisão do modelo.

---

## 12. Comunicação dos resultados

Depois de tudo pronto, o último passo é traduzir isso para quem não é técnico — dashboards,
relatórios, apresentações.

> De nada adianta um modelo excelente se ninguém no negócio entende o impacto dele.

---

## Próximos passos / ideias para evoluir o projeto

- [ ] Comparar formalmente modelo sem balanceamento vs. `class_weight` vs. SMOTE
- [ ] Adicionar matriz de custo (falso positivo vs. falso negativo)
- [ ] Otimizar threshold com base em custo, não em valor arbitrário
- [ ] Expandir análise SHAP com `beeswarm` e `waterfall` para casos individuais
- [ ] Construir dashboard no Power BI com os resultados do modelo