"""
Projeto: Detecção de Fraudes em Transações
Autor: Danielly Toledo
Descrição: Pipeline de análise de dados e machine learning para identificar
transações fraudulentas, construído seguindo o fluxo documentado em
docs/plano_de_estudo/processo_de_estudo.md
"""

"""
1. DEFINIÇÃO DO PROBLEMA:
    Pergunta: dado o histórico de transações, conseguimos identificar quais são fraudulentas?
    Métrica prioritária: RECALL- deixar passar uma fraude é pior do que investigar uma transação normal à toa (detalhes em notas_aplicadas_ao_projeto.md)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
2. COLETA DOS DADOS
"""

print("\n=== Coleta dos dados / df.head() ===")

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print(df.head())

"""
3. EXPLORAÇÃO INICIAL (EDA)
"""

print("\n=== 3. Exploração Inicial ===")

# Estrutura geral do dataset
print(df.shape) 
print(df.info()) 

# df.shape  → (284807 transações, 31 colunas)
# df.info() → Non-Null Count: 284807 non-null em todas as colunas, ou seja, nenhuma coluna tem valores nulos
# dtypes    → float64(30), int64(1)
# Class     → 0: 0.998273 / 1: 0.001727 (99.83% transações normais, 0.17% são fraudes)

# Distribuição da variável alvo - revela o problema central do projeto
print(df["Class"].value_counts(normalize=True))

# Visualização da distribuição de Amount — checagem exploratória de outliers
sns.boxenplot(x=df["Amount"])
plt.title("Distribuição de Amount — verificação de outliers")
plt.show()

"""
4. LIMPEZA DOS DADOS (Data Cleaning)
"""

print("\n=== 4. Limpeza ===")

# Nulos já verificamos na EDA que nenhuma coluna possui valores ausentes
# Tipos de dados já verificamos na EDA que todos estão corretos

# Duplicados
duplicados = df.duplicated().sum()
print(f"Linhas duplicadas: {duplicados}") # 1081 neste momento

# Removendo as linhas duplicadas
df = df.drop_duplicates()
print(df.shape) # (283726 transações, 31 colunas)

# Checando se as duplicatas afetaram a proporção de fraude
print(df["Class"].value_counts(normalize=True)) 
# Class → 0: 0.998333 / 1: 0.001667 (99.8333% transações normais, 0.1667% são fraudes)

# ---

# Outliers em Amount - depois da remoção de linhas duplicadas

sns.boxenplot(x=df["Amount"])
plt.title("Distribuição de Amount — verificação de outliers (dados sem duplicados)")
plt.show()

# Decisão sobre os outliers: MANTER
# Em detecção de fraude, valores atípicos podem representar justamente o padrão que o
# modelo precisa aprender a reconhecer, não necessariamente um erro de coleta.
# Nenhuma remoção de outliers aplicada nesta etapa.

"""
5. FEATURE ENGINEERING
"""

print("\n=== 5. Feature Engineering ===")

import numpy as np

# Amount_log: suaviza a assimetria observada no boxenplot (cauda longa de outliers)
# Não depende de estatísticas do dataset inteiro — pode ser calculado antes do split
df["Amount_log"] = np.log1p(df["Amount"])

# Hour: converte Time (segundos desde a primeira transação) em hora do dia (0-23)
df["Hour"] = (df["Time"] // 3600) % 24


"""
6. SEPARAÇÃO TREINO/TESTE
"""

print("\n=== 6. Separação treino/teste ===")

from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)


"""
6.1 PADRONIZAÇÃO (StandardScaler) — aplicada apenas após o split
"""

print("\n=== 6.1 Padronização (StandardScaler - após o split) ===")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit_transform: aprende a média/desvio padrão E aplica, mas só no treino
X_train["Amount_scaled"] = scaler.fit_transform(X_train[["Amount"]])

# transform apenas: aplica a mesma transformação aprendida no treino, sem "aprender" de novo
X_test["Amount_scaled"] = scaler.transform(X_test[["Amount"]])

"""
7. BALANCEAMENTO DAS CLASSES
"""

print("\n=== 7. Balanceamento das classes ===")

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Conferindo a proporção original no treino, antes de qualquer balanceamento
print("Antes do balanceamento:")
print(y_train.value_counts(normalize=True))

# --- SMOTE: cria exemplos sintéticos da classe minoritária (fraude) ---
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# --- Undersampling: reduz exemplos da classe majoritária (normal) ---
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Conferindo o resultado de cada técnica
print("\nApós SMOTE:")
print(y_train_smote.value_counts(normalize=True))

print("\nApós Undersampling:")
print(y_train_under.value_counts(normalize=True))

# Diferença de tamanho final do dataset entre SMOTE e Undersampling
print(f"\nTamanho treino original: {len(X_train)}")
print(f"Tamanho treino SMOTE: {len(X_train_smote)}")
print(f"Tamanho treino Undersampling: {len(X_train_under)}")

"""
8. TREINAMENTO DOS MODELOS — comparando as versões balanceadas
"""

print("\n=== 8. Treinamento dos modelos ===")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

# Dicionário com cada versão do treino que queremos comparar
versoes_treino = {
    "Sem balanceamento": (X_train, y_train),
    "SMOTE": (X_train_smote, y_train_smote),
    "Undersampling": (X_train_under, y_train_under),
}

resultados = []

for nome, (X_tr, y_tr) in versoes_treino.items():
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_test)

    resultados.append({
        "Versão": nome,
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    })

    print(f"\n--- {nome} ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraude"]))

# Tabela final comparativa
df_resultados = pd.DataFrame(resultados)
print("\n=== Comparação final (sem o class_weight=balanced)===")
print(df_resultados)

# Testando class_weight="balanced" que usa os dados reais, sem ignorar a classe minoritária e ainda prestando mais atenção a essa pequena porcentagem

modelo_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
modelo_balanced.fit(X_train, y_train)
y_pred_balanced = modelo_balanced.predict(X_test)

resultados.append({
    "Versão": "class_weight=balanced",
    "Recall": recall_score(y_test, y_pred_balanced),
    "Precision": precision_score(y_test, y_pred_balanced),
    "F1": f1_score(y_test, y_pred_balanced),
})

print(classification_report(y_test, y_pred_balanced, target_names=["Normal", "Fraude"]))

"""
=== Resultados ===

              Versão    Recall  Precision        F1
0  Sem balanceamento  0.767606   0.964602  0.854902
1              SMOTE  0.781690   0.902439  0.837736
2      Undersampling  0.866197   0.050534  0.095497
---
              precision    recall  f1-score   support

      Normal       1.00      1.00      1.00     84976
      Fraude       0.93      0.76      0.84       142
    accuracy                           1.00     85118
   macro avg       0.97      0.88      0.92     85118
weighted avg       1.00      1.00      1.00     85118

Ranking final, considerando recall como prioridade (mas sem ignorar precision)
1. SMOTE — melhor recall entre as opções "saudáveis" (78,2%), com queda moderada de precision (90,2%)
2. Sem balanceamento — recall um pouco menor (76,8%), mas precision bem mais alta (96,5%)
3. class_weight="balanced" — praticamente empatado com "sem balanceamento", sem justificar a complexidade extra
4. Undersampling — descartado, colapsou a precision

"""

"""
10. AJUSTE FINO — testando threshold para os dois candidatos de balanceamento
"""

print("\n=== 10. Ajude fino - testando threshold com 'sem balanceamento' e SMOTE ===")

from sklearn.metrics import precision_recall_curve

# Retreinando os dois candidatos, agora guardando o modelo (não só a previsão)
modelo_sem_balanceamento = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_sem_balanceamento.fit(X_train, y_train)

modelo_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_smote.fit(X_train_smote, y_train_smote)

# Probabilidades da classe fraude (coluna 1), não a previsão final
probs_sem_balanceamento = modelo_sem_balanceamento.predict_proba(X_test)[:, 1]
probs_smote = modelo_smote.predict_proba(X_test)[:, 1]

# Testando vários thresholds manualmente
thresholds_testados = [0.5, 0.4, 0.3, 0.2, 0.1]

print("\n === Testando várias threshold ===")

for nome, probs in [("Sem balanceamento", probs_sem_balanceamento), ("SMOTE", probs_smote)]:
    print(f"\n=== {nome} ===")
    for t in thresholds_testados:
        y_pred_t = (probs > t).astype(int)
        r = recall_score(y_test, y_pred_t)
        p = precision_score(y_test, y_pred_t)
        f1 = f1_score(y_test, y_pred_t)
        print(f"Threshold {t:.1f} → Recall: {r:.3f} | Precision: {p:.3f} | F1: {f1:.3f}")

"""
10.1 ANÁLISE DE CUSTO — FN custa 10x mais que FP (suposição documentada do projeto)
"""

print("\n=== 10.1 Análise de custo ===")

from sklearn.metrics import confusion_matrix

CUSTO_FN = 10  # custo de deixar passar uma fraude
CUSTO_FP = 1   # custo de investigar uma transação normal à toa

def calcular_custo(y_real, y_pred, custo_fn=CUSTO_FN, custo_fp=CUSTO_FP):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
    custo_total = (fn * custo_fn) + (fp * custo_fp)
    return custo_total, fn, fp

thresholds_testados = [0.5, 0.4, 0.3, 0.2, 0.1]
resultados_custo = []

for nome, probs in [("Sem balanceamento", probs_sem_balanceamento), ("SMOTE", probs_smote)]:
    for t in thresholds_testados:
        y_pred_t = (probs > t).astype(int)
        custo, fn, fp = calcular_custo(y_test, y_pred_t)
        resultados_custo.append({
            "Versão": nome,
            "Threshold": t,
            "Falsos Negativos": fn,
            "Falsos Positivos": fp,
            "Custo Total": custo
        })

df_custo = pd.DataFrame(resultados_custo)
df_custo = df_custo.sort_values("Custo Total")
print(df_custo)

"""
              Versão  Threshold  Falsos Negativos  Falsos Positivos  Custo Total
7              SMOTE        0.3                26                25          285
8              SMOTE        0.2                24                64          304
6              SMOTE        0.4                29                16          306
2  Sem balanceamento        0.3                30                10          310
4  Sem balanceamento        0.1                28                33          313
1  Sem balanceamento        0.4                31                 5          315
3  Sem balanceamento        0.2                30                16          316
5              SMOTE        0.5                31                12          322
0  Sem balanceamento        0.5                33                 4          334
9              SMOTE        0.1                21               249          459

Posição	Versão	Threshold	FN	FP	Custo Total
🥇 1º	SMOTE	0,3	        26  25	285
"""

"""
10.2 AJUSTE DE HIPERPARÂMETROS — SMOTE por fold, otimizando pelo custo (não pelo recall isolado)
"""

print("\n Ajuste de hiperparâmetros - smote por fold (custo)")

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix

# Pipeline: SMOTE + modelo, para que o SMOTE seja recalculado em cada fold do CV
pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [5, 10, 20, None],
}

# Scorer customizado: otimiza pelo custo total (FN pesa 10x mais que FP), não pelo recall isolado
def custo_scorer(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    custo = (fn * CUSTO_FN) + (fp * CUSTO_FP)
    return -custo  # negativo porque o GridSearchCV sempre tenta MAXIMIZAR o score

scorer_customizado = make_scorer(custo_scorer)

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring=scorer_customizado,
    cv=5,
    n_jobs=-1
)

# Usamos X_train/y_train ORIGINAIS aqui (sem SMOTE pré-aplicado)
# O SMOTE roda dentro do pipeline, em cada fold, automaticamente
grid.fit(X_train, y_train)

print("Melhores hiperparâmetros:", grid.best_params_)
print("Melhor custo (validação cruzada):", -grid.best_score_)


# Avaliando o melhor modelo encontrado, no threshold já decidido (0.3)
melhor_modelo = grid.best_estimator_

probs_melhor_modelo = melhor_modelo.predict_proba(X_test)[:, 1]
y_pred_final = (probs_melhor_modelo > 0.3).astype(int)

print(classification_report(y_test, y_pred_final, target_names=["Normal", "Fraude"]))

custo_final, fn_final, fp_final = calcular_custo(y_test, y_pred_final)
print(f"Custo total (modelo ajustado): {custo_final} | FN: {fn_final} | FP: {fp_final}")