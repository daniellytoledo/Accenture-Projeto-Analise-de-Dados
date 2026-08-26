
# %%
"""
Etapa 11: Interpretação / Explicabilidade (SHAP)
Descrição: Carrega o modelo já treinado (models/modelo_fraude_final.pkl) e gera as
explicações SHAP, sem precisar retreinar o modelo (GridSearchCV já rodou em
deteccao_fraude.py).
"""

import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split

# Carregando o modelo final e o scaler já treinados
artefatos = joblib.load("models/modelo_fraude_final.pkl")
melhor_modelo = artefatos["modelo"]
scaler = artefatos["scaler"]

# Recarregando os dados e refazendo o pipeline até o split (rápido, sem GridSearchCV)
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
df = df.drop_duplicates()
df["Amount_log"] = np.log1p(df["Amount"])
df["Hour"] = (df["Time"] // 3600) % 24

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Aplicando o scaler já treinado (sem fit novamente — só transform)
X_train["Amount_scaled"] = scaler.transform(X_train[["Amount"]])
X_test["Amount_scaled"] = scaler.transform(X_test[["Amount"]])

# Extraindo o Random Forest de dentro do pipeline
rf_final = melhor_modelo.named_steps["rf"]

# Amostra do teste para explicar
X_test_amostra = X_test[:200]

explainer = shap.Explainer(rf_final)
shap_values = explainer(X_test_amostra)

shap.plots.bar(shap_values[:, :, 1])
shap.plots.beeswarm(shap_values[:, :, 1])

# %%
"""
11.1 CASOS INDIVIDUAIS — buscando em todo o X_test, não só numa amostra pequena
"""

# Previsões para todo o conjunto de teste
probs_teste_completo = melhor_modelo.predict_proba(X_test)[:, 1]
y_pred_teste_completo = (probs_teste_completo > 0.3).astype(int)
y_real_teste_completo = y_test.values

idx_acerto_fraude = None
idx_falso_positivo = None
idx_falso_negativo = None

for i in range(len(y_real_teste_completo)):
    real = y_real_teste_completo[i]
    previsto = y_pred_teste_completo[i]

    if real == 1 and previsto == 1 and idx_acerto_fraude is None:
        idx_acerto_fraude = i

    if real == 0 and previsto == 1 and idx_falso_positivo is None:
        idx_falso_positivo = i

    if real == 1 and previsto == 0 and idx_falso_negativo is None:
        idx_falso_negativo = i

    # Para assim que os três forem encontrados, sem precisar varrer tudo
    if None not in (idx_acerto_fraude, idx_falso_positivo, idx_falso_negativo):
        break

print(f"Índice de acerto de fraude: {idx_acerto_fraude}")
print(f"Índice de falso positivo: {idx_falso_positivo}")
print(f"Índice de falso negativo: {idx_falso_negativo}")

# %%

"""
11.2 WATERFALL — explicando os três casos específicos encontrados
"""

# Selecionando as linhas exatas do X_test usando os índices encontrados
casos_especificos = X_test.iloc[[idx_acerto_fraude, idx_falso_positivo, idx_falso_negativo]]

# Gerando as explicações SHAP só para essas 3 linhas (rápido, não precisa da amostra de 200)
shap_values_casos = explainer(casos_especificos)

# --- Caso 1: Acerto de fraude ---
print("=== Acerto de Fraude (fraude real, o modelo identificou corretamente) ===")
shap.plots.waterfall(shap_values_casos[0, :, 1])

# --- Caso 2: Falso Positivo ---
print("=== Falso Positivo (transação normal, o modelo marcou como fraude) ===")
shap.plots.waterfall(shap_values_casos[1, :, 1])

# --- Caso 3: Falso Negativo ---
print("=== Falso Negativo (fraude real, o modelo não detectou) ===")
shap.plots.waterfall(shap_values_casos[2, :, 1])