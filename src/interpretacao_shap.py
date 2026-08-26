
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
11.1 CASOS INDIVIDUAIS — encontrando exemplos de cada tipo (acerto, falso positivo, falso negativo)
"""

# Reconstruindo as previsões do modelo para a mesma amostra usada no SHAP
probs_amostra = melhor_modelo.predict_proba(X_test_amostra)[:, 1]
y_pred_amostra = (probs_amostra > 0.3).astype(int)
y_real_amostra = y_test[:200].values

# Variáveis que vão guardar o índice de cada tipo de caso, começando vazias
idx_acerto_fraude = None
idx_falso_positivo = None
idx_falso_negativo = None

for i in range(len(y_real_amostra)):
    se_e_fraude_real = y_real_amostra[i] == 1
    se_previu_fraude = y_pred_amostra[i] == 1

    if se_e_fraude_real and se_previu_fraude and idx_acerto_fraude is None:
        idx_acerto_fraude = i

    if not se_e_fraude_real and se_previu_fraude and idx_falso_positivo is None:
        idx_falso_positivo = i

    if se_e_fraude_real and not se_previu_fraude and idx_falso_negativo is None:
        idx_falso_negativo = i

# Mostrando o que foi encontrado (ou não) na amostra
print(f"Índice de acerto de fraude: {idx_acerto_fraude}")
print(f"Índice de falso positivo: {idx_falso_positivo}")
print(f"Índice de falso negativo: {idx_falso_negativo}")
