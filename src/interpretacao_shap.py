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