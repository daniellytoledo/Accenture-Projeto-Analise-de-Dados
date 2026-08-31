"""
Etapa 12: Comunicação dos Resultados — Exportação para o Power BI
Descrição: Carrega o modelo já treinado (models/modelo_fraude_final.pkl) e gera um CSV
com previsões, probabilidades e custo de cada transação do conjunto de teste, pronto para
ser consumido pelo Power BI. Não precisa retreinar o modelo.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================================
# Carregando o modelo final e o scaler já treinados
# ==========================================================
artefatos = joblib.load("models/modelo_fraude_final.pkl")
melhor_modelo = artefatos["modelo"]
scaler = artefatos["scaler"]

# ==========================================================
# Recarregando os dados e refazendo o pipeline até o split
# (rápido — não envolve o GridSearchCV)
# ==========================================================
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

# ==========================================================
# 12. COMUNICAÇÃO — exportando dados para o Power BI
# ==========================================================

CUSTO_FN = 10  # custo de deixar passar uma fraude (mesma suposição usada em todo o projeto)
CUSTO_FP = 1   # custo de investigar uma transação normal à toa

# Probabilidades e previsão final para todo o conjunto de teste
probs_final = melhor_modelo.predict_proba(X_test)[:, 1]
y_pred_final_completo = (probs_final > 0.3).astype(int)

df_powerbi = X_test.copy()
df_powerbi["Class_Real"] = y_test.values
df_powerbi["Probabilidade_Fraude"] = probs_final
df_powerbi["Previsao"] = y_pred_final_completo

# Classificando cada transação pelo tipo de resultado (facilita filtros no Power BI)
def classificar_resultado(row):
    if row["Class_Real"] == 1 and row["Previsao"] == 1:
        return "Acerto - Fraude Detectada"
    elif row["Class_Real"] == 0 and row["Previsao"] == 0:
        return "Acerto - Transação Normal"
    elif row["Class_Real"] == 0 and row["Previsao"] == 1:
        return "Falso Positivo"
    else:
        return "Falso Negativo"

df_powerbi["Resultado"] = df_powerbi.apply(classificar_resultado, axis=1)

# Custo individual de cada transação (para somar depois no Power BI)
df_powerbi["Custo"] = df_powerbi["Resultado"].map({
    "Falso Negativo": CUSTO_FN,
    "Falso Positivo": CUSTO_FP,
    "Acerto - Fraude Detectada": 0,
    "Acerto - Transação Normal": 0,
})

# Garantindo que a pasta de saída existe, mesmo em outra máquina
os.makedirs("outputs", exist_ok=True)

df_powerbi.to_csv(
    "outputs/dados_powerbi.csv",
    index=False,
    sep=";",
    decimal=","
)
print("Arquivo exportado:", df_powerbi.shape)