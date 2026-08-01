import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)
df.head()

print(df["Class"].value_counts(normalize=True))

#----------------------
# Feature Engineering

import numpy as np

df["Amount_log"] = np.log1p(df["Amount"])

#----------------------
# Padronização dos dados

from sklearn.preprocessing import StandardScaler

# O standard Scaler vai transformar os dados para que tenha a média = 0 e desvio padrão = 1
scaler = StandardScaler()
# Criando outra variável chamada Amount_scaled com o resultado da função
df["Amount_scaled"] = scaler.fit_transform(
    df[["Amount"]]
)

#----------------------
# Preparando os dados para treinar o modelo

from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)  # vai representar todas as variáveis de entrada, e remove a coluna Class
y = df["Class"]               # representa o target, o que a gente quer ver, neste caso a Class

# divindo em teste(30%) e treino(70%), sendo 30% em teste declarado no test_size=0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

#----------------------


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#----------------------
# Usando outro método de avaliação: A curva ROC



from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_probs = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate") # representa quantas transações normais foram classificadas como fraude
plt.ylabel("True Positive Rate")  # representa o mesmo que o recall: fraudes reais identificadas
plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))

#----------------------
# Precision Recall Curve

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")    # fraudes reais identificadas
plt.ylabel("Precision") # precisão, quantas realmente são fraudes
plt.show()

#----------------------
# Formas de balancear:

# Undersampling
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0]
df_under = pd.concat([fraudes, normais])

# Oversamping
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

#----------------------

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50, # escolhendo 50 árvores
    max_depth=10, # limitando a profundidade das árvores
    class_weight="balanced", # ajustando o peso das classes
    n_jobs=-1, 
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

#----------------------
# Testando um pipeline

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

#----------------------
# THRESHOLD

threshhold = 0.3

y_pred_custom = (y_probs > threshhold).astype(int)
print(classification_report(y_test, y_pred_custom))

#----------------------
# Modelo Avançado - XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier(
    scale_pos_weight=10, # ajuda com desbalanceamento
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))

#----------------------
# Importância de variáveis

import matplotlib.pyplot as plt

importancias = xgb.feature_importances_

plt.bar(range(len(importancias)), importancias)
plt.title("Importância das variáveis")
plt.show()

#----------------------
# Ajuste de Hiperparâmetros
# Testando várias combinações para melhorar o modelo. 

from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
}

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss"),
    param_grid,
    scoring="recall",
    cv=3 
)

grid.fit(X_train, y_train)
print("Melhor modelo:", grid.best_params_)

#----------------------
# Explicabilidade (SHAP)

import shap

explainer = shap.Explainer(xgb)
shap_values = explainer(X_test[:100])
shap.plots.bar(shap_values)

