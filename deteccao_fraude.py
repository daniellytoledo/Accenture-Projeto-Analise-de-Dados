import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)
df.head()

# A coluna mais importante é a coluna Class, sendo o valor 0 - não contém fraude, e se possui valor diferente - possível fraude na transação.

# Problema de classificação desbalanceada
# Fraudes são raras - modelo pode ignorar a classe 1.

print(df["Class"].value_counts(normalize=True))
# 0    0.998273
# 1    0.001727
# Name: proportion, dtype: float64

# Ao rodar essa função, estamos calculando a proporção de cada tipo de transação. O resultado mostra que 0.001 transações são fraudes. Isso mostra um desbalanceamento entre as classes 0 e 1, não significa que é algo bom.
# Isso pode ser um problema porque o modelo pode aprender a prever quase sempre a classe 0 (não fraude), ignorando a classe 1 (fraude).
# O Recall mede a proporção de fraudes reais que o modelo conseguiu identificar.

#----------------------

# Feature Engineering
# Criando variáveis que ajudam o modelo
# Por que criar outras variáveis? Nem sempre as variáveis originais como as desse projeto como Time, Amount e V1-V28 são o suficiente, então é melhor criar novas variáveis que vão representar melhor o comportamento dos dados. A ideia é transformar os dados brutos em informações que sejam úteis para o modelo.
# O modelo não entende o significado dos dados; ele aprende padrões matemáticos presentes nas variáveis.

import numpy as np

# Criando uma nova variável chamada Amount_log. log1p aplica uma transformação logarítimica nos valores da coluna Amount. É uma forma de comprimir valores muito grande, diminuindo-os na escala; valores pequenos são menos afetados. A função log1p(x) calcula log(1 + x), permitindo transformar valores iguais a 0 sem gerar erro, já que log(1) = 0. Algumas transações são muito pequenas e outras são muito grandes e isso pode dificultar o aprendizado do modelo porque os valores vão ficar muito espalhados, e o log reduz essa diferença e a distribuição fica mais equilibrada, ajudando o modelo a identificar padrões com mais facilidade.
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

# Prever se a transação é fraude (classe 1) ou não fraude (classe 0)
# Por que da regressão logistíca? É um dos modelos mais simples e é geralmente usado para base line para servir de modelo para os outros. 
# O max_iter define o número máximo de iterações que o algoritmo poderá executar para encontrar os melhores coeficientes.
# O model.fit é o modelo aprende padrões a partir dos dados de train, encontrando relações entre as variáveis
# Model.predict vai fazer previsões dos dados de teste que ele nunca viu, agora é momento que ele vai olhar para os 30% de teste.

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# ao executar o código acima quando max_inter=1000 vai dar um aviso de:
# ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s) (status=1): STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
# não significa que deu erro, mas que o modelo não conseguiu terminar o processo de aprendizado completamente com o número definido no max_inter porque ele não encontrou a melhor solução quando o treinamento foi interrompido, ou seja, o modelo parou porque chegou no máximo permitido, não porque terminou de aprender.
# neste caso, o que podemos fazer é aumentar o número de interações e melhorar a escala dos dados.

# podemos ver como o modelo se saiu usando o classification_report que é onde conseguimos ver as métricas mais importantes pra nossa avaliação.

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#                 precision    recall  f1-score   support
#
#           0       1.00      1.00      1.00     85295
#           1       0.86      0.64      0.73       148  # representa a classe 1 - fraudes
#
#    accuracy                           1.00     85443
#   macro avg       0.93      0.82      0.87     85443
# weighted avg      1.00      1.00      1.00     85443

# ANALISANDO O REPORT:
# Na coluna precision, onde mostra a classe 1, das transações que o modelo classificou como fraude, 86% realmente eram fraudes.
# O Recall de 64% indica que o modelo conseguiu identificar 64% das fraudes reais.. Ponto crítico porque o modelo está deixando passar 36% de fraude
# f1-score de 0.73 que é um equilibrio, uma média harmonica entre precisão e recall, dando uma visão geral da qualidade do modelo
# A accuracy aparece como 1.00 devido ao arredondamento, mas, em problemas desbalanceados, ela pode ser enganosa. O modelo pode apresentar alta accuracy mesmo deixando de identificar muitas fraudes.
# Então, o dado mais importante é o recall com 64% de fraudes reais identificadas

#----------------------

# Usando outro método de avaliação: A curva ROC
# É uma métrica muito importante em modelo de classificação


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_probs = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_probs)

# O gráfico mostra o desempenho do modelo em diferentes limiares de decisão
# Cada ponto da curva vai representar um limite diferente da decisão do modelo,dando para analisar o comportamento geral do modelo
# Quanto mais a curva se aproxima do canto superior esquerdo, melhor é o modelo. alto recall e baixo erro

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate") # representa quantas transações normais foram classificadas como fraude
plt.ylabel("True Positive Rate")  # representa o mesmo que o recall: fraudes reais identificadas
plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))
# AUC: 0.9283981824605543
# A AUC foi aproximadamente 0,93 (ou 93%), indicando que o modelo tem boa capacidade de distinguir transações fraudulentas de transações normais, mas ainda não está perfeito.

#----------------------

# Precision Recall Curve
# Ainda mais importante em problemas de fraudes desbalanceadas
# A curva Precision-Recall mostra a relação entre a precisão (quantas previsões de fraude realmente eram fraudes) e o recall (quantas fraudes reais foram identificadas).
# A partir do momento que o recall aumenta, o modelo começa a identificar mais fraudes

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
# Pega todas as fraudes e seleciona a mesma quantidade de transações, ou seja, reduzimos a classe majoritária 0 para ficar do mesmo tamanho da classe minoritária que é a classe 1
# Junta as fraudes com todas as transações normais novamente.
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0]
df_under = pd.concat([fraudes, normais])

# Oversamping
# O oversampling aumenta a quantidade de exemplos da classe minoritária para equilibrar o conjunto de dados. Uma das técnicas mais utilizadas é o SMOTE.
# O modelo aprende melhor a identificar fraudes
# O problema é que acabamos perdendo dados normais, o que reduz a qualidade do dado, ou seja, nunca é perfeito
from imblearn.over_sampling import SMOTE

# O smote() cria um dataset equilibrado com a mesma quantidade de frauses e não fraudes.
# O SMOTE gera exemplos sintéticos da classe minoritária a partir das amostras existentes, em vez de simplesmente duplicá-las.
# Não perde dados e ainda aumenta a representação da fraude, por outro lado, estamos criando dados artificiais, o que pode incluir ruídos
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

#----------------------

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

#               precision    recall   f1-score  support
#
#           0       1.00      1.00      1.00     85295
#           1       0.74      0.80      0.77       148
#
#    accuracy                           1.00     85443
#   macro avg       0.87      0.90      0.89     85443
# weighted avg       1.00      1.00      1.00     85443

