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

# Ao rodar essa função, estamos calculando a proporção de cada tipo de transação. O resultado mostra que 0.001 transações são fraudes. Isso mostra um desbalanceamento entre as classes 0 e 1, não significa que é algo bom. Isso ode ser um problema porque o modelo pode aprender que 1 não é fraude.
# Neste caso, podemos usar o Recall para identificar quantas fraudes reais ele encontrou.

#----------------------

# Feature Engineering
# Criando variáveis que ajudam o modelo
# Por que criar outras variáveis? Nem sempre as variáveis originais como as desse projeto como Time, Amount e V1-V28 são o suficiente, então é melhor criar novas variáveis que vão representar melhor o comportamento dos dados. A ideia é transformar os dados brutos em informações que sejam úteis para o modelo. O modelo não interpreta o contexto, ele só aprende contadores matemáticos.

import numpy as np

# Criando uma nova variável chamada Amount_log. log1p aplica uma transformação logarítimica nos valores da coluna Amount. É uma forma de comprimir valores muito grande, diminuindo-os na escala; valores pequenos são menos afetados. Isso evita problemas com valores iguais = 0, já que log = 0 não vai existir. Algumas transações são muito pequenas e outras são muito grandes e isso pode dificultar o aprendizado do modelo porque os valores vão ficar muito espalhados, e o log reduz essa diferença e a distribuição fica mais equilibrada, ajudando o modelo a identificar padrões com mais facilidade.
df["Amount_log"] = np.log1p(df["Amount"])

#----------------------

# Padronização dos dados

from sklearn.preprocessing import StandardScaler

# O standard Scaler vai transformar os dados para que tenha a média = 0 e desvio padrão = 1
scaler = StandardScaler()
# Criando outra variável chamada Amount_scaled com o resultado da função
df["Amount_scaled"] = scaler.fit_transform(df["Amount"])

#----------------------

# Preparando os dados para treinar o modelo

from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)  # vai representar todas as variáveis de entrada, e remove a coluna Class
y = df["Class"]               # representa o target, o que a gente quer ver, neste caso a Class

# divindo em teste(30%) e treino(70%), sendo 30% em teste declarado no test_size=0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

