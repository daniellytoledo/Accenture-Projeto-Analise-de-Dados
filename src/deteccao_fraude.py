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

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print(df.head())

"""
3. EXPLORAÇÃO INICIAL (EDA)
"""

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