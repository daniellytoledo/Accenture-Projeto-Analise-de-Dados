# Conclusão do Projeto — Detecção de Fraudes

---

## O que o projeto respondeu

**Pergunta original:** dado o histórico de transações, é possível identificar quais são
fraudulentas?

**Resposta curta:** sim, de forma bastante eficaz. O modelo final consegue identificar
corretamente **82% das fraudes reais**, gerando apenas **25 falsos alarmes a cada 85 mil
transações analisadas** — uma taxa de falso positivo muito baixa (cerca de 0,03%).

## Um esclarecimento importante sobre o que o modelo entrega

O modelo **não gera regras explicáveis** do tipo "transações de valor alto à noite são
fraude". Ele calcula, para cada transação, uma **probabilidade** de ser fraude, com base em
padrões estatísticos aprendidos a partir de milhares de exemplos — não uma lista de
condições que se possa ler como uma frase.

Isso é ainda mais verdadeiro neste projeto porque as variáveis mais importantes (`V14`,
`V12`, `V4`...) são **componentes de PCA**: combinações matemáticas abstratas criadas para
anonimizar os dados reais dos cartões. Não é possível dizer o que "V14" representa no mundo
real — só é possível dizer que, estatisticamente, valores muito negativos dela estão
fortemente associados a fraude segundo o que o modelo aprendeu.

## Principais conclusões

### 1. O sistema é viável e traz valor de negócio

Considerando que deixar passar uma fraude custa (nesta suposição) 10x mais do que investigar
uma transação normal à toa, o modelo reduz drasticamente o prejuízo esperado em comparação a
não ter nenhum sistema de detecção — capturando a maior parte das fraudes com um número
pequeno de falsos alarmes.

### 2. O modelo depende fortemente de um padrão específico (`V14`)

A variável `V14` é, disparada, a mais influente na decisão do modelo. Quando ela está em
valores muito negativos, o modelo identifica fraude com bastante confiança. Isso gera um
padrão de acerto e erro bem definido:

- **Acerta com folga** quando a fraude segue esse padrão extremo
- **Erra por falso negativo** quando a fraude é mais "sutil" (V14 não tão extremo)
- **Erra por falso positivo** quando uma transação normal, por coincidência, também tem V14
  muito negativo

Isso significa que o modelo tem um **ponto único de fragilidade** conhecido — informação
valiosa tanto para melhorias futuras quanto para monitoramento em produção.

### 3. Valor e horário da transação pesam menos do que o esperado

As duas variáveis criadas na etapa de feature engineering (`Amount_log`, o valor da
transação; e `Hour`, a hora do dia) **não apareceram entre as variáveis mais importantes**
do modelo. Isso contraria uma intuição comum (de que transações de valor alto ou em
horários incomuns seriam automaticamente mais suspeitas) — neste dataset, os padrões
capturados pelas variáveis anonimizadas de PCA são muito mais determinantes do que essas
características superficiais.

### 4. A escolha do balanceamento e do threshold não foi arbitrária

Testamos formalmente quatro abordagens de balanceamento (SMOTE, undersampling,
`class_weight`, e nenhum balanceamento) e cinco pontos de corte diferentes, decidindo pela
combinação de menor custo total — não pela combinação com a métrica isolada mais alta. Essa
metodologia é replicável para qualquer projeto de classificação desbalanceada, não só fraude.

## Limitações conhecidas do projeto

- O dataset é de um período curto (2 dias de transações), o que pode não capturar padrões
  sazonais (ex: fraude aumentando em datas comemorativas)
- Os pesos de custo (falso negativo = 10x falso positivo) são uma **suposição documentada**,
  não valores reais de nenhuma empresa específica — numa aplicação real, esses valores
  viriam de dados financeiros concretos
- Como as variáveis mais importantes vêm de PCA, o modelo não oferece uma explicação de
  negócio "humanamente legível" para cada decisão — apenas uma explicação estatística

## Se alguém perguntar "então quais transações são fraude?"

A resposta correta para essa pergunta seria algo como:

> *"O modelo não aponta características fixas tipo 'toda fraude tem tal perfil' — ele calcula
> uma probabilidade de fraude para cada transação, baseada em padrões complexos que aprendeu
> nos dados. Consegui identificar, via SHAP, que a decisão do modelo é fortemente guiada por
> uma variável específica (V14, um componente de PCA), mas como os dados são anonimizados,
> não dá para traduzir isso numa regra de negócio simples — o valor está na capacidade
> preditiva do modelo, não numa lista de regras explicáveis."*