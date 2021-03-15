#Regressao Linear Simples

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

base = pd.read_csv('house_prices.csv')

# Atributo previsor - metragem quadrada da casa
x = base.iloc[:, 5:6].values
# preco de cada casa
y = base.iloc[:, 2].values


# "x_treinamento" para os parametros "b0 e b1"
# "x_teste" para avaliação do desempenho sobre a previsao
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
# Criacao do objeto
regressor = LinearRegression()

# Treinamento da base: atributos "previsores e respostas"
regressor.fit(x_treinamento, y_treinamento)

# evidenciando o resultado de 0.49
# somente pelo tamanho da casa nao e possi�vel achar o preco correto
score = regressor.score(x_treinamento, y_treinamento)

# plotando o resultado de 0.49
plt.scatter(x_treinamento, y_treinamento)
plt.plot(x_treinamento, regressor.predict(x_treinamento), color = 'red')

# Criando as previsoes (testes)
previsoes = regressor.predict(x_teste)

# Gerando base das diferencas obtidas no resultado da previsao
resultado = abs(y_teste - previsoes)

# media do erro
resultado.mean() 
# ou
mae = mean_absolute_error(y_teste, previsoes)

# Elevando o erro ao quadrado
msq = mean_squared_error(y_teste, previsoes)

# Gerando grafico com base de dados testes
plt.scatter(x_teste, y_teste)
plt.plot(x_teste, regressor.predict(x_teste), color = 'red')

# score dos testes
regressor.score(x_teste, y_teste)
