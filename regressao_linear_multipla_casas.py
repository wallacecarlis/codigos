# Regressão Linear Múltipla
# Base de dados de preços das casas

# Importando as bibliotecas
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


base = pd.read_csv('https://raw.githubusercontent.com/wallacecarlis/arquivos/main/house_prices.csv')

# Selecionando os atributos previsores "x"
x = base.iloc[:, 3:19].values
# Selecionando os preços das casas
y = base.iloc[:, 2].values


# Divisão da base de dados em treino e teste
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
# Criando o objeto
regressor = LinearRegression()

# Treinamento da base: atributos "previsores e respostas"
regressor.fit(x_treinamento, y_treinamento)

# evidenciando o resultado de 0.70 (correlacao positiva forte)
# com mais variáveis ja e possi­vel se aproximar do preco correto frete
# a regressão linear simples
score = regressor.score(x_treinamento, y_treinamento)

# Criando as previsoes (testes)
previsoes = regressor.predict(x_teste)

# Obtendo a media do erro
mae = mean_absolute_error(y_teste, previsoes)

# score dos testes
regressor.score(x_teste, y_teste)

# Parametros 
regressor.intercept_ # constante
regressor.coef_ # coeficiente
len(regressor.coef_) # quantidade de coeficientes
