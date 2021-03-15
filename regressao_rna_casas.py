# Regressao Linear Multipla - Rede Neural
# Base de dados precos das casas

# Importando as bibliotecas
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Atriuindo o arquivo a variavel "base"
base = pd.read_csv('https://raw.githubusercontent.com/wallacecarlis/arquivos/main/house_prices.csv')

# Selecionando os atributos previsores "x"
x = base.iloc[:, 3:19].values
# Selecionando os preços das casas
y = base.iloc[:, 2:3].values

# Criando os objetos da padronizacao (necessaria para Redes Neurais)
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# Treinamento da base: atributos "previsores e respostas"
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

# Divisão da base de dados em treino e teste
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

# Criando o objeto de duas camadas ocultas com 9 neurônios cada
# Obs.: 16 colunas + 1 = 17. 17 / 2 = 8,5. Arredondando para 9.
regressor = MLPRegressor(hidden_layer_sizes = (9,9))
# Treinamento da base: atributos "previsores e respostas"
regressor.fit(x_treinamento, y_treinamento.ravel())

# Evidenciando o resultado do treinamento
score = regressor.score(x_treinamento, y_treinamento)

# score dos testes
regressor.score(x_teste, y_teste)

# Criando as previsoes (testes)
previsoes = regressor.predict(x_teste)

# Desnormalizando
y_teste = scaler_y.inverse_transform(y_teste)
previsoes = scaler_y.inverse_transform(previsoes)

# Obtendo a media do erro
mae = mean_absolute_error(y_teste, previsoes)
