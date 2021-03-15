# Regressao Linear Multipla - Rede Neural
# Base de dados precos das casas

import pandas as pd
base = pd.read_csv('https://raw.githubusercontent.com/wallacecarlis/arquivos/main/house_prices.csv')

x = base.iloc[:, 3:19].values
y = base.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes = (9,9))
regressor.fit(x_treinamento, y_treinamento.ravel())
score = regressor.score(x_treinamento, y_treinamento)

regressor.score(x_teste, y_teste)

previsoes = regressor.predict(x_teste)
y_teste = scaler_y.inverse_transform(y_teste)
previsoes = scaler_y.inverse_transform(previsoes)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)
