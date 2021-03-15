# Regressao Linear Multipla - Random Forest
# Base de dados de preços das casas retiradas do Kaggle
# "House Sales in King County, USA"

# Importando as bibliotecas
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Atriuindo o arquivo a variavel "base"
base = pd.read_csv('https://raw.githubusercontent.com/wallacecarlis/arquivos/main/house_prices.csv')

# Selecionando os atributos previsores "x"
x = base.iloc[:, 3:19].values
# Selecionando os preços das casas
y = base.iloc[:, 2].values

# Divisão da base de dados em treino e teste
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y,
                                                                 test_size = 0.3,
                                                                 random_state = 0)

# Criando o objeto
# quantidade de arvores = 100
regressor = RandomForestRegressor(n_estimators = 100)

# Treinamento da base: atributos "previsores e respostas"
regressor.fit(x_treinamento, y_treinamento)

# Evidenciando o resultado do treinamento
score = regressor.score(x_treinamento, y_treinamento)

# Criando as previsoes (testes)
previsoes = regressor.predict(x_teste)

# Obtendo a media do erro
mae = mean_absolute_error(y_teste, previsoes)

# score dos testes
regressor.score(x_teste, y_teste)
