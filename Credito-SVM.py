# Márcio, segue um arquivo onde o projeto era subir de 65% para 75%
# os ganhos de uma empresa de empréstimos que apresentava perda de 35% (100-65),
# classificando os clientes em bom pagador ou ruim, ou seja, liberando o crédito ou não.

# Vou mencionar cada trecho do código.
# Neste exemplo vamos de SVM.

# Iniciando com a importação do pandas:
import pandas as pd

# Márcio, aqui o parâmetro encoding indica que os dados estão em português:
base = pd.read_csv('Credito.csv', sep = ';', encoding = 'cp860')

# Criação da variável X que presenta os atributos previsores,
# alguns professores usam o nome da variável de previsores:
X = base.iloc[:, 0:19].values

# Criação da variável y que contém as respostas,
# alguns professores usam o nome da variável de classe:
y = base.iloc[:, 19].values

# Márcio, outra opção seria usar o drop, por exemplo,
# variável = base_de_dados.columns.drop(['variáveis_não_utilizadas']),
# retornando assim a coluna desejada.


# Márcio, aqui será o momento do pré-processamento:

# Transformação dos atributos categóricos no formato string para números
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 5] = labelencoder.fit_transform(X[:, 5])
X[:, 6] = labelencoder.fit_transform(X[:, 6])
X[:, 8] = labelencoder.fit_transform(X[:, 8])
X[:, 9] = labelencoder.fit_transform(X[:, 9])
X[:, 11] = labelencoder.fit_transform(X[:, 11])
X[:, 13] = labelencoder.fit_transform(X[:, 13])
X[:, 14] = labelencoder.fit_transform(X[:, 14])
X[:, 16] = labelencoder.fit_transform(X[:, 16])
X[:, 18] = labelencoder.fit_transform(X[:, 18])


# Esse é o passo da padronização dos dados,
# testei sem este passo e deu mais ou menos 5% de diferença:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Finalmente, após definição e separação das variáveis, pré-processamento e
# padronização, será o momento do treino e teste: 

# Divisão da base em treino e teste (70% para treinamento e 30% para teste)
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)


# Márcio, até aqui os algoritmos são iguais,
# agora vamos rodar o SVM:
from sklearn.svm import SVC
classificador = SVC(kernel = 'rbf', random_state = 1, C = 2.0, gamma = 'auto')

# Vamos rodar os treinamentos e fazer a previsão:
classificador.fit(X_treinamento, y_treinamento)
previsoes = classificador.predict(X_teste)

# Vamos confrontar a precisão e a matriz de confusão
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(y_teste, previsoes)
matriz = confusion_matrix(y_teste, previsoes)
taxa_erro = 1 - precisao

# Márcio, aqui é uma visualização do resultado:
import collections
collections.Counter(y_teste)

# Após algumas passagens, consegui 76.7%
# Abs.
