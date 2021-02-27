# Vamos instalar o pacote "randomForest" e suas dependências:
install.packages('randomForest',dependencies=T)

# Agora vamos carregar o pacote:
library(randomForest)

# Aqui vamos carregar o arquivo "Crédito" com o separador e cabeçalho:
credito = read.csv(file.choose(),sep=';',header=T)

# Geramos dois conjuntos de dados aleatórios para treino e teste,
# com aproximadamente 70% e 30%:
amostra = sample(2,1000,replace=T, prob=c(0.7,0.3))
creditotreino = credito[amostra==1,]
creditoteste = credito[amostra==2,]

# Gerando o modelo usando dados de treino:
floresta = randomForest(CLASSE ~ .,data=creditotreino, ntree=100,proximity=T)

# Testando o modelo fazendo a previsão com dados de teste:
previsao = predict(floresta,creditoteste)

# Gerando a matriz de confusao:
floresta$confusion

# Calculando a taxa de erro:
taxaerro = (floresta$confusion[2] + floresta$confusion[3]) / sum(floresta$confusion)

# Verificando a taxa de erro:
taxaerro

# Após algumas passagens, ficamos com 23%, ou seja, cerca de 77% de acerto.
