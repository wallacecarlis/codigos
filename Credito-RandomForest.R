# M�rcio, segue um arquivo onde o projeto era subir de 65% para 75%
# os ganhos de uma empresa de empr�stimos que apresentava perda de 35% (100-65),
# ou seja, melhorar a taxa de erro fazendo-a diminuir para menos de 35%.
# Al�m disso, verificar a classifica��o dos clientes em bom pagador ou ruim,
# ou seja, liberando o cr�dito ou n�o.


# Vamos instalar o pacote "randomForest" e suas depend�ncias:
install.packages('randomForest',dependencies=T)

# Agora vamos carregar o pacote:
library(randomForest)

# Aqui vamos carregar o arquivo "Cr�dito" com o separador e cabe�alho:
credito = read.csv(file.choose(),sep=';',header=T)

# Geramos dois conjuntos de dados aleat�rios para treino e teste,
# com aproximadamente 70% e 30%:
amostra = sample(2,1000,replace=T, prob=c(0.7,0.3))
creditotreino = credito[amostra==1,]
creditoteste = credito[amostra==2,]

# Gerando o modelo usando dados de treino:
floresta = randomForest(CLASSE ~ .,data=creditotreino, ntree=100,proximity=T)

# Testando o modelo fazendo a previs�o com dados de teste:
previsao = predict(floresta,creditoteste)

# Gerando a matriz de confusao:
floresta$confusion

# Calculando a taxa de erro:
taxaerro = (floresta$confusion[2] + floresta$confusion[3]) / sum(floresta$confusion)

# Verificando a taxa de erro:
taxaerro

# Ap�s algumas passagens, ficamos com 23%, ou seja, cerca de 77% de acerto.

# Como se pode perceber, no R o c�digo foi menor.
# Cada linguagem tem seus pr�s e contras.
# Abs.