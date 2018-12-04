from sklearn.naive_bayes import MultinomialNB
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from collections import Counter

#teste inicial: home, busca, logado => comprou
#home, busca
#home, logado
#busca, logado
#busca: 85.71% (7 testes)

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    # TREINANDO O ALGORITMO
    modelo.fit(treino_dados, treino_marcacoes)

    # TESTANDO O ALGORITMO
    resultado = modelo.predict(teste_dados)
    total_de_acertos = sum(resultado == teste_marcacoes)  # somando os valores true da comparaçao

    # CALCULANDO A PORCENTAGEM DE ACERTO DO ALGORITMO
    total_de_elementos = len(teste_dados)
    porcentagem_de_acertos = (100 * total_de_acertos / total_de_elementos)
    msg = "{} acertou {}%".format(nome, porcentagem_de_acertos)

    # PRINTANDO RESULTADO
    print(msg)
    return porcentagem_de_acertos


#DEFINICAO DE CONSTANTES
porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1
porcentagem_de_validacao = 1 - (porcentagem_de_treino + porcentagem_de_teste)


#ABERTURA DE ARQUIVOS E DEFINICAO DE DADOS E MARCADORES
df = pd.read_csv('buscas.csv')
X_df = df[['home','busca','logado']]
Y_df = df['comprou']


#TRATAMENTO DE DADOS COM DUMMIES
Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

#DEFININDO TAMANHO DO TREINO, TESTE E VALIDAÇAO E ATRIBUINDO SEUS DADOS
tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_teste = int(porcentagem_de_teste * len(Y))
tamanho_de_validacao = int(porcentagem_de_validacao * len(Y))

treino_dados = X[0:(tamanho_de_treino )]
treino_marcacoes = Y[0:(tamanho_de_treino)]

fim_de_teste = tamanho_de_treino + tamanho_de_teste

teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]


validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]


numero_de_teste = len(validacao_marcacoes)


#TESTE DO ALGORITMO BASE
acertos_base = max(Counter(teste_marcacoes).values()) #quantidade de vezes que o que mais se repete aparece
porcentagem_de_acertos_base = 100 * (acertos_base/len(validacao_marcacoes))


#MAIN
modeloMultinomial = MultinomialNB()
algoritmo1 = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

modeloAdaBoost = AdaBoostClassifier()
algoritmo2 = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if algoritmo1 >= algoritmo2:
    vencedor =  modeloMultinomial
else:
    vencedor = modeloAdaBoost

vencedor_predict = vencedor.predict(validacao_dados)
acertos_vencedor = sum(vencedor_predict == validacao_marcacoes)
porcentagem_acerto_validacao = 100 * (acertos_vencedor/len(validacao_marcacoes))
print("Porcentagem de acertos na validação: {}".format(porcentagem_acerto_validacao))

print("Algoritmo base: {}%".format(porcentagem_de_acertos_base))
print("Numero de testes: {}".format(numero_de_teste))
