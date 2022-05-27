# -*- coding: utf-8 -*-
#Fábio Franz
#Matheus Pasold
#Minéia Maschio

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from matplotlib.animation import adjusted_figsize
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    # Importar dados pré processados
    import pickle

    with open('risco_credito.pkl', 'rb') as f:
        X_risco_credito, y_risco_credito = pickle.load(f)

    # Calcular árvore de decisão utilizando como critério a entropia
    arvore_risco_credito = dtc(criterion='entropy')

    arvore_risco_credito.fit(X_risco_credito, y_risco_credito)

    # Utilizar o feature_importances_ para retornar a importância de cada atributo
    print('\n feature_importances')
    print(arvore_risco_credito.feature_importances_)
    print('O maior ganho é da Renda Anual')

    # Gerar uma visualização da árvore de decisão
    fn = ['Historia de credito', 'Divida', 'Garantias', 'Renda anual']
    cn = ['Alto', 'Baixo', 'Moderado']

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=3000)

    sklearn.tree.plot_tree(arvore_risco_credito, feature_names=fn, class_names=cn, filled=True)

    fig.savefig('imagem.png')

    # Previsões
    print('\nPrevisão história boa, dívida alta, garantia nenhuma, renda > 35')
    previsao1 = arvore_risco_credito.predict([[0, 0, 1, 2]]);
    print(previsao1)

    print('\nPrevisão história ruim, dívida alta, garantia adequada, renda < 15')
    previsao2 = arvore_risco_credito.predict([[2, 0, 0, 0]]);
    print(previsao2)

    # Abrir arquivo
    with open('credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

    # Verificar tamanho das bases
    print('\nTamanho base treinamento')
    print('x =' + str(np.shape(X_credit_treinamento)))
    print('y =' + str(np.shape(y_credit_treinamento)))

    print('\nTamanho base de teste')
    print('x =' + str(np.shape(X_credit_teste)))
    print('y =' + str(np.shape(y_credit_teste)))

    # Utilizar o parametro ramdom_state = 0 para gerar novo árvore
    arvore_risco_credito_treinamento = dtc(random_state=0)

    arvore_risco_credito_treinamento.fit(X_credit_treinamento, y_credit_treinamento)

    # Previsão com a base de testes
    y_previsto = arvore_risco_credito_treinamento.predict(X_credit_teste)

    print('\nDados previstos')
    print(y_previsto)
    print('Dados reais')
    print(y_credit_teste)

    # Calculo da acurácia
    print('\nCálculo da acurácia')
    print(accuracy_score(y_credit_teste, y_previsto))

    # Matriz de confusão
    cm = ConfusionMatrix(arvore_risco_credito_treinamento, classes=['0', '1'])

    cm.fit(X_credit_treinamento, y_credit_treinamento)

    cm.score(X_credit_teste, y_credit_teste)

    cm.show();

    # Análise da Matriz de Confusão
    print('\nAnálise da Matriz de Confusão')
    print('\nQuantos clientes foram classificados corretamente que pagam a dívida? 430')
    print('\nQuantos clientes foram classificados incorretamente como não pagantes? 4')
    print('\nQuantos clientes foram classificados corretamente que não pagam? 60')
    print('\nQuantos clientes foram classificados incorretamente como pagantes? 6')

    # Classificação
    report = classification_report(y_credit_teste, y_previsto, output_dict=True)

    # Parâmetro classification_report
    print('\nClassification Report')
    print(report)

    # Relação entre precision e recall
    print('\n Análise da precisão e sensibilidade(recall)')
    print('\n O recall representa a quantidade de dados identificados corretamente como positivo, e com a precisão sabemos a quantidade dentre os marcados como positivo é realmente positivo')
    print('\n Podemos observar que a classe 0 dos que pagam 0.99 foram identificados corretamente como positivos (430) e 0.99 de precisão então ouveram poucos Falsos Positivos (6)')
    print('\n Podemos observar que a classe 1 dos que não pagam 0.94 foram identificados corretamente como positivos (60) e 0.91 de precisão então ouveram poucos Falsos Positivos (4)')
    print('\n O algoritmo foi melhor identificando os clientes que pagam')
    print('\n')

    # Visualização da árvore
    fn = ['Historia de credito', 'Divida', 'Garantias', 'Renda anual']
    cn = ['Alto', 'Baixo', 'Moderado']

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=3000)

    sklearn.tree.plot_tree(arvore_risco_credito_treinamento, feature_names=fn, class_names=cn, filled=True)

    fig.savefig('imagemTreinamento.png')

    # Algoritmo Random Forest
    rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

    rf.fit(X_credit_treinamento, y_credit_treinamento)

    # Previsão dos dados
    y_pred = rf.predict(X_credit_teste)

    print('\nDados reais')
    print(y_credit_teste)
    print('\nDados previstos')
    print(y_pred)

    # Cálculo da acurácia
    print('\nCálculo da acurácia')
    print(accuracy_score(y_credit_teste, y_pred))
    print('\nO resultado é pior que a árvore simples. Para melhorar os resultados podemos testar com diferentes números para n')

    #Teste com n = 40
    rf = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)

    rf.fit(X_credit_treinamento, y_credit_treinamento)

    # Previsão dos dados
    y_pred = rf.predict(X_credit_teste)

    # Cálculo da acurácia
    print('\nCálculo da acurácia com n = 40')
    print(accuracy_score(y_credit_teste, y_pred))

    # Matriz de confusão
    cm = ConfusionMatrix(rf, classes=['0', '1'])

    cm.fit(X_credit_treinamento, y_credit_treinamento)

    cm.score(X_credit_teste, y_credit_teste)

    cm.show();

    # Análise da Matriz de Confusão
    print('\nAnálise da Matriz de Confusão')
    print('\nQuantos clientes foram classificados corretamente que pagam a dívida? 433')
    print('\nQuantos clientes foram classificados incorretamente como não pagantes? 5')
    print('\nQuantos clientes foram classificados corretamente que não pagam? 59')
    print('\nQuantos clientes foram classificados incorretamente como pagantes? 3')

    # Classificação
    report = classification_report(y_credit_teste, y_pred, output_dict=True)

    # Parâmetro classification_report
    print('\nClassification Report')
    print(report)

    # Relação entre precision e recall
    print('\n Análise da precisão e sensibilidade(recall)')
    print('\n O recall representa a quantidade de dados identificados corretamente como positivo, e com a precisão sabemos a quantidade dentre os marcados como positivo é realmente positivo')
    print('\n Podemos observar que a classe 0 dos que pagam 0.99 foram identificados corretamente como positivos (433) e 0.99 de precisão então ouveram poucos Falsos Positivos (3)')
    print('\n Podemos observar que a classe 1 dos que não pagam 0.92 foram identificados corretamente como positivos (59) e 0.95 de precisão então ouveram poucos Falsos Positivos (5)')
    print('\n O algoritmo continua melhor identificando os clientes que pagam')
    print('\n Podemos identificar também comparando com o algoritmo simples, o melhor resultado da floresta randomica tem resultados um pouco melhores sobre o recall e um pouco pior para precisão sobre a classe 0, logo identifica mais clientes corretamente como pagantes e identificou um pouco mais clientes não corretamente como pagantes')
    print('\n Já verificando a classe 1 podemos ver resultados um pouco piores sobre o recall e melhores sobre a precisão, logo identifica menos clientes corretamente como não pagantes e identificou mais clientes corretamente como não pagantes')