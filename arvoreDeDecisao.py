# -*- coding: utf-8 -*-

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

  #Calcular árvore de decisão utilizando como critério a entropia
  arvore_risco_credito = dtc(criterion='entropy')

  arvore_risco_credito.fit(X_risco_credito, y_risco_credito)

  #Utilizar o feature_importances_ para retornar a importância de cada atributo
  print(arvore_risco_credito.feature_importances_)
  print('O maior ganho é do')

  #Gerar uma visualização da árvore de decisão
  fn = ['Historia de credito', 'Divida', 'Garantias', 'Renda anual' ]
  cn = ['Alto', 'Baixo', 'Moderado']

  fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=3000)

  sklearn.tree.plot_tree(arvore_risco_credito, feature_names=fn, class_names=cn, filled=True)

  fig.savefig('imagem.png')

  #Previsões
  print('\nPrevisão história boa, dívida alta, garantia nenhuma, renda > 35')
  previsao1 = arvore_risco_credito.predict([[0, 0, 1, 2]]);
  print(previsao1)

  print('\nPrevisão história ruim, dívida alta, garantia adequada, renda < 15')
  previsao2 = arvore_risco_credito.predict([[2, 0, 0, 0]]);
  print(previsao2)

  #Abrir arquivo
  with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

  #Verificar tamanho das bases
  print('\nTamanho base treinamento')
  print('x =' + str(np.shape(X_credit_treinamento)))
  print('y =' + str(np.shape(y_credit_treinamento)))

  print('\nTamanho base de teste')
  print('x =' + str(np.shape(X_credit_teste)))
  print('y =' + str(np.shape(y_credit_teste)))

  #Utilizar o parametro ramdom_state = 0 para gerar novo árvore
  arvore_risco_credito_treinamento = dtc(random_state = 0)

  arvore_risco_credito_treinamento.fit(X_credit_treinamento, y_credit_treinamento)
  
  #Previsão com a base de testes
  y_previsto = arvore_risco_credito_treinamento.predict(X_credit_teste)

  print('\nDados previstos')
  print(y_previsto)
  print('Dados reais')
  print(y_credit_teste)

  #Calculo da acurácia
  print('\nCálculo da acurácia')
  print(accuracy_score(y_credit_teste, y_previsto))

  #Classificação
  report = classification_report(y_credit_teste, y_previsto, output_dict = True) 

  #Análise da Matriz de Confusão
  print('\nQuantos clientes foram classificados corretamente que pagam a dívida?' + str(report['1']['recall'] ) )  
  print('\nQuantos clientes foram classificados incorretamente como não pagantes? FP ')
  print('\nQuantos clientes foram classificados corretamente que não pagam? VP')
  print('\nQuantos clientes foram classificados incorretamente como pagantes? FP')

  #Matriz de confusão
  cm = ConfusionMatrix(arvore_risco_credito_treinamento)
  cm.fit(X_credit_treinamento, y_credit_treinamento)
  cm.score(X_credit_teste, y_credit_teste)

  #Parâmetro classification_report
  print('\nClassification Report')
  print(report)

  #Relação entre precision e recall
  print('\nNa precissão os Falsos Positivos são considerado mais prejudiciais que os Falsos Negativos, logo ao classificar algo como positivo o modelo deve estar correto, mesmo que acabe classificando alguns como Falso Negativo')
  print('\nNo recall os Falsos Negativos são considerados mais prejudiciais que os Falsos Positivos, logo ao classificar algo deve encontrar por exemplo todos os doentes  mesmo que classifique saudáveis como doentes (Falso Positivo)')
  print('\nNa classificação da nossa árvore podemos observar que a precisão e o recall da classe 0 que são os clientes que pagam é melhor que a classe 1, significa que é um pouco pior que dizer quais clientes não pagam a dívida')

  #Visualização da árvore
  fn = ['Historia de credito', 'Divida', 'Garantias', 'Renda anual' ]
  cn = ['Alto', 'Baixo', 'Moderado']

  fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=3000)

  sklearn.tree.plot_tree(arvore_risco_credito_treinamento, feature_names=fn, class_names=cn, filled=True)

  fig.savefig('imagemTreinamento.png')

  # Algoritmo Random Forest
  rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state = 0)

  rf.fit(X_credit_treinamento, y_credit_treinamento)

  #Previsão dos dados 
  y_pred = rf.predict(X_credit_teste)

  print('\nDados reais')
  print(y_credit_teste)
  print('\nDados previstos')
  print(y_pred)

  #Cálculo da acurácia
  print('\nCálculo da acurácia')
  print(accuracy_score(y_credit_teste, y_pred))
  print('\n O resultado é pior que a árvore simples')


"""
e) Se o resultado foi inferior, como você poderia resolver isso? Quais foram os resultados obtidos?
Aqui se faz o teste com:
40 árvores - o melhor resultado 0.984
mas pode fazer com 100 (default), com 50 e 70.

f) Faça a análise da Matriz de Confusão.

g) Faça um print do parâmetro classification_report entre os dados de teste e as previsões. Explique qual é a relação entre precision e recall nos dados. Como você interpreta esses dados?
"""
