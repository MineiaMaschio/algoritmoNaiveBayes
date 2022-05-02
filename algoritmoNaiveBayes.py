from operator import xor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
  #Ler aquivo de dados
  data = pd.read_csv("dataset_risco_credito.csv", header=None)

  #Imprimit base de dados
  print('\nDataset')
  print(data)
  
  #Separar base em variaveis e classe
  X_risco_credito = data.iloc[:,0:4].values
  y_risco_credito = data.iloc[:,4].values

  #Aplicar laber enconder
  label_encoder_historia = LabelEncoder()
  label_encoder_divida = LabelEncoder()
  label_encoder_garantias = LabelEncoder()
  label_encoder_renda = LabelEncoder()
  label_encoder_risco = LabelEncoder()

  X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0])
  X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1])
  X_risco_credito[:,2] = label_encoder_garantias.fit_transform(X_risco_credito[:,2])
  X_risco_credito[:,3] = label_encoder_renda.fit_transform(X_risco_credito[:,3])
  y_risco_credito = label_encoder_risco.fit_transform(y_risco_credito)

  #Printar resultado do laber enconder
  print('\nResultado do label enconder')
  print(X_risco_credito[:,0])
  print(X_risco_credito[:,1])
  print(X_risco_credito[:,2])
  print(X_risco_credito[:,3])
  print(y_risco_credito)

  #Salvar arquivo pré-processado
  with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)

  with open('risco_credito.pkl', 'rb') as f:
    dataPkl = pickle.load(f)

  #Printar resultado do arquivo pré-processado
  print('\nResultado do arquivo pré-processado')
  print(dataPkl)

  # Criar o objeto Nayve Bayes
  naiveb_risco_credito = GaussianNB()

  #Treina o modelo
  naiveb_risco_credito.fit(X_risco_credito, y_risco_credito)

  #Previsões
  print('\nPrevisão história boa, dívida alta, garantia nenhuma, renda > 35')
  previsao1 = naiveb_risco_credito.predict([[0, 0, 2, 2]])
  print(previsao1)

  print('\nPrevisão história ruim, dívida alta, garantia adequada, renda < 15')
  previsao2 = naiveb_risco_credito.predict([[3, 0, 0, 0]])
  print(previsao2)

  #Printar atibuto .classes
  print('\nClasses utilizadas pelo algoritmo')
  print(naiveb_risco_credito.classes_)

  # Printar atibuto .class_count_
  print('\nQuantidade de registros em cada classe')
  print(naiveb_risco_credito.class_count_)


