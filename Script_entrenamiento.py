# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:09:04 2024

@author: Josué Muñoz Pauta
"""

# Bibliotecas
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor


# Ruta de los archivos
filepath = os.getcwd()

# Carga y partición de datos ABI filtrados (bandas de 8 a 11)
ABI = pd.read_pickle(filepath[:-7] +  r'\Data\ABI\2019_2023-ABI-filtrado.pkl')
ABI=ABI[['CMI_C08', 'CMI_C09','CMI_C10', 'CMI_C11']]

# Carga y partición de datos IMERG filtrados
IMERG = pd.read_pickle(filepath[:-7]+ r'\Data\IMERG\2019_2023-IMERG-filtrado.pkl')

# Divide los datos en conjuntos de entrenamiento y prueba
# Entrenamiento: 2019 a 2022
X_train = ABI.loc['2019':'2022']
Y_train = IMERG.loc['2019':'2022']

# Definición de los hiperparámetros
n_estimators = 100  # Número de árboles en el bosque
max_features = None  # Número máximo de características a considerar en cada división
min_samples_split = 20  # Número mínimo de muestras necesarias para dividir un nodo interno
min_samples_leaf = 20  # Número mínimo de muestras necesarias para estar en un nodo hoja
max_depth = None  # Profundidad máxima del árbol

# Creación del modelo con los hiperparámetros especificados
rf_model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_features=max_features,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  max_depth=max_depth,
                                  random_state=42)
# Entrenamiento del modelo
rf_model.fit(X_train, Y_train)
pickle.dump(rf_model, open( filepath+'/'+"RF_model.p", "wb" ))