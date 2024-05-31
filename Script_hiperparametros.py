# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:17:07 2024

@author: Josué Muñoz Pauta
"""

# Bibliotecas
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import time

# Ruta de los archivos
filepath = os.getcwd()

# Carga y partición de datos ABI filtrados (bandas de 8 a 14)
ABI = pd.read_pickle(filepath[:-7] +  r'\Data\ABI\2019_2023-ABI-filtrado.pkl')
# Carga y partición de datos IMERG filtrados
IMERG = pd.read_pickle(filepath[:-7]+ r'\Data\IMERG\2019_2023-IMERG-filtrado.pkl')

# Divide los datos en conjuntos de entrenamiento y prueba
# Entrenamiento: 2019 a 2022
X_train = ABI.loc['2019':'2022']
Y_train = IMERG.loc['2019':'2022']

def RF_hypertuning(folder_name, max_trees):
    # Shift all columns except the last one by four time steps

                                                         #Random Hyperparameter Grid
#To use RandomizedSearchCV, we first need to create a parameter grid
#to sample from during fitting:
# Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = max_trees, num = 1)]
# Number of features to consider at every split
    max_features = ['sqrt', 'log2', None]
# Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(start = 10, stop = max_trees, num = 8)]
    max_depth.append(None)
# Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10,20]
# Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5, 10, 20]
    
    warm_start=[True]
    
    oob_score=[True]
    
    n_jobs=[-1]
# Method of selecting samples for training each tree
    bootstrap = [True]
# Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'oob_score': oob_score,
               'warm_start': warm_start,
               'n_jobs': n_jobs,
               'bootstrap': bootstrap}
 # Use the random grid to search for best hyperparameters
# First create the base model to tune
    rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 1, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
    rf_random.fit(X_train, Y_train)
    best_parameters = rf_random.best_params_
    
    filepath=os.getcwd()
    output_folder_figures = filepath + '/'+folder_name
    
    try:
        os.makedirs(output_folder_figures)
    except OSError:
        if not os.path.isdir(output_folder_figures):
            raise
            
    f=open(os.path.join(output_folder_figures, 'Hypertuning.txt'),'w')
    f.write(str(random_grid))
    f.write(str(best_parameters)) 
    f.close()
    return best_parameters


max_trees = 1000                                                
forecast_horizon = 'Best_parameters' 
start = time.time()
# run your code
best_parameters = RF_hypertuning(folder_name='Best_hyperparameters',max_trees=max_trees)
folder_hyperparameters = filepath   + '/Best_hyperparameters'   
pickle.dump(best_parameters, open( folder_hyperparameters+'/'+"Best_parameters.p", "wb" ) )
end = time.time()
elapsed = end - start
print("Computation time =",elapsed,'seconds')


