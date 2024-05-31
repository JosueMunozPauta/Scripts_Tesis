# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:20:08 2024

@author: Josué Muñoz Pauta
"""
# Bibliotecas
import pandas as pd
import xarray as xr
import os
import pickle

# Ruta de los archivos
filepath = os.getcwd()

# Carga y partición de datos ABI filtrados (bandas de 8 a 14)
ABI = pd.read_pickle(filepath[:-7] +  r'\Data\ABI\2019_2023-ABI-filtrado.pkl')
ABI=ABI[['CMI_C08', 'CMI_C09','CMI_C10', 'CMI_C11']]
# Carga y partición de datos IMERG filtrados
IMERG = pd.read_pickle(filepath[:-7]+ r'\Data\IMERG\2019_2023-IMERG-filtrado.pkl')

# Prueba: 2023
X_test = ABI.loc['2023']
Y_test = IMERG.loc['2023']

# Resampling X_test and Y_test by month
X_test_resampled = X_test.resample('Y').mean()
Y_test_resampled = Y_test.resample('Y').mean()


# Cargar el modelo desde el archivo pickle
rf_model = pickle.load(open( filepath+'/'+"RF_model_regressor_8_9_10_11.p", "rb" ))
    
predictions = rf_model.predict(X_test_resampled)

# From array of NumPy to Dataset (Predicciones)
predictions = pd.DataFrame(predictions, columns=Y_test_resampled.columns)
predictions.index = Y_test_resampled.index
predictions = predictions.stack(level=[0, 1],dropna=False).reset_index()
predictions=predictions.dropna()
df_predictions = predictions.set_index(['t','lat','lon'])
df_predictions.columns=['Precip']

ds_predictions = xr.Dataset.from_dataframe(df_predictions)
df_predictions.to_pickle(filepath[:-7] + r'\Results\IMERG_predictions_annually-2023.pkl', protocol=3)

# From pivot to Dataset (Observaciones)
obs=Y_test_resampled
obs = obs.stack(level=[0, 1],dropna=False).reset_index()
obs=obs.dropna()
df_obs = obs.set_index(['t','lat','lon'])
df_obs.columns=['Precip']

ds_obs = xr.Dataset.from_dataframe(df_obs)
df_obs.to_pickle(filepath[:-7] + r'\Results\IMERG_observations_annually-2023.pkl', protocol=3)

merged_df = pd.concat([df_obs, df_predictions], axis=0)
