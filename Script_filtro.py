# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:41:29 2024

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
df_ABI = pd.read_pickle(filepath[:-7] +  r'\Data\ABI\2019_2023-ABI-recortado.pkl')
df_ABI["t"] = pd.to_datetime(df_ABI["t"] )
df_ABI.set_index('t', inplace=True)
df_ABI=df_ABI.dropna()
# Carga y partición de datos IMERG filtrados
df_IMERG = pd.read_pickle(filepath[:-7]+ r'\Data\IMERG\2019_2023-IMERG-recortado.pkl')
df_IMERG["t"] = pd.to_datetime(df_IMERG["t"] )
df_IMERG.set_index('t', inplace=True)

# Filtro
indices_df_ABI = set(df_ABI.index)
indices_df_IMERG = set(df_IMERG.index)

# Encontrar los índices que están presentes en df1 pero no en df2, y viceversa
indices_solo_en_df_ABI = indices_df_ABI - indices_df_IMERG
indices_solo_en_df_IMERG = indices_df_IMERG - indices_df_ABI

# Dataframes filtrados para el modelo
ABI = df_ABI.drop(indices_solo_en_df_ABI)
IMERG = df_IMERG.drop(indices_solo_en_df_IMERG)

# Descarga de dataframes filtrados
ABI.to_pickle(filepath[:-7]+ r'\Data\ABI\2019_2023-ABI-filtrado.pkl',protocol=3)
IMERG.to_pickle(filepath[:-7]+ r'\Data\IMERG\2019_2023-IMERG-filtrado.pkl',protocol=3)

del indices_solo_en_df_ABI, indices_solo_en_df_IMERG
del indices_df_ABI, indices_df_IMERG
del df_ABI, df_IMERG         



