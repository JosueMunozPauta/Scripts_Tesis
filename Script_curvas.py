# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:15:20 2024

@author: Josué Muñoz Pauta
"""
# Bibliotecas
import pandas as pd
import xarray as xr
import pickle
import os
import matplotlib.pyplot as plt


# Ruta de los archivos
filepath = os.getcwd()

# Carga y partición de datos ABI filtrados (bandas de 8 a 14)
df_obs = pd.read_pickle(filepath[:-7] + r'\Results\IMERG_observations_hourly-2023.pkl')
ds_obs = xr.Dataset.from_dataframe(df_obs)

# Carga y partición de datos IMERG filtrados
df_pred = pd.read_pickle(filepath[:-7] + r'\Results\IMERG_predictions_hourly-2023.pkl')
ds_pred = xr.Dataset.from_dataframe(df_pred)


# Gráfica anual
for tiempo in range(0,1):     
    # Comparación precipitación media IMERG vs Pronóstico
    df_media_obs = df_obs.reset_index().set_index('t').drop(['lat','lon'], axis=1).resample('H').mean()
    df_media_pred = df_pred.reset_index().set_index('t').drop(['lat','lon'], axis=1).resample('H').mean()
    fig, ax = plt.subplots(figsize=(12,4))
    plt.plot(df_media_obs['Precip'], label='Observado')
    plt.plot(df_media_pred['Precip'], label='Pronosticado')
    plt.legend()
    plt.ylabel('Precip $(mm)$')
    fecha_inicio = df_media_obs.index.min()
    fecha_fin = df_media_obs.index.max()
    plt.xlim(fecha_inicio, fecha_fin)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig((filepath[:-7]+ '\Plots\MEDIA_OBS-PRED-2023.tiff'),format='tiff',dpi=300) 
    plt.show()


# Crear una figura para todas las subfiguras
fig, axs = plt.subplots(4, 3, figsize=(12, 16))
mes = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'] 
# Iterar sobre los meses
for tiempo in range(12):
    # Comparacion de precipitación acumulada (mean, max y min) IMERG vs Pronóstico
    df_acum_media_obs = df_obs.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').mean().cumsum()
    df_acum_media_pred = df_pred.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').mean().cumsum()
    df_acum_max_obs = df_obs.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').max().cumsum()
    df_acum_max_pred = df_pred.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').max().cumsum()
    df_acum_min_obs = df_obs.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').min().cumsum()
    df_acum_min_pred = df_pred.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').min().cumsum()
    
    # Seleccionar el subgráfico correspondiente
    row = tiempo // 3
    col= tiempo % 3
    ax = axs[row, col]
    
    # Trazar los datos en el subgráfico
    ax.plot(df_acum_media_obs.index.day, df_acum_media_obs['Precip'], label='IMERG Mean Cumulative', color='blue')
    ax.plot(df_acum_media_pred.index.day, df_acum_media_pred['Precip'], label='Predicted Mean Cumulative', color='blue', linestyle='--')
    ax.plot(df_acum_max_obs.index.day, df_acum_max_obs['Precip'], label='IMERG Max Cumulative', color='red')
    ax.plot(df_acum_max_pred.index.day, df_acum_max_pred['Precip'], label='Predicted Max Cumulative', color='red', linestyle='--')
    ax.plot(df_acum_min_obs.index.day, df_acum_min_obs['Precip'], label='IMERG Min Cumulative', color='green')
    ax.plot(df_acum_min_pred.index.day, df_acum_min_pred['Precip'], label='Predicted Min Cumulative', color='green', linestyle='--')
    
    ax.set_title(mes[tiempo])
    ax.set_ylabel('Precip Acum $(mm)$')
    ax.grid(True)
plt.legend()

# Ajustar el diseño de las subfiguras
plt.tight_layout()

# Guardar la figura
plt.savefig((filepath[:-7]+ '\Plots\ACUM_Monthly_OBS-PRED-2023.tiff'), format='tiff', dpi=300)

# Mostrar la figura
plt.show()


# Crear una figura para todas las subfiguras
fig, axs = plt.subplots(4, 3, figsize=(12, 16))
mes = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'] 
# Iterar sobre los meses
for tiempo in range(12):
    # Comparacion de precipitación acumulada (mean, max y min) IMERG vs Pronóstico
    df_acum_media_obs = df_obs.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').mean().cumsum()
    df_acum_media_pred = df_pred.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').mean().cumsum()
    df_acum_max_obs = df_obs.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').max().cumsum()
    df_acum_max_pred = df_pred.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').max().cumsum()
    df_acum_min_obs = df_obs.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').min().cumsum()
    df_acum_min_pred = df_pred.reset_index().set_index('t').drop(['lat', 'lon'], axis=1).loc['2023-'+str(tiempo+1)].resample('H').min().cumsum()
    
    # Seleccionar el subgráfico correspondiente
    row = tiempo // 3
    col= tiempo % 3
    ax = axs[row, col]
    
    # Trazar los datos en el subgráfico
    ax.plot(df_acum_media_obs.index.day, df_acum_media_obs['Precip'], label='IMERG Mean Cumulative', color='blue')
    ax.plot(df_acum_media_pred.index.day, df_acum_media_pred['Precip'], label='Predicted Mean Cumulative', color='blue', linestyle='--')
    ax.plot(df_acum_max_obs.index.day, df_acum_max_obs['Precip'], label='IMERG Max Cumulative', color='red')
    ax.plot(df_acum_max_pred.index.day, df_acum_max_pred['Precip'], label='Predicted Max Cumulative', color='red', linestyle='--')
    ax.plot(df_acum_min_obs.index.day, df_acum_min_obs['Precip'], label='IMERG Min Cumulative', color='green')
    ax.plot(df_acum_min_pred.index.day, df_acum_min_pred['Precip'], label='Predicted Min Cumulative', color='green', linestyle='--')
    
    ax.set_title(mes[tiempo])
    ax.set_ylabel('Precip Acum $(mm)$')
    ax.grid(True)
plt.legend()

# Ajustar el diseño de las subfiguras
plt.tight_layout()

# Guardar la figura
plt.savefig((filepath[:-7]+ '\Plots\ACUM_Monthly_OBS-PRED-2023.tiff'), format='tiff', dpi=300)

# Mostrar la figura
plt.show()



# Comparacion de precipitación acumulada (mean, max and min) IMERG vs Pronóstico
df_acum_media_obs = df_obs.reset_index().set_index('t').drop(['lat','lon'], axis=1).resample('H').mean().cumsum()
df_acum_media_pred = df_pred.reset_index().set_index('t').drop(['lat','lon'], axis=1).resample('H').mean().cumsum()
df_acum_max_obs = df_obs.reset_index().set_index('t').drop(['lat','lon'], axis=1).resample('H').max().cumsum()
df_acum_max_pred = df_pred.reset_index().set_index('t').drop(['lat','lon'], axis=1).resample('H').max().cumsum()
df_acum_min_obs = df_obs.reset_index().set_index('t').drop(['lat','lon'], axis=1).resample('H').min().cumsum()
df_acum_min_pred = df_pred.reset_index().set_index('t').drop(['lat','lon'], axis=1).resample('H').min().cumsum()

fig, ax = plt.subplots(figsize=(6,10))
plt.plot(df_acum_media_obs['Precip'], label='IMERG Mean Cumulative',color='blue')
plt.plot(df_acum_media_pred['Precip'], label='Predicted Mean Cumulative',color='blue', linestyle='--')
plt.plot(df_acum_max_obs['Precip'], label='IMERG Max Cumulative',color='red')
plt.plot(df_acum_max_pred['Precip'], label='Predicted Max Cumulative',color='red', linestyle='--')
plt.plot(df_acum_min_obs['Precip'], label='IMERG Min Cumulative',color='green')
plt.plot(df_acum_min_pred['Precip'], label='Predicted Min Cumulative',color='green', linestyle='--')
plt.legend()
plt.ylabel('Precip Acum $(mm)$')
fecha_inicio = df_acum_media_obs.index.min()
fecha_fin = df_acum_media_obs.index.max()
plt.xlim(fecha_inicio, fecha_fin)
plt.grid(True)
plt.tight_layout()
plt.savefig((filepath[:-7]+ '\Plots\ACUM_OBS-PRED-2023.tiff'),format='tiff',dpi=300) 
plt.show()