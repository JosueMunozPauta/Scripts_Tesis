# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:08:00 2024

@author: Josue
"""

# Bibliotecas
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import CRS
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Ruta de los archivos
filepath = os.getcwd()
escala='daily'
# Carga y partición de datos ABI filtrados (bandas de 8 a 14)
df_obs = pd.read_pickle(filepath[:-7] + r'\Results\IMERG_observations_'+str(escala)+'-2023.pkl')

# Carga y partición de datos IMERG filtrados
df_pred = pd.read_pickle(filepath[:-7] + r'\Results\IMERG_predictions_'+str(escala)+'-2023.pkl')

def calculate_precipitation_metrics_pixel(observation_df, prediction_df):
    # Verificar que los dos DataFrames tengan el mismo tamaño
    if observation_df.shape != prediction_df.shape:
        raise ValueError("Los DataFrames de observación y predicción deben tener el mismo tamaño")

    # Crear listas para almacenar las métricas de cada píxel
    rmse_list = []

    # Iterar sobre cada píxel
    for col in observation_df.columns:
        # Obtener los valores observados y predichos para este píxel
        obs_pixel = observation_df[col].values
        pred_pixel = prediction_df[col].values

        # Calcular el RMSE (Root Mean Squared Error) para este píxel
        rmse_pixel = np.sqrt(mean_squared_error(obs_pixel, pred_pixel))
        
        # Agregar las métricas a las listas
        rmse_list.append(rmse_pixel)

        
    return rmse_list

# Horario y Diario
rmse_list=[]
for tiempo in range(6,12):
    # Comparacion de precipitación acumulada (mean, max y min) IMERG vs Pronóstico
    observation_df = df_obs.reset_index().set_index('t').loc['2023-'+str(tiempo+1)].pivot_table(index='t', columns=['lat', 'lon'], values= 'Precip')
    prediction_df = df_pred.reset_index().set_index('t').loc['2023-'+str(tiempo+1)].pivot_table(index='t', columns=['lat', 'lon'], values= 'Precip')
    rmse_mes = calculate_precipitation_metrics_pixel(observation_df, prediction_df)
    rmse_list.append(rmse_mes)
rmse_invertida = [[lista[i] for lista in  rmse_list] for i in range(45)]
rmse_mean = []
for lista in rmse_invertida:
    promedio_lista = sum(lista) / len(lista)
    rmse_mean.append(promedio_lista)

        
#  Mensual
observation_df = df_obs.reset_index().set_index('t').pivot_table(index='t', columns=['lat', 'lon'], values= 'Precip')
prediction_df = df_pred.reset_index().set_index('t').pivot_table(index='t', columns=['lat', 'lon'], values= 'Precip')
rmse_mean = calculate_precipitation_metrics_pixel(observation_df, prediction_df)
        

general_mean_rmse = np.mean(rmse_mean)
resultado_rmse = df_obs.reset_index()[['lat', 'lon']].iloc[:45]
resultado_rmse['RMSE'] = rmse_mean
resultado_rmse=resultado_rmse.set_index(['lat','lon'])
ds_resultado_rmse = xr.Dataset.from_dataframe(resultado_rmse)

# Mapa de la cuenca
#Change shapefile
jubones_shp = gpd.read_file(filepath[:-7]+ r'\Data\Jubones\jubonesMSF_catch.shp')
jubones_geometry = jubones_shp.geometry.iloc[0]  # Get the boundary geometry
# Projecting the shapefile to EPSG: 4326
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs')  # Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326)  # To what CRS is going to be projected
jubones_proj = jubones_shp.to_crs(target_crs)

fig, ax = plt.subplots(figsize=(6,6))
level=np.arange(0,0.4,0.05)
im1=ds_resultado_rmse['RMSE'].plot(cmap='YlOrRd',levels=level)
# plt.title("Root Mean Squared Error in "+str(escala)+" scale for 2023")
plt.title("Root Mean Squared Error for Dry Season in 2023")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
jubones_proj.boundary.plot(ax=ax, color='black')
plt.text(0.76, 0.96, 'Media: {:.2f}'.format(general_mean_rmse), transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
ax.collections[0].colorbar.remove()

# Agregar una única barra de colores para ambos subplots
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im1, cax=cax)
cbar.set_label('RMSE')
plt.tight_layout()
# plt.savefig((filepath[:-7] + r'\Plots\RMSE_'+str(escala)+'-Acum2023.tiff'),format='tiff',dpi=300) 
plt.savefig((filepath[:-7] + r'\Plots\RMSE_Dry-Acum2023.tiff'),format='tiff',dpi=300) 
plt.show()
