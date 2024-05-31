# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:19:54 2024

@author: Josue
"""

# Bibliotecas
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import CRS
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Ruta de los archivos
filepath = os.getcwd()
escala='monthly'
# Carga y partición de datos ABI filtrados (bandas de 8 a 14)
df_obs = pd.read_pickle(filepath[:-7] + r'\Results\IMERG_observations_'+str(escala)+'-2023.pkl')

# Carga y partición de datos IMERG filtrados
df_pred = pd.read_pickle(filepath[:-7] + r'\Results\IMERG_predictions_'+str(escala)+'-2023.pkl')

def calculate_precipitation_metrics_pixel(observation_df, prediction_df):
    # Verificar que los dos DataFrames tengan el mismo tamaño
    if observation_df.shape != prediction_df.shape:
        raise ValueError("Los DataFrames de observación y predicción deben tener el mismo tamaño")

    # Crear listas para almacenar las métricas de cada píxel
    cc_list = []

    # Iterar sobre cada píxel
    for col in observation_df.columns:
        # Obtener los valores observados y predichos para este píxel
        obs_pixel = observation_df[col].values
        pred_pixel = prediction_df[col].values

        # Calcular el coeficiente de correlación (CC) para este píxel
        cc_pixel = np.corrcoef(obs_pixel, pred_pixel)[0, 1]
        
        # Agregar las métricas a las listas
        cc_list.append(cc_pixel)
        
    return cc_list

# Horario y Diario
cc_list=[]
for tiempo in range(1,12):
    # Comparacion de precipitación acumulada (mean, max y min) IMERG vs Pronóstico
    observation_df = df_obs.reset_index().set_index('t').loc['2023-'+str(tiempo+1)].pivot_table(index='t', columns=['lat', 'lon'], values= 'Precip')
    prediction_df = df_pred.reset_index().set_index('t').loc['2023-'+str(tiempo+1)].pivot_table(index='t', columns=['lat', 'lon'], values= 'Precip')
    cc_mes = calculate_precipitation_metrics_pixel(observation_df, prediction_df)
    cc_list.append(cc_mes)
cc_invertida = [[lista[i] for lista in  cc_list] for i in range(45)]
cc_mean = []
for lista in cc_invertida:
    promedio_lista = sum(lista) / len(lista)
    cc_mean.append(promedio_lista)

       
#  Mensual
mes=[1,2,3,4,5,6,7,8,9,10,11,12]
observation_df=df_obs[df_obs.reset_index().set_index('t').index.month.isin(mes)]
prediction_df=df_pred[df_pred.reset_index().set_index('t').index.month.isin(mes)]

observation_df = observation_df.reset_index().set_index('t').pivot_table(index='t', columns=['lat', 'lon'], values= 'Precip')
prediction_df = prediction_df.reset_index().set_index('t').pivot_table(index='t', columns=['lat', 'lon'], values= 'Precip')
cc_mean = calculate_precipitation_metrics_pixel(observation_df, prediction_df)
        

general_mean_cc = np.mean(cc_mean)
resultado_cc = df_obs.reset_index()[['lat', 'lon']].iloc[:45]
resultado_cc['PCC'] = cc_mean
resultado_cc=resultado_cc.set_index(['lat','lon'])
ds_resultado_cc = xr.Dataset.from_dataframe(resultado_cc)

# Mapa de la cuenca
#Change shapefile
jubones_shp = gpd.read_file(filepath[:-7]+ r'\Data\Jubones\jubonesMSF_catch.shp')
jubones_geometry = jubones_shp.geometry.iloc[0]  # Get the boundary geometry
# Projecting the shapefile to EPSG: 4326
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs')  # Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326)  # To what CRS is going to be projected
jubones_proj = jubones_shp.to_crs(target_crs)

fig, ax = plt.subplots(figsize=(6,6))
level=np.arange(-0.5,1,0.05)
# level=np.arange(0.2,1,0.01)
im2=ds_resultado_cc['PCC'].plot(cmap='darkmint',levels=level)
plt.title("Pearson Correlation Coefficient in "+str(escala)+" scale for 2023")
# plt.title("Pearson Correlation Coefficient for Dry Season in 2023")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
jubones_proj.boundary.plot(ax=ax, color='black')
plt.text(0.76, 0.96, 'Media: {:.2f}'.format(general_mean_cc), transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
ax.collections[0].colorbar.remove()

# Agregar una única barra de colores para ambos subplots
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im2, cax=cax)
cbar.set_label('PCC')
plt.tight_layout()
# plt.savefig((filepath[:-7] + r'\Plots\CC_'+str(escala)+'-Acum2023.tiff'),format='tiff',dpi=300) 
# plt.savefig((filepath[:-7] + r'\Plots\CC_Dry-Acum2023.tiff'),format='tiff',dpi=300) 
plt.show()
