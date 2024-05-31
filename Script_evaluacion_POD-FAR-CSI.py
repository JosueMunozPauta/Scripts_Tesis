# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:51:33 2024

@author: Josué Muñoz Pauta
"""
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ruta de los archivos
filepath = os.getcwd()

escala='hourly'
# Carga y partición de datos ABI filtrados (bandas de 8 a 14)
df_obs = pd.read_pickle(filepath[:-7] + r'\Results\IMERG_observations_'+str(escala)+'-2023.pkl')
df_obs = df_obs.reset_index()
# Carga y partición de datos IMERG filtrados
df_pred = pd.read_pickle(filepath[:-7] + r'\Results\IMERG_predictions_'+str(escala)+'-2023.pkl')
df_pred = df_pred.reset_index()

# Define el umbral para clasificar la precipitación (fuera del bucle principal)
umbral = 0.01
    
def calculate_metrics(observed, predicted):
    # Calculate confusion matrix
    TN, FP, FN, TP = confusion_matrix(observed, predicted, labels=[0,1]).ravel()

    # Calculate metrics
    POD = TP / (TP + FN) if (TP + FN) > 0 else None
    FAR = FP / (FP + TP) if (FP + TP) > 0 else None
    CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else None

    # Return metrics as dictionary
    return {'POD': POD, 'FAR': FAR, 'CSI': CSI, 'Umbral': umbral}

# Group dataframes by date and calculate metrics for each day
results = []
for date, obs_group in df_obs.groupby('t'):
    pred_group = df_pred[df_pred['t'] == date]
    
    observed_binary = (round(obs_group['Precip'], 2) > umbral).astype(int).tolist()
    predicted_binary = (round(pred_group['Precip'], 2) > umbral).astype(int).tolist()
    
    metrics = calculate_metrics(observed_binary, predicted_binary)
    results.append({'t': date, **metrics})

# Convert results to dataframe
metrics_df = pd.DataFrame(results)
metrics_df = metrics_df.set_index('t').mean()
metrics_df.to_csv(filepath[:-7] + r'\Results\POD-FAR-CSI_'+str(escala)+'-2023.csv')


# Extract month from the date column
metrics_df['Month'] = pd.to_datetime(metrics_df['t']).dt.month
metrics_df=metrics_df.set_index('t').resample('D').mean()
# metrics_df.to_csv(filepath[:-7] + r'\Results\Metricas_POD-FAR-CSI_mensual-2023.csv')


# Define grayscale colors for each metric
colors = {'POD': '#AAAAAA', 'FAR': '#666666', 'CSI': '#333333'}

# Define figure and subplots
fig, axes = plt.subplots(3, 1, figsize=(16, 8))

# Plot metrics as boxplots by month
for i, metric in enumerate(['POD', 'FAR', 'CSI']):
    ax = axes[i]
    boxplot = ax.boxplot([metrics_df[metrics_df['Month'] == month][metric].dropna() for month in range(1, 13)], patch_artist=True)
    for box in boxplot['boxes']:
        # Set facecolor of the boxplot to grayscale
        box.set_facecolor(colors[metric])
    # ax.set_title(metric + ' by Month')
    ax.set_ylabel(metric)
    ax.set_xticks([])
    ax.grid(True, axis='y')

# Set xticks for the last subplot
axes[-1].set_xticks(range(1, 13))
axes[-1].set_xticklabels([pd.to_datetime(str(i), format='%m').strftime('%B') for i in range(1, 13)], rotation=45)

plt.tight_layout()
plt.savefig((filepath[:-7]+ '\Plots\Boxplot_010-POD_FAR_CSI-2023.tiff'),format='tiff',dpi=300) 
plt.show()