import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping
import pickle
from pyproj import CRS

filepath = r'D:\Josue\IDRHICA\Jubones' 

# Abrir archivo NetCDF de IMERG-ER
ds_imerg = xr.open_dataset(filepath + r'\Data\NetCDF\IMERG\IMERG-ER_Hrly_2019-2023\IMERG-ER_Hrly_2019-2023.nc')
PMA = ds_imerg.sum(dim = 't')


#Change shapefile
jubones_shp = gpd.read_file(filepath + r'\Data\Jubones\jubonesMSF_catch.shp')
jubones_geometry = jubones_shp.geometry.iloc[0]  # Get the boundary geometry
# jubones_shp.plot()  # To plot the shp
# Projecting the shapefile to EPSG: 4326
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs')  # Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326)  # To what CRS is going to be projected
jubones_proj = jubones_shp.to_crs(target_crs)

ds_imerg.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds_imerg.rio.write_crs("epsg:4326", inplace=True)   
final_clip = ds_imerg.rio.clip(jubones_shp.geometry.apply(mapping),jubones_shp.crs,all_touched=True)

# Convertir el xarray.Dataset a un pandas DataFrame
IMERG_recortado = final_clip.to_dataframe()

# Resetear el índice y convertir el dataframe en formato csv
IMERG_recortado = IMERG_recortado.reset_index()
IMERG_recortado["t"] = pd.to_datetime(IMERG_recortado["t"] )
pivot_df = IMERG_recortado.pivot_table(index='t', columns=['lat', 'lon'], values= 'Precip').reset_index()
pivot_df.to_pickle(filepath + r'\Results\2019_2023-IMERG-recortado.pkl', protocol=4)


rectangle_data2 = final_clip.resample(t="1Y").sum()
PMA2 = rectangle_data2.mean(dim = 't')
PMA2['Precip'] = PMA2['Precip'].where(PMA2['Precip'] != 0, np.nan)
PMA2['Precip'].mean()
fig, ax = plt.subplots(figsize=(6,6))
PMA['Precip'].plot(cmap='GnBu')
#  vmin = rectangle_data.min()['Precip'], vmax = rectangle_data.max()['Precip']
plt.title("PMA IMERG-ER 2019-2023")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
jubones_proj.boundary.plot(ax=ax, color='black')
plt.savefig(filepath + r'\Plots\IMERG_2019-2023_shape', dpi=300, bbox_inches='tight')
plt.show()
