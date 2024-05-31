import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping
import pickle
from pyproj import CRS

filepath = r'D:\Josue\IDRHICA\Jubones' 
# Abrir archivo NetCDF de las 16 bandas ABI
ds_abi = xr.open_dataset(filepath + r'\Data\NetCDF\ABI\ABI_Hrly_enhanced.nc')

# Cargar contorno de la cuenca del Jubones
jubones_shp = gpd.read_file(filepath + r'\Data\Jubones\jubonesMSF_catch.shp')
jubones_geometry = jubones_shp.geometry.iloc[0]  # Get the boundary geometry
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs')  # Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326)  # To what CRS is going to be projected
jubones_proj = jubones_shp.to_crs(target_crs)

# Recortar los mapas ABI en función del contorno de la cuenca
ds_abi.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds_abi.rio.write_crs("epsg:4326", inplace=True)   
final_clip = ds_abi.rio.clip(jubones_shp.geometry.apply(mapping),jubones_shp.crs,all_touched=True)

# Seleccionar las bandas utilizadas en el modelo
variables = ['CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C11']

# Convertir el xarray.Dataset a un pandas DataFrame
ABI_recortado = final_clip[variables].to_dataframe()

# Resetear el índice y convertir el dataframe en formato pickle
ABI_recortado = ABI_recortado.reset_index()
ABI_recortado["t"] = pd.to_datetime(ABI_recortado["t"] )
pivot_df = ABI_recortado.pivot_table(index='t', columns=['lat', 'lon'], values= variables).reset_index()
pivot_df.to_pickle(filepath + r'\Results\2019_2023-ABI-recortado.pkl', protocol=4)

# Plot
radiacion_media = ds_abi['CMI_C11'].sum(dim = 't')

fig, ax = plt.subplots(figsize=(6,6))
radiacion_media.plot(cmap='GnBu')
plt.title("ABI RADIACION (CMI_12) 2019-2023")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
jubones_proj.boundary.plot(ax=ax, color='black')
plt.savefig(filepath +r'\Plots\CMI_C12-2019_2023-ABI-shape', dpi=300, bbox_inches='tight')
plt.show()