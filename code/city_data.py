""" city_data.py
City data to extracted from the following sources and saved in csv files:
    https://worldclim.org/ (for each city, extract the mean BioX across all cells)
    http://www.paleoclim.org/ (for each city, extract the mean BioX across all cells)
    https://landscan.ornl.gov/ (for each city, first, extract the total value (sum) across all cells in 2000 and 2021; second, estimate the absolute change
"""
import extract_tif_data as etd
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import time

# Reading in the example .shp file with geopandas.
# Data obtained from https://catalog.data.gov/dataset/500-cities-city-boundaries
gdf = gpd.read_file('..\\..\\city-boundaries\\ne_10m_urban_areas_landscan.shp')
gdf = gdf.to_crs('epsg:4326') # converts to lat/long coords


cities = pd.DataFrame({'City' : gdf['name_conve']})
data_path = '..\\..\\datasets\\'
os.makedirs(data_path + 'csv_data', exist_ok=True)
# get data for all cities.
start = time.time()
# df = etd.worldclimCityData(gdf)
# df = pd.merge(cities,df,left_index=True,right_index=True)
# df.to_csv(data_path + 'csv_data\\worldclim_cities.csv',index=False)
wctime = time.time()
print("WorldClim datset to csv complete: " + str(wctime-start) + " seconds\n")

df = etd.paleoclimCityData(gdf)
df = pd.merge(cities,df,left_index=True,right_index=True)
df.to_csv(data_path + 'csv_data\\paleoclim_cities.csv',index=False)
pctime = time.time()
print("PaleoClim datset to csv complete: " + str(pctime-wctime) + " seconds\n")

df = etd.landscanCityData(gdf)
df = pd.merge(cities,df,left_index=True,right_index=True)
df.to_csv(data_path + 'csv_data\\landscan_cities.csv',index=False)
lstime = time.time()
print("Landscan datset to csv complete: " + str(lstime-pctime) + " seconds\n")