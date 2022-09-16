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

# Reading in the example .shp file with geopandas.
# Data obtained from https://catalog.data.gov/dataset/500-cities-city-boundaries
gdf = gpd.read_file('..\\..\\city-boundaries\\ne_10m_urban_areas_landscan.shp')
gdf = gdf.to_crs('epsg:4326') # converts to lat/long coords
# x1, y1 = gdf.iloc[400].geometry.exterior.xy # defined boundaries for city
# plt.plot(x1,y1) # can plot the boundaries
# plt.show()
# gdf.iloc[400].geometry.contains(Point(-0.06+-1.182e2, 33.84))
# tells if a point is in a city.
# some elements are "MultiPolygons", that have multiple Polygons.

cities = pd.DataFrame({'City' : gdf['name_conve']})
data_path = '..\\..\\datasets\\'

# get data for all cities.
df = etd.worldclimCityData(gdf.geometry)
df = pd.merge(cities,df,left_index=True,right_index=True)
df.to_csv(data_path + 'csv_data\\worldclim_cities.csv',index=False)

df = etd.paleoclimCityData(gdf.geometry)
df = pd.merge(cities,df,left_index=True,right_index=True)
df.to_csv(data_path + 'csv_data\\paleoclim_cities.csv',index=False)

df = etd.landscanCityData(gdf.geometry)
df = pd.merge(cities,df,left_index=True,right_index=True)
df.to_csv(data_path + 'csv_data\\landscan_cities.csv',index=False)