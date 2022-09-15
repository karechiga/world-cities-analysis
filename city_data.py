""" city_data.py
City data to extracted from the following sources and saved in csv files:
    https://worldclim.org/ (for each city, extract the mean BioX across all cells)
    http://www.paleoclim.org/ (for each city, extract the mean BioX across all cells)
    https://landscan.ornl.gov/ (for each city, first, extract the total value (sum) across all cells in 2000 and 2021; second, estimate the absolute change
"""
import extract_tif_data as etd
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Reading in the example .shp file with geopandas.
# Data obtained from https://catalog.data.gov/dataset/500-cities-city-boundaries
gdf = gpd.read_file('..\example_data\CityBoundaries.shp')
gdf = gdf.to_crs('epsg:4326') # converts to lat/long coords
# x1, y1 = gdf.iloc[400].geometry.exterior.xy # defined boundaries for city
# plt.plot(x1,y1) # can plot the boundaries
# plt.show()
# gdf.iloc[400].geometry.contains(Point(-0.06+-1.182e2, 33.84))
# tells if a point is in a city.
# some elements are "MultiPolygons", that have multiple Polygons.

cities = pd.DataFrame({'City' : gdf['NAME']})

# get data for all cities.
etd.worldclimCityData(gdf.geometry)
etd.paleoclimCityData(gdf.geometry)
etd.landscanCityData(gdf.geometry)

city_bounds = 0