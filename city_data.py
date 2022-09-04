"""
city_data.py

Reads .shp files and creates .csv dataset of features for each city in the dataset.

"""

import geopandas as gpd
import matplotlib.pyplot as plt

# Reading in the example .shp file with geopandas.
# Data obtained from https://catalog.data.gov/dataset/500-cities-city-boundaries
gdf = gpd.read_file('.\example_data\CityBoundaries.shp')

gdf.plot()

plt.show()
