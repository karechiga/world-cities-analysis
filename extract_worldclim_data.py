"""
city_data.py

Reads .shp files and creates .csv dataset of features for each city in the dataset.

Features:

"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import gdal
import matplotlib.pyplot as plt

def convertIndexToLong(x_ind_arr, rast):
    # input is an array of indices, and the raster dataset object
    # output is a linearly transformed array listing longitudes instead of indices
    ulx, xres, xskew, uly, yskew, yres  = rast.GetGeoTransform()    
    long_arr = ulx + ((x_ind_arr+1) * xres)
    return long_arr

def convertIndexToLat(y_ind_arr, rast):
    # input is an array of indices, and the raster dataset object
    # output is a linearly transformed array listing latitudes instead of indices
    ulx, xres, xskew, uly, yskew, yres  = rast.GetGeoTransform()        
    lat_arr = uly + (y_ind_arr * yres)
    return lat_arr

# Read WorldClim data
# downloaded from https://worldclim.org/data/worldclim21.html and placed in different folders within the following path:
data_path = '..\\datasets\\WorldClim\\'
os.makedirs(data_path, exist_ok=True) 
dir_list = os.listdir(data_path)
for folder in dir_list:
    if os.path.isfile(data_path + folder):
        continue
    file_list = os.listdir(data_path + folder)
    # get latitudes and longitudes from x and y indices
    df = pd.DataFrame()
    for file in file_list:
        if file.endswith(".tif"):
            tif = gdal.Open(data_path + folder + "/" + file)
            arr = tif.ReadAsArray().flatten()
            x_inds = np.mod(np.linspace(0,len(arr)-1, num = len(arr)),tif.RasterXSize)
            y_inds = np.floor(np.linspace(0,len(arr)-1, num = len(arr)) / tif.RasterXSize)
            lats = convertIndexToLat(y_inds, tif)
            longs = convertIndexToLong(x_inds, tif)
            df = pd.DataFrame({"Latitude":lats, "Longitude":longs})
            break

    for file in file_list:
        if file.endswith(".tif"):
            tif = gdal.Open(data_path + folder + "/" + file)
            arr = tif.ReadAsArray().flatten()   # turns x and y indexes into one long vector with a single index
            df[file] = arr
    if len(df) != 0:
        os.makedirs(data_path + 'csv_data', exist_ok=True)  
        df.to_csv(data_path + 'csv_data\\' + folder + '_worldclim.csv', index=False, header=True) 

# Reading in the example .shp file with geopandas.
# Data obtained from https://catalog.data.gov/dataset/500-cities-city-boundaries

# gdf = gpd.read_file('.\example_data\CityBoundaries.shp')

# plt.show()
