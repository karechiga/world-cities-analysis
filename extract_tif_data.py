"""
extract_worldclim_data.py

Reads .shp files and creates .csv dataset of features for each city in the dataset.

Features:

"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import gdal
import matplotlib.pyplot as plt
import errno, stat
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise

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

def tifToDF(data_path, chunkx = 0, chunky = 0, offsetx = 0, offsety = 0): 
    # Reads data from .tif files and stores in a pandas DataFrame
    # Optionally data can be stored in a DataFrame in chunks (for large datasets)
    file_list = os.listdir(data_path)
    for file in file_list:
        if file.endswith(".tif"):
            tif = gdal.Open(data_path + file)
            # If the file is large, read it in chunks
            cols = tif.RasterXSize
            rows = tif.RasterYSize
            win_xsize = chunkx if chunkx != 0 else cols
            win_ysize = chunky if chunky != 0 else cols
            x_range = cols - offsetx if (cols - offsetx) <= win_xsize else win_xsize
            y_range = rows - offsety if (rows - offsety) <= win_ysize else win_ysize
            arr = tif.ReadAsArray(offsetx, offsety, x_range, y_range).flatten()

            # find all the latitudes and longtidues of the dataset
            x_coords = np.mod(np.linspace(0,len(arr)-1,num=len(arr)), x_range) + offsetx
            y_coords = np.floor(np.linspace(0,len(arr)-1, num = len(arr)) / x_range) + offsety
            lats = convertIndexToLat(y_coords, tif)
            longs = convertIndexToLong(x_coords, tif)
            df = pd.DataFrame({'Latitude':lats, 'Longitude':longs})
            df[file[:-4]] = arr # Add values from this file to the dataframe
            break
    for i in range(1,len(file_list)):   # Starting with the second file since the first was already added to the df
        if file_list[i].endswith(".tif"):
            tif = gdal.Open(data_path + file_list[i])
            # NOTE: The following assumes that each .tif file in the data_path has the same dimensions
            arr = tif.ReadAsArray(offsetx, offsety, x_range, y_range).flatten()
            df[file_list[i][:-4]] = arr
    return df

def worldclimCityData(poly_list):
    """
    This script reads in WorldClim .tif data and aggregates it into the mean value within each city.

    Args:
        poly_list (List of Polygons): represents the list of cities geometries.
        Each element is a Polygon or MultiPolygon object.
        
    Writes the mean features for each city to csv.
    """
    # WorldClim .tif files downloaded from https://worldclim.org/data/worldclim21.html and placed within the following path:
    data_path = '..\\datasets\\WorldClim\\'
    os.makedirs(data_path + 'csv_data', exist_ok=True) 
    # Read the .tif files, store in a dataframe
    df = tifToDF(data_path)
    df[df < -3e30] = None # Makes gibberish numbers NaN values
    for poly in poly_list:
        if str(type(poly)) == "<class 'shapely.geometry.polygon.Polygon'>": # cities with single polygon bounds
            df[poly.contains(Point(df['Longitude'],df['Latitude']))]
    return

def paleoclimCityData(poly_list):
    # paleoclim data files downloaded from http://www.paleoclim.org/. Files placed into different directories with data_path below:
    data_path = '..\\datasets\\paleoclim\\'
    return

def landscanCityData(gdf_list):
    # LandScan data files downloaded from https://landscan.ornl.gov/metadata.
    data_path = '..\\datasets\\landscan\\'
    return