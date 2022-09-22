"""
extract_worldclim_data.py

Reads .shp files and creates .csv dataset of features for each city in the dataset.

Features:

"""
import os
import time
import pandas as pd
import geopandas as gpd
import numpy as np
import gdal
import matplotlib.pyplot as plt
import errno, stat
from shapely.geometry import Point, asMultiPoint
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

def openTifsInDirectory(data_path):
    # Reads all .tif files within a directory and returns a list of gdf objects
    # Also returns a list of the names of the files.
    file_list = os.listdir(data_path)
    tifs = []
    names = []
    for file in file_list:
        if file.endswith(".tif"):
            tifs.append(gdal.Open(data_path + file))
            names.append(file)
    return tifs, names

def tifsToDF(tifs, chunkx = 0, chunky = 0, offsetx = 0, offsety = 0):
    # Stores list of "tifs" (gdal objects representing read .tif files) in a pandas DataFrame
    # Optionally data can be stored in a DataFrame in chunks (for large datasets)
    for tif in tifs:
        # If the file is large, the following will only read the specified "chunk" of the data
        cols = tif.RasterXSize
        rows = tif.RasterYSize
        win_xsize = chunkx if chunkx != 0 else cols
        win_ysize = chunky if chunky != 0 else cols
        x_range = cols - offsetx if (cols - offsetx) <= win_xsize else win_xsize
        y_range = rows - offsety if (rows - offsety) <= win_ysize else win_ysize
        arr = tif.ReadAsArray(offsetx, offsety, x_range, y_range).flatten()

        # find all the latitudes and longitudes of the dataset
        x_coords = np.mod(np.linspace(0,len(arr)-1,num=len(arr)), x_range) + offsetx
        y_coords = np.floor(np.linspace(0,len(arr)-1, num = len(arr)) / x_range) + offsety
        lats = convertIndexToLat(y_coords, tif)
        longs = convertIndexToLong(x_coords, tif)
        df = pd.DataFrame({'Latitude':lats, 'Longitude':longs})
        df[0] = arr # Add values from this file to the dataframe
        break
    for i in range(1,len(tifs)):   # Starting with the second file since the first was already added to the df
        # NOTE: The following assumes that each .tif file in the data_path has the same dimensions
        arr = tifs[i].ReadAsArray(offsetx, offsety, x_range, y_range).flatten()
        df[i] = arr
    return df

def worldclimCityData(poly_list):
    """
    This script reads in WorldClim .tif data and aggregates it into the mean value within each city.

    Args:
        poly_list (List of Polygons): represents the list of cities geometries.
        Each element is a Polygon object.
        
    Writes the mean features for each city to csv.
    """
    # WorldClim .tif files downloaded from https://worldclim.org/data/worldclim21.html and placed within the following path:
    data_path = '..\\..\\datasets\\WorldClim\\'
    # Read the .tif files, store in a dataframe, store location resolutions
    tifs, file_names = openTifsInDirectory(data_path)
    df = tifsToDF(tifs)
    long_res = (np.unique(df['Longitude']).max() - np.unique(df['Longitude']).min())/(len(np.unique(df['Longitude']))-1)
    lat_res = (np.unique(df['Latitude']).max() - np.unique(df['Latitude']).min())/(len(np.unique(df['Latitude']))-1)
    df[df < -3e30] = None # Makes gibberish numbers NaN values
    cols = file_names
    cols.insert(0,'Longitude')
    cols.insert(0,'Latitude')
    cols.insert(0,'tif_count')
    city_arr = np.zeros((len(poly_list),len(cols))) # initializing the array to be returned at the end
    # f1 = plt.figure()   # for debugging
    for i, poly in enumerate(poly_list):
        # p1x,p1y = poly.exterior.xy # for debugging
        if str(type(poly)) == "<class 'shapely.geometry.polygon.Polygon'>": # cities with single polygon bounds
            temp_df = df[(df['Latitude'].between(poly.bounds[1], poly.bounds[3]+lat_res)) &
                         (df['Longitude'].between(poly.bounds[0], poly.bounds[2]+long_res))]
            for index, point in temp_df.iterrows():
                x = point['Longitude']
                y = point['Latitude']
                temp_poly = Polygon([(x-long_res, y-lat_res),   # Bottom left corner
                                     (x, y-lat_res),            # Bottom right corner
                                     (x, y),                   # Top right corner
                                     (x-long_res, y)])           # Top left corner
                # For debugging below:
                # p2x,p2y = temp_poly.exterior.xy
                # plt.plot(p2x,p2y)
                # plt.plot(p1x,p1y)
                # plt.show()
                if temp_poly.intersects(poly):   # if the data point is inside the city polygon
                    # should consider adding by a percentage of the intersection of the polygon
                    city_arr[i][0] += 1
                    city_arr[i][1:] = np.add(city_arr[i][1:], point.values)
            # city_arr[i][1:] = city_arr[i][1:]/city_arr[i][0]
    city_arr_t = city_arr.T
    city_arr_t[1:] = city_arr_t[1:] / city_arr_t[0]   # Averaging all values in city_arr
    city_arr = city_arr_t.T
    return pd.DataFrame(city_arr,columns=cols)

def paleoclimCityData(poly_list):
    """
    This script reads in paleoclim .tif data and aggregates it into the mean value within each city.

    Args:
        poly_list (List of Polygons): represents the list of cities geometries.
        Each element is a Polygon object.
        
    Writes the mean features for each city to csv.
    """
    # paleoclim data files downloaded from http://www.paleoclim.org/. Files placed into the data_path below:
    data_path = '..\\..\\datasets\\paleoclim\\'
    # Read the .tif files, store in a dataframe, store location resolutions
    tifs, file_names = openTifsInDirectory(data_path)
    df = tifsToDF(tifs)
    long_res = (np.unique(df['Longitude']).max() - np.unique(df['Longitude']).min())/(len(np.unique(df['Longitude']))-1)
    lat_res = (np.unique(df['Latitude']).max() - np.unique(df['Latitude']).min())/(len(np.unique(df['Latitude']))-1)
    df[df < -3e30] = None # Makes gibberish numbers NaN values
    cols = file_names
    cols.insert(0,'Longitude')
    cols.insert(0,'Latitude')
    cols.insert(0,'tif_count')
    city_arr = np.zeros((len(poly_list),len(cols))) # initializing the array to be returned at the end
    # f1 = plt.figure()   # for debugging
    for i, poly in enumerate(poly_list):
        # p1x,p1y = poly.exterior.xy # for debugging
        if str(type(poly)) == "<class 'shapely.geometry.polygon.Polygon'>": # cities with single polygon bounds
            temp_df = df[(df['Latitude'].between(poly.bounds[1], poly.bounds[3]+lat_res)) &
                         (df['Longitude'].between(poly.bounds[0], poly.bounds[2]+long_res))]
            for index, point in temp_df.iterrows():
                x = point['Longitude']
                y = point['Latitude']
                temp_poly = Polygon([(x-long_res, y-lat_res),   # Bottom left corner
                                     (x, y-lat_res),            # Bottom right corner
                                     (x, y),                   # Top right corner
                                     (x-long_res, y)])           # Top left corner
                # For debugging below:
                # p2x,p2y = temp_poly.exterior.xy
                # plt.plot(p2x,p2y)
                # plt.plot(p1x,p1y)
                # plt.show()
                if temp_poly.intersects(poly):   # if the data point is inside the city polygon
                    # should consider adding by a percentage of the intersection of the polygon
                    city_arr[i][0] += 1
                    city_arr[i][1:] = np.add(city_arr[i][1:], point.values)
            # city_arr[i][1:] = city_arr[i][1:]/city_arr[i][0]
    city_arr_t = city_arr.T
    city_arr_t[1:] = city_arr_t[1:] / city_arr_t[0]   # Averaging all values in city_arr
    city_arr = city_arr_t.T
    return pd.DataFrame(city_arr,columns=cols)

def landscanCityData(poly_list):
    """
    This script reads in landscan .tif data and aggregates it into the sum value within each city.

    Args:
        poly_list (List of Polygons): represents the list of cities geometries.
        Each element is a Polygon object.
        
    Writes the sum of populations for each city to csv.
    """
    start = time.time()
    # landscan data files downloaded from https://landscan.ornl.gov/. Files placed into the data_path below:
    data_path = '..\\..\\datasets\\landscan\\'
    # Read the .tif files, store in a dataframe, store location resolutions
    # for landscan, read in chunks (too much data to store all in a dataframe)
    
    tifs, file_names = openTifsInDirectory(data_path)
    cols = file_names
    cols.insert(0,'Longitude')
    cols.insert(0,'Latitude')
    cols.insert(0,'tif_count')
    cols.append('Absolute_Change')
    totalX = tifs[0].RasterXSize  # Number of columns in each .tif file (assumes all .tifs in the directory have the same dimensions)
    totalY = tifs[0].RasterYSize  # Number of rows in each .tif file (assumes all .tifs in the directory have the same dimensions)
    win_xsize = 12000    # Size of the chunks (x)
    win_ysize = 6000    # Size of the chunks (y)
    city_arr = np.zeros((len(poly_list),len(cols))) # initializing the array to be returned at the end
    count = 0
    for r in range(0, totalY, win_ysize):
        for c in range(0, totalX, win_xsize):
            df = tifsToDF(tifs, chunkx=win_xsize, chunky=win_ysize, offsetx=c, offsety=r)   # stores only a chunk of the .tif data
            long_res = (np.unique(df['Longitude']).max() - np.unique(df['Longitude']).min())/(len(np.unique(df['Longitude']))-1)
            lat_res = (np.unique(df['Latitude']).max() - np.unique(df['Latitude']).min())/(len(np.unique(df['Latitude']))-1)
            index_to_drop = df[(df[0] < 0) | (df[1] < 0)].index # Remove Rows that have negative numbers as population sizes.
            df.drop(index_to_drop , inplace=True)
            index_to_drop = df[(df[0] == 0) & (df[1] == 0)].index
            df.drop(index_to_drop , inplace=True)
            # This will significantly reduce computation time
            new_time = time.time()
            print("{} percent complete at ".format(count*win_xsize*win_ysize/(totalX*totalY)) + "{} seconds from the start of the landscan function.".format(new_time-start))
            count += 1
            for i, poly in enumerate(poly_list):
                if str(type(poly)) == "<class 'shapely.geometry.polygon.Polygon'>": # cities with single polygon bounds
                    # p1x,p1y = poly.exterior.xy # for debugging
                    temp_df = df[(df['Latitude'].between(poly.bounds[1], poly.bounds[3]+lat_res)) &
                                (df['Longitude'].between(poly.bounds[0], poly.bounds[2]+long_res))]
                    for index, point in temp_df.iterrows():
                        x = point['Longitude']
                        y = point['Latitude']
                        temp_poly = Polygon([(x-long_res, y-lat_res),   # Bottom left corner
                                            (x, y-lat_res),            # Bottom right corner
                                            (x, y),                   # Top right corner
                                            (x-long_res, y)])           # Top left corner
                        # # For debugging below:
                        # p2x,p2y = temp_poly.exterior.xy
                        # plt.plot(p2x,p2y)
                        # plt.plot(p1x,p1y)
                        # plt.show()
                        if temp_poly.intersects(poly):   # if the data point is inside the city polygon
                            # should consider adding by a percentage of the intersection of the polygon
                            city_arr[i][0] += 1
                            city_arr[i][1:-1] = np.add(city_arr[i][1:-1], point.values)
                else:
                    # For cities represented as MultiPolygons, extract each polygon within them and evaluate
                    for p in poly.geoms:
                        # p1x,p1y = poly.exterior.xy # for debugging
                        temp_df = df[(df['Latitude'].between(p.bounds[1], p.bounds[3]+lat_res)) &
                                    (df['Longitude'].between(p.bounds[0], p.bounds[2]+long_res))]
                        for index, point in temp_df.iterrows():
                            x = point['Longitude']
                            y = point['Latitude']
                            temp_poly = Polygon([(x-long_res, y-lat_res),   # Bottom left corner
                                                (x, y-lat_res),            # Bottom right corner
                                                (x, y),                   # Top right corner
                                                (x-long_res, y)])           # Top left corner
                            # # For debugging below:
                            # p2x,p2y = temp_poly.exterior.xy
                            # plt.plot(p2x,p2y)
                            # plt.plot(p1x,p1y)
                            # plt.show()
                            if temp_poly.intersects(p):   # if the data point is inside the city polygon
                                # should consider adding by a percentage of the intersection of the polygon
                                city_arr[i][0] += 1
                                city_arr[i][1:-1] = np.add(city_arr[i][1:-1], point.values)
    city_arr_t = city_arr.T
    city_arr_t[1:3] = city_arr_t[1:3] / city_arr_t[0]   # Averaging the latitude and longitude for each city
    city_arr_t[-1] = city_arr_t[-2] - city_arr_t[-3]    # Overall_Change calculation
    city_arr = city_arr_t.T
    return pd.DataFrame(city_arr,columns=cols)