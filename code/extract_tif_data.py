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
from shapely.geometry.polygon import Polygon


def convertIndexToLong(x_ind_arr, rast):
    # input is an array of indices, and the raster dataset object
    # output is a linearly transformed array listing longitudes instead of indices
    ulx, xres, xskew, uly, yskew, yres  = rast.GetGeoTransform()    
    long_arr = ulx + ((x_ind_arr) * xres)
    return long_arr

def convertIndexToLat(y_ind_arr, rast):
    # input is an array of indices, and the raster dataset object
    # output is a linearly transformed array listing latitudes instead of indices
    ulx, xres, xskew, uly, yskew, yres  = rast.GetGeoTransform()        
    lat_arr = uly + (y_ind_arr * yres)
    return lat_arr

def convertLongToIndex(long_arr, rast):
    # input is a longitude or array of longitudes, and the raster dataset object
    # output is an array of x indices for the raster file.
    ulx, xres, xskew, uly, yskew, yres  = rast.GetGeoTransform()    
    x_arr = np.floor((long_arr - ulx) / xres)
    # index 0 => longitude -180, index 1 => -179.9166666 and so on
    # so if long = -179.95, then index will be 0
    return x_arr

def convertLatToIndex(lat_arr, rast):
    # input is a latitude or array of latitudes, and the raster dataset object
    # output is an array of y indices for the raster file.
    ulx, xres, xskew, uly, yskew, yres  = rast.GetGeoTransform()        
    y_arr = np.floor((lat_arr - uly) / yres)
    return y_arr

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
    # NOTE: This function assumes that all the tifs have the same dimensions!
    # If they have different Raster sizes, this function will return 0.
    for tif in tifs:
        # Reads the specified "chunk" of the data
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
        if tifs[i].RasterYSize != rows or tifs[i].RasterXSize != cols:
            print(".tif file #{} has different dimensions. Make sure all tifs input into this function have the same dimensions.".format(i+1))
            return 0
        arr = tifs[i].ReadAsArray(offsetx, offsety, x_range, y_range).flatten()
        df[i] = arr
    return df

def plotCity(poly, pixels = None, res = None, title='', xlabel='Longitude (degrees)', ylabel='Latitude (degrees)'):
    """
    This function plots the city exterior and interior boundaries.
    Optionally it also plots the neighboring pixels and their intersections.
    
    inputs:
        poly -> a Polygon object representing a city
        pixels -> a pandas DataFrame containing the coordinates
        res -> list of two floats representing the x resolution and y resolution of the pixels.
        of each pixel within range of the city.
        title, xlabel, ylabel -> Optional title, x-label, and y-label strings to name the plot.
    """
    plt.figure()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    x1,y1 = poly.exterior.xy
    line1 = plt.plot(x1,y1,'k',label='City Bounds')
    plt.fill(x1,y1,'gainsboro')
    for inter in poly.interiors:
        x1,y1 = inter.xy
        plt.plot(x1,y1,'k',label='City Bounds')
        plt.fill(x1,y1,'white')
    # iterate over each pixel and plot it
    if pixels is None:
        return
    for index, point in pixels.iterrows():
        x = point['Longitude']
        y = point['Latitude']
        temp_poly = Polygon([(x, y),                    # Top left corner
                            (x+res[0], y),            # Top right corner
                            (x+res[0], y+res[1]),    # Bottom right corner
                            (x, y+res[1])])            # Bottom left corner
        # For generating an example visualization
        x2,y2 = temp_poly.exterior.xy
        if temp_poly.intersects(poly):
            line2 = plt.plot(x2,y2,'b:',label='Interior Pixel')
            if temp_poly.intersection(poly).area / temp_poly.area == 1:
                plt.fill(x2,y2,'lightblue')   # fill polygons if they fully intersect
        else:
            line3 = plt.plot(x2,y2,'r:',label='Exterior Pixel')
            plt.fill(x2,y2,'mistyrose')   # fill polygons if they fully intersect
    plt.legend(handles=[line1[0],line2[0],line3[0]],framealpha=1,loc='upper right')
    return

def worldclimCityData(gdf):
    """
    This script reads in WorldClim .tif data and aggregates it into the mean value within each city.
    NOTE: This function assumes that all .tif Raster files in the directory have the same dimensions
    
    Args:
        gdf (GeoDataFrame): represents the list of cities and their geometries.
        
    Writes the mean features for each city to csv.
    """
    start = time.time()
    names =  gdf['name_conve'].values
    # city = "Boston"     # For plotting city pixel data
    poly_list = gdf.geometry
    # WorldClim .tif files downloaded from https://worldclim.org/data/worldclim21.html and placed within the following path:
    data_path = '../../datasets/WorldClim/'
    # Read the .tif files, store in a dataframe, store location resolutions
    tifs, file_names = openTifsInDirectory(data_path)
    long_res = tifs[0].GetGeoTransform()[1]
    lat_res = tifs[0].GetGeoTransform()[5]
    cols = file_names
    cols.insert(0,'Longitude')
    cols.insert(0,'Latitude')
    cols.insert(0,'tif_count')
    city_arr = np.zeros((len(poly_list),len(cols))) # initializing the array to be returned at the end
    for i, poly in enumerate(poly_list):
        old_time = time.time()
        print("Starting to aggregate WorldClim data for city {} ({})".format(i,names[i]))
        if str(type(poly)) == "<class 'shapely.geometry.polygon.Polygon'>": # cities with single polygon bounds
            if not poly.is_valid:
                # Some cities have boundaries that cross each other which causes the polygons to be invalid.
                # This poly.buffer(0) makes it so the external bounds don't cross.
                ### Below demonstrates the difference between the original city and the buffered city. ###
                # a1 = poly.area
                # a2 = poly.buffer(0).area
                # if a1 != a2:
                #     plotCity(poly,title='{} (before buffer) Area = {}'.format(names[i],poly.area), xlabel='Longitude (degrees)', ylabel='Latitude (degrees)')
                #     x1,y1 = poly.exterior.xy
                #     plt.plot(x1,y1,'r-')
                #     plotCity(poly.buffer(0),title='{} (after buffer) Area = {}'.format(names[i],poly.buffer(0).area), xlabel='Longitude (degrees)', ylabel='Latitude (degrees)')
                #     x1,y1 = poly.buffer(0).exterior.xy
                #     plt.plot(x1,y1,'b-')
                #     plt.show()
                poly= poly.buffer(0)    

            # get only chunk of tif files that could be within city bounds
            y_bounds = convertLatToIndex(np.array([poly.bounds[1], poly.bounds[3]]), tifs[0])
            x_bounds = convertLongToIndex(np.array([poly.bounds[0], poly.bounds[2]]), tifs[0])
            df = tifsToDF(tifs, chunkx=int(abs(x_bounds[1]-x_bounds[0])+1), chunky=int(abs(y_bounds[1]-y_bounds[0])+1), offsetx=int(min(x_bounds)), offsety=int(min(y_bounds)))
            df[(df < -3e30)] = None
            df = df.dropna()
            # For plotting a specific city:
            # if names[i] == city:
            #     plotCity(poly, df, [long_res, lat_res],'Pixel Intersections in {} (five arcminute)'.format(city), 'Longitude (degrees)', 'Latitude (degrees)')
            #     plt.savefig('../figures/{}_5m_pixels.png'.format(city))
            #     plt.show()
            for index, point in df.iterrows():
                x = point['Longitude']
                y = point['Latitude']
                temp_poly = Polygon([(x, y),                    # Top left corner
                                    (x+long_res, y),            # Top right corner
                                    (x+long_res, y+lat_res),    # Bottom right corner
                                    (x, y+lat_res)])            # Bottom left corner
                if temp_poly.intersects(poly):   # if the data point is inside the city polygon
                    # The following calculates the percentage of the tif point that intersects the city
                    weight = temp_poly.intersection(poly).area / temp_poly.area
                    city_arr[i][0] += weight
                    # Increments by the intersection percentage weight
                    city_arr[i][1:] = np.add(city_arr[i][1:], weight*point.values)
        else:
            # For cities represented as MultiPolygons, extract each polygon within them and evaluate
            for p in poly.geoms:
                if not p.is_valid:
                    p = p.buffer(0)
                # get only chunk of tif files that could be within city bounds
                y_bounds = convertLatToIndex(np.array([p.bounds[1], p.bounds[3]]), tifs[0])
                x_bounds = convertLongToIndex(np.array([p.bounds[0], p.bounds[2]]), tifs[0])
                df = tifsToDF(tifs, chunkx=int(abs(x_bounds[1]-x_bounds[0])+1), chunky=int(abs(y_bounds[1]-y_bounds[0])+1), offsetx=int(min(x_bounds)), offsety=int(min(y_bounds)))
                df[(df < -3e30)] = None
                df = df.dropna()
                for index, point in df.iterrows():
                    x = point['Longitude']
                    y = point['Latitude']
                    temp_poly = Polygon([(x, y),                    # Top left corner
                                        (x+long_res, y),            # Top right corner
                                        (x+long_res, y+lat_res),    # Bottom right corner
                                        (x, y+lat_res)])            # Bottom left corner
                    if temp_poly.intersects(p):   # if the data point is inside the city polygon
                        # The following calculates the percentage of the tif point that intersects the city
                        weight = temp_poly.intersection(p).area / temp_poly.area
                        city_arr[i][0] += weight
                        # Increments by the intersection percentage weight
                        city_arr[i][1:] = np.add(city_arr[i][1:], weight*point.values)
        new_time = time.time()
        print("City {} ({}) WorldClim data completed in {}sec ({}sec from the start time)".format(i,names[i],new_time-old_time,new_time-start))
        # For generating an example visualization
    city_arr_t = city_arr.T
    city_arr_t[1:] = city_arr_t[1:] / city_arr_t[0]   # Averaging all values in city_arr
    city_arr = city_arr_t.T
    return pd.DataFrame(city_arr,columns=cols)

def paleoclimCityData(gdf):
    """
    This script reads in paleoclim .tif data and aggregates it into the mean value within each city.
    NOTE: This function assumes that all .tif Raster files in the directory have the same dimensions

    Args:
        gdf (GeoDataFrame): represents the list of cities and their geometries.
        
    Writes the mean features for each city to csv.
    """
    start = time.time()
    names =  gdf['name_conve'].values
    poly_list = gdf.geometry
    # paleoclim data files downloaded from http://www.paleoclim.org/. Files placed into the data_path below:
    data_path = '../../datasets/paleoclim/'
    # Read the .tif files, store in a dataframe, store location resolutions
    tifs, file_names = openTifsInDirectory(data_path)
    long_res = tifs[0].GetGeoTransform()[1]
    lat_res = tifs[0].GetGeoTransform()[5]
    cols = file_names
    cols.insert(0,'Longitude')
    cols.insert(0,'Latitude')
    cols.insert(0,'tif_count')
    city_arr = np.zeros((len(poly_list),len(cols))) # initializing the array to be returned at the end
    for i, poly in enumerate(poly_list):
        old_time = time.time()
        print("Starting to aggregate PaleoClim data for city {} ({})".format(i,names[i]))
        if str(type(poly)) == "<class 'shapely.geometry.polygon.Polygon'>": # cities with single polygon bounds
            if not poly.is_valid:
                # x1,y1 = poly.exterior.xy
                # plt.plot(x1,y1)
                # plt.show()
                poly = poly.buffer(0)
            # get only chunk of tif files that could be within city bounds
            y_bounds = convertLatToIndex(np.array([poly.bounds[1], poly.bounds[3]]), tifs[0])
            x_bounds = convertLongToIndex(np.array([poly.bounds[0], poly.bounds[2]]), tifs[0])
            df = tifsToDF(tifs, chunkx=int(abs(x_bounds[1]-x_bounds[0])+1), chunky=int(abs(y_bounds[1]-y_bounds[0])+1), offsetx=int(min(x_bounds)), offsety=int(min(y_bounds)))
            df[(df < -3e30)] = None
            df = df.dropna()
            for index, point in df.iterrows():
                x = point['Longitude']
                y = point['Latitude']
                temp_poly = Polygon([(x, y),                    # Top left corner
                                    (x+long_res, y),            # Top right corner
                                    (x+long_res, y+lat_res),    # Bottom right corner
                                    (x, y+lat_res)])            # Bottom left corner
                if temp_poly.intersects(poly):   # if the data point is inside the city polygon
                    # The following calculates the percentage of the tif point that intersects the city
                    weight = temp_poly.intersection(poly).area / temp_poly.area
                    city_arr[i][0] += weight
                    # Increments by the intersection percentage weight
                    city_arr[i][1:] = np.add(city_arr[i][1:], weight*point.values)
        else:
            # For cities represented as MultiPolygons, extract each polygon within them and evaluate
            for p in poly.geoms:
                if not p.is_valid:
                    # x1,y1 = p.exterior.xy
                    # plt.plot(x1,y1)
                    # plt.show()
                    p = p.buffer(0)
                # get only chunk of tif files that could be within city bounds
                y_bounds = convertLatToIndex(np.array([p.bounds[1], p.bounds[3]]), tifs[0])
                x_bounds = convertLongToIndex(np.array([p.bounds[0], p.bounds[2]]), tifs[0])
                df = tifsToDF(tifs, chunkx=int(abs(x_bounds[1]-x_bounds[0])+1), chunky=int(abs(y_bounds[1]-y_bounds[0])+1), offsetx=int(min(x_bounds)), offsety=int(min(y_bounds)))
                df[(df < -3e30)] = None # Remove Rows that have negative numbers as population sizes.
                df = df.dropna()
                for index, point in df.iterrows():
                    x = point['Longitude']
                    y = point['Latitude']
                    temp_poly = Polygon([(x, y),                    # Top left corner
                                        (x+long_res, y),            # Top right corner
                                        (x+long_res, y+lat_res),    # Bottom right corner
                                        (x, y+lat_res)])            # Bottom left corner
                    if temp_poly.intersects(p):   # if the data point is inside the city polygon
                        # The following calculates the percentage of the tif point that intersects the city
                        weight = temp_poly.intersection(p).area / temp_poly.area
                        city_arr[i][0] += weight
                        # Increments by the intersection percentage weight
                        city_arr[i][1:] = np.add(city_arr[i][1:], weight*point.values)
        new_time = time.time()
        print("City {} ({}) PaleoClim data completed in {}sec ({}sec from the start time)".format(i,names[i],new_time-old_time,new_time-start))
    city_arr_t = city_arr.T
    city_arr_t[1:] = city_arr_t[1:] / city_arr_t[0]   # Averaging all values in city_arr
    city_arr = city_arr_t.T
    return pd.DataFrame(city_arr,columns=cols)

def landscanCityData(gdf):
    """
    This script reads in landscan .tif data from two different years, and aggregates it into the sum value within each city.

    Args:
        gdf (GeoDataFrame): represents the list of cities and their geometries.
    Writes the sum of populations for each city to csv.
    """
    start = time.time()
    names =  gdf['name_conve'].values
    poly_list = gdf.geometry
    # landscan data files downloaded from https://landscan.ornl.gov/. Files placed into the data_path below:
    data_path = '../../datasets/landscan/'
    # Read the .tif files, store in a dataframe, store location resolutions
    # for landscan, read in chunks (too much data to store all in a dataframe)
    # city = "Tucson"     # City to be plotted
    tifs, file_names = openTifsInDirectory(data_path)
    # NOTE: Landscan .tif files for 2000 and 2020 have different dimensions and must be aggregated separately
    long_res = (tifs[0].GetGeoTransform()[1], tifs[1].GetGeoTransform()[1])
    lat_res = (tifs[0].GetGeoTransform()[5], tifs[1].GetGeoTransform()[5])
    cols = ['tif_count_2000', 'tif_count_2021', 'Latitude',
            'Longitude', 'pop_2000', 'pop_2021', 'pop_change']
    city_arr = np.zeros((len(poly_list),len(cols))) # initializing the array to be returned at the end
    for i, poly in enumerate(poly_list):
        old_time = time.time()
        print("Starting to aggregate population data for city {} ({})".format(i,names[i]))
        if str(type(poly)) == "<class 'shapely.geometry.polygon.Polygon'>": # cities with single polygon bounds
            if not poly.is_valid:
                poly = poly.buffer(0)
            for j,t in enumerate(tifs): # iterating over both tifs because they have different dimensions.
                # get only chunk of tif files that could be within city bounds
                y_bounds = convertLatToIndex(np.array([poly.bounds[1], poly.bounds[3]]), t)
                x_bounds = convertLongToIndex(np.array([poly.bounds[0], poly.bounds[2]]), t)
                df = tifsToDF([t], chunkx=int(abs(x_bounds[1]-x_bounds[0])+1), chunky=int(abs(y_bounds[1]-y_bounds[0])+1), offsetx=int(min(x_bounds)), offsety=int(min(y_bounds)))
                df[(df[0] < 0)] = None # Remove Rows that have negative numbers as population sizes.
                df = df.dropna()
                # For plotting a specific city:
                # if names[i] == city:
                #     plotCity(poly, df, [long_res[j], lat_res[j]],'Pixel Intersections in {} (30 arcsec resolution)'.format(city), 'Longitude (degrees)', 'Latitude (degrees)')
                #     plt.savefig('../figures/{}_30s_pixels.png'.format(city))
                #     plt.show()
                for index, point in df.iterrows():
                    x = point['Longitude']
                    y = point['Latitude']
                    temp_poly = Polygon([(x, y),                    # Top left corner
                                        (x+long_res[j], y),            # Top right corner
                                        (x+long_res[j], y+lat_res[j]),    # Bottom right corner
                                        (x, y+lat_res[j])])            # Bottom left corner
                    if temp_poly.intersects(poly):   # if the data point is inside the city polygon
                        # The following calculates the percentage of the tif point that intersects the city
                        weight = temp_poly.intersection(poly).area / temp_poly.area
                        # Increments by the intersection percentage weight
                        city_arr[i][j] += weight
                        city_arr[i][2:4] = np.add(city_arr[i][2:4], weight*point.values[0:2])
                        city_arr[i][4+j] += weight*point.values[2]
        else:
            # For cities represented as MultiPolygons, extract each polygon within them and evaluate
            for p in poly.geoms:
                if not p.is_valid:
                    p = p.buffer(0)
                for j,t in enumerate(tifs): # iterating over both tifs because they have different dimensions.
                    # get only chunk of tif files that could be within city bounds
                    y_bounds = convertLatToIndex(np.array([p.bounds[1], p.bounds[3]]), t)
                    x_bounds = convertLongToIndex(np.array([p.bounds[0], p.bounds[2]]), t)
                    df = tifsToDF([t], chunkx=int(abs(x_bounds[1]-x_bounds[0])+1), chunky=int(abs(y_bounds[1]-y_bounds[0])+1), offsetx=int(min(x_bounds)), offsety=int(min(y_bounds)))
                    df[(df[0] < 0)] = None # Remove Rows that have negative numbers as population sizes.
                    df = df.dropna()
                    # This will reduce some computation time
                    for index, point in df.iterrows():
                        x = point['Longitude']
                        y = point['Latitude']
                        temp_poly = Polygon([(x, y),                    # Top left corner
                                            (x+long_res[j], y),            # Top right corner
                                            (x+long_res[j], y+lat_res[j]),    # Bottom right corner
                                            (x, y+lat_res[j])])            # Bottom left corner
                        if temp_poly.intersects(p):   # if the data point is inside the city polygon
                            # The following calculates the percentage of the tif point that intersects the city
                            weight = temp_poly.intersection(p).area / temp_poly.area
                            # Increments by the intersection percentage weight
                            city_arr[i][j] += weight
                            city_arr[i][2:4] = np.add(city_arr[i][2:4], weight*point.values[0:2])
                            city_arr[i][4+j] += weight*point.values[2]
        new_time = time.time()
        print("City {} ({}) population data completed in {}sec ({}sec from the start time)".format(i,names[i],new_time-old_time,new_time-start))
    city_arr_t = city_arr.T
    city_arr_t[2:4] = city_arr_t[2:4] / (city_arr_t[0]+city_arr_t[1])   # Averaging the latitude and longitude for each city
    city_arr_t[-1] = city_arr_t[-2] - city_arr_t[-3]    # Overall_Change calculation
    city_arr = city_arr_t.T
    return pd.DataFrame(city_arr,columns=cols)