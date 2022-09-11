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
import shutil
import errno, stat
import time

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

def tifToCsv(data_path): 
    # Reads data from .tif files and writes to a CSV file
    os.makedirs(data_path, exist_ok=True) 
    dir_list = os.listdir(data_path)
    os.makedirs(data_path + 'csv_data', exist_ok=True) 
    for folder in dir_list:
        if os.path.isfile(data_path + folder):
            continue
        file_list = os.listdir(data_path + folder)
        # get dimensions of Rasters. NOTE: This assumes that every .tif in the same folder have the same dimensions.
        win_xsize = 0
        win_ysize = 0
        for file in file_list:
            if file.endswith(".tif"):
                tif = gdal.Open(data_path + folder + "/" + file)
                # If the file is large, read it in chunks
                cols = tif.RasterXSize
                rows = tif.RasterYSize
                win_xsize = 5000
                win_ysize = 2500
                break
        if win_xsize == 0:
            # no .tif files found in the directory
            continue
        # Adding headers first
        headers = 'Latitude, Longitude'
        for file in file_list:
            if file.endswith(".tif"):
                headers = headers + ',' + file
        # The following writes to the csv by rows
        headers_w = False # tells if the headers have been written to the csv yet
        for i in range(0, rows, win_ysize):
            for j in range(0, cols, win_xsize):
                # Finding the latitude and longitude:
                x_range = cols - j if (cols - j) <= win_xsize else win_xsize
                y_range = rows - i if (rows - i) <= win_ysize else win_ysize
                arr = tif.ReadAsArray(j, i, x_range, y_range).flatten()
                x_coords = np.mod(np.linspace(0,len(arr)-1,num=len(arr)), x_range) + j
                y_coords = np.floor(np.linspace(0,len(arr)-1, num = len(arr)) / x_range) + i
                lats = convertIndexToLat(y_coords, tif)
                longs = convertIndexToLong(x_coords, tif)
                out_arr = np.vstack((lats, longs))
                # Add the datapoints from the .tif files to the output array
                for file in file_list:
                    if file.endswith(".tif"):
                        tif = gdal.Open(data_path + folder + "/" + file)
                        out_arr = np.vstack((out_arr, tif.ReadAsArray(j, i, x_range, y_range).flatten())) # reads a chunk of the array
                out_arr = out_arr.T
                out_arr[abs(out_arr) > 1e30] = None # Removes infinite values, saves a ton of storage space
                # Append this "Chunk" to the csv.
                if not headers_w:
                    # if no headers, write a new csv
                    with open(data_path + 'csv_data\\' + folder + '.csv','w') as csvfile:
                        np.savetxt(csvfile, out_arr,delimiter=',',header=headers,fmt='%2.5f', comments='')
                    headers_w = True
                else:
                    # if already headers, append to the current csv
                    with open(data_path + 'csv_data\\' + folder + '.csv','a') as csvfile:
                        np.savetxt(csvfile,out_arr,delimiter=',',fmt='%2.5f', comments='')

# keep track of elapsed time:
start = time.time()

# WorldClim .tif files downloaded from https://worldclim.org/data/worldclim21.html and placed in different folders within the following path:
data_path = '..\\datasets\\WorldClim\\'
# Placed files in the following directories within data_path:
# # avr_temp, bioclim, max_temp, min_temp, precipitation, solar_rad
# # vapor_pressure, wind_speed
tifToCsv(data_path) # creates one CSV file per directory of .tif files. Creates a new directory "csv_data" within data_path
# # # NOTE: pandas `to_csv` will take some time to generate all of the .csv files.
shutil.make_archive(data_path + 'WorldClim_csv_data', 'zip', data_path + 'csv_data') # archive csv files in zip
shutil.rmtree(data_path + 'csv_data', ignore_errors=False, onerror=handleRemoveReadonly) # removes the large folder of csv's after creating the .zip
wctime = time.time()
print("WorldClim datset to csv complete: " + str(wctime-start) + " seconds\n")

# paleoclim data files downloaded from http://www.paleoclim.org/. Files placed into different directories with data_path below:
data_path = '..\\datasets\\paleoclim\\'
# Placed files in the following directories within data_path:
# bolling_allerod, dryas_stadial, heinrich_stadial, holocene_greenlandian,
# holocene_meghalayan, holocene_northgrippian, last_interglacial, mis19,
# # # pliocene_m2 and pliocene_warm.
tifToCsv(data_path) # creates one CSV file per directory of .tif files. Creates a new directory "csv_data" within data_path
shutil.make_archive(data_path + 'paleoclim_csv_data', 'zip', data_path + 'csv_data')
shutil.rmtree(data_path + 'csv_data', ignore_errors=False, onerror=handleRemoveReadonly) # removes the large folder of csv's after creating the .zip
ptime = time.time()
print("Paleoclim datset to csv complete: " + str(ptime-wctime) + " seconds\n")

# LandScan data files downloaded from https://landscan.ornl.gov/metadata.
data_path = '..\\datasets\\landscan\\'
# Placed all of the non-colorized .tif files in folders within data_path
tifToCsv(data_path) # creates one CSV file per directory of .tif files. Creates a new directory "csv_data" within data_path
shutil.make_archive(data_path + 'landscan_csv_data', 'zip', data_path + 'csv_data')
shutil.rmtree(data_path + 'csv_data', ignore_errors=False, onerror=handleRemoveReadonly) # removes the large folder of csv's after creating the .zip
ltime = time.time()
print("Landscan datset to csv complete: " + str(ltime-ptime) + " seconds\n Data collection complete! Time elapsed: " + str(ltime-start))