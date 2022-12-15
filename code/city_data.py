""" city_data.py
City data to extracted from the following sources and saved in csv files:
    https://worldclim.org/ (for each city, extract the mean BioX across all cells)
    http://www.paleoclim.org/ (for each city, extract the mean BioX across all cells)
    https://landscan.ornl.gov/ (for each city, first, extract the total value (sum) across all cells in 2000 and 2021; second, estimate the absolute change
"""
import extract_data as etd
import pandas as pd
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import optparse

data_path = '../../datasets/'

def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-c', '--plotcity',action='store',
                         metavar="C", type='string',dest='plot_city',default=None,
                         help='City to be plotted (case sensitive; options are any city in our database (Boston, Los Angeles, Tokyo, Beijing, etc.)' )
    optParser.add_option('-w', '--worldclim',action='store_true',
                         dest='worldclim',default=False,
                         help='Extract WorldClim data stored at "{}"'.format(data_path + 'worldclim'))
    optParser.add_option('-p', '--paleoclim',action='store_true',
                         dest='paleoclim',default=False,
                         help='Extract PaleoClim data stored at "{}"'.format(data_path + 'paleoclim'))
    optParser.add_option('-l', '--landscan',action='store_true',
                         dest='landscan',default=False,
                         help='Extract Landscan data stored at "{}"'.format(data_path + 'landscan'))
    optParser.add_option('-b', '--brightness',action='store_true',
                         dest='brightness',default=False,
                         help='Extract Sky Brightness data stored at "{}"'.format(data_path + 'brightness'))
    optParser.add_option('-r', '--roads',action='store_true',
                         dest='roads',default=False,
                         help='Extract road density data stored at "{}"'.format(data_path + 'roads'))
    optParser.add_option('-m', '--human',action='store_true',
                         dest='human',default=False,
                         help='Extract human modification data stored at "{}"'.format(data_path + 'human_modification'))
    optParser.add_option('-u', '--urban_heat',action='store_true',
                         dest='urban_heat',default=False,
                         help='Extract urban heat data stored at "{}"'.format(data_path + 'roads'))
    optParser.add_option('-y', '--land_use',action='store_true',
                         dest='land_use',default=False,
                         help='Extract land use data stored at "{}"'.format(data_path + 'land_use'))
    optParser.add_option('-e', '--elevation',action='store_true',
                         dest='elevation',default=False,
                         help='Extract elevation data stored at "{}"'.format(data_path + 'elevation'))
    optParser.add_option('-g', '--geodist',action='store_true',
                         dest='geodist',default=False,
                         help='Output geographical distances between cities.')
    opts, args = optParser.parse_args()

    return opts


if __name__ == '__main__':

    opts = parseOptions()
    gdf = gpd.read_file('../../city-boundaries/ne_10m_urban_areas_landscan.shp')
    cities = pd.DataFrame({'City' : gdf['name_conve']})
    os.makedirs(data_path + '/csv_data', exist_ok=True)
    # get data for all cities.
    if not opts.plot_city is None:
        # plotting a city
        ind = np.where(cities.values == opts.plot_city)[0][0]
        poly = gdf.geometry[ind]
        if not poly.is_valid:
            poly = poly.buffer(0)
        etd.plotCity(poly, title='Plot of the city boundaries of {}'.format(opts.plot_city))
        plt.savefig('../figures/{}_bounds.png'.format(opts.plot_city))
        if opts.worldclim:
            # plot city with worldclim pixels
            tifs, file_names = etd.openTifsInDirectory(data_path + '/worldclim/')
            long_res = tifs[0].GetGeoTransform()[1]
            lat_res = tifs[0].GetGeoTransform()[5]
            # get only chunk of tif files that could be within city bounds
            y_bounds = etd.convertLatToIndex(np.array([poly.bounds[1], poly.bounds[3]]), tifs[0])
            x_bounds = etd.convertLongToIndex(np.array([poly.bounds[0], poly.bounds[2]]), tifs[0])
            df = etd.tifsToDF(tifs, chunkx=int(abs(x_bounds[1]-x_bounds[0])+1), chunky=int(abs(y_bounds[1]-y_bounds[0])+1), offsetx=int(min(x_bounds)), offsety=int(min(y_bounds)))
            df[(df < -3e30)] = None # Remove Rows that have negative numbers as population sizes.
            df = df.dropna()
            etd.plotCity(poly, pixels=df, res=[long_res,lat_res],
                         title='WorldClim Pixel (5 arcmin resolution) Intersections for {}'.format(opts.plot_city))
            plt.savefig('../figures/{}_worldclim_pixels_5m.png'.format(opts.plot_city))
        
        if opts.paleoclim:
            # plot city with paleoclim pixels
            tifs, file_names = etd.openTifsInDirectory(data_path + '/paleoclim/')
            long_res = tifs[0].GetGeoTransform()[1]
            lat_res = tifs[0].GetGeoTransform()[5]
            # get only chunk of tif files that could be within city bounds
            y_bounds = etd.convertLatToIndex(np.array([poly.bounds[1], poly.bounds[3]]), tifs[0])
            x_bounds = etd.convertLongToIndex(np.array([poly.bounds[0], poly.bounds[2]]), tifs[0])
            df = etd.tifsToDF(tifs, chunkx=int(abs(x_bounds[1]-x_bounds[0])+1), chunky=int(abs(y_bounds[1]-y_bounds[0])+1), offsetx=int(min(x_bounds)), offsety=int(min(y_bounds)))
            df[(df < -3e30)] = None # Remove Rows that have negative numbers as population sizes.
            df = df.dropna()
            etd.plotCity(poly, pixels=df, res=[long_res,lat_res],
                         title='PaleoClim Pixel (5 arcmin resolution) Intersections for {}'.format(opts.plot_city))
            plt.savefig('../figures/{}_paleoclim_pixels_5m.png'.format(opts.plot_city))
        
        if opts.landscan:
            # plot city with landscan pixels
            # landscan uses tifs with different dimensions, so these should graphed twice.
            tifs, file_names = etd.openTifsInDirectory(data_path + '/landscan/')
            for i,t in enumerate(tifs):
                # tifs[0] would represent the year 2000 in the directory in our case
                year = file_names[i][-8:-4]
                long_res = t.GetGeoTransform()[1]
                lat_res = t.GetGeoTransform()[5]
                # get only chunk of tif files that could be within city bounds
                y_bounds = etd.convertLatToIndex(np.array([poly.bounds[1], poly.bounds[3]]), t)
                x_bounds = etd.convertLongToIndex(np.array([poly.bounds[0], poly.bounds[2]]), t)
                df = etd.tifsToDF([t], chunkx=int(abs(x_bounds[1]-x_bounds[0])+1), chunky=int(abs(y_bounds[1]-y_bounds[0])+1), offsetx=int(min(x_bounds)), offsety=int(min(y_bounds)))
                df[(df[0] < 0)] = None # Remove Rows that have negative numbers as population sizes.
                df = df.dropna()
                etd.plotCity(poly, pixels=df, res=[long_res,lat_res],
                            title='Landscan Population Pixel Intersections for {} in the year {} (resolution 30 arcsec)'.format(opts.plot_city, year))
                plt.savefig('../figures/{}_landscan_pixels_{}_30s.png'.format(opts.plot_city, year))
        plt.show()
    else:
        conts = gpd.read_file('../../continent-boundaries/ne_50m_geography_regions_polys.shp')
        conts = conts[conts['SCALERANK'] == 0]
        # Find what continent each city resides in.
        cont_list = etd.getContinents(gdf, conts)
        cities = pd.DataFrame({'City' : gdf['name_conve'], 'Region' : cont_list})
        if opts.worldclim:
            # get all cities worldclim data
            df = etd.worldclimCityData(gdf)
            df = pd.merge(cities,df,left_index=True,right_index=True)
            df.to_csv(data_path + 'csv_data/worldclim_cities.csv',index=False)
        if opts.paleoclim:
            # get all cities paleoclim data
            df = etd.paleoclimCityData(gdf)
            df = pd.merge(cities,df,left_index=True,right_index=True)
            df.to_csv(data_path + 'csv_data/paleoclim_cities.csv',index=False)
        if opts.landscan:
            # get all cities landscan data
            df = etd.landscanCityData(gdf)
            df = pd.merge(cities,df,left_index=True,right_index=True)
            df.to_csv(data_path + 'csv_data/landscan_cities.csv',index=False)
        if opts.brightness:
            # get all cities brightness data
            df = etd.brightnessData(gdf)
            df = pd.merge(cities,df,left_index=True,right_index=True)
            df.to_csv(data_path + 'csv_data/brightness_cities.csv',index=False)
        if opts.roads:
            # get all cities road density data
            df = etd.roadData(gdf)
            df = pd.merge(cities,df,left_index=True,right_index=True)
            df.to_csv(data_path + 'csv_data/roads_cities.csv',index=False)
        if opts.human:
            # get all cities human modification data
            df = etd.humanModificationData(gdf)
            df = pd.merge(cities,df,left_index=True,right_index=True)
            df.to_csv(data_path + 'csv_data/human_mod_cities.csv',index=False)
        if opts.urban_heat:
            # get all cities urban heat data
            df = etd.urbanHeatData(gdf)
            df = pd.merge(cities,df,left_index=True,right_index=True)
            df.to_csv(data_path + 'csv_data/urban_heat_cities.csv',index=False)
        if opts.land_use:
            # get all cities land usage data
            df = etd.landUseData(gdf)
            df = pd.merge(cities,df,left_index=True,right_index=True)
            df.to_csv(data_path + 'csv_data/land_use_cities.csv',index=False)
        if opts.elevation:
            # get all cities elevation data
            df = etd.elevationData(gdf)
            df = pd.merge(cities,df,left_index=True,right_index=True)
            df.to_csv(data_path + 'csv_data/elevation_cities.csv',index=False)
        if opts.geodist:
            # get geographical distances between cities
            cen = gdf.geometry.to_crs(3857).centroid    # city centroids
            df = pd.DataFrame((cen.x , cen.y)).T
            city_dists = np.zeros(shape=(len(cen), len(cen)))  # Distances of each city from each city
            for i, row in df.iterrows():
                city_dists[i, :] = np.sqrt(np.sum((row.values - df.values) ** 2, axis=1))
            city_dists = city_dists / 1000  # distances in km
            pd.merge(cities[['City', 'Region']], pd.DataFrame(city_dists, columns=cities['City']),
                    left_index=True,right_index=True).to_csv('../cluster_data/geo_dists.csv',index=False)