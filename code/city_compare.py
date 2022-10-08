""" city_compare.py
Uses data aggregated in the csv files located at /../../datasets/csv_data/
"""

import functions as fnc
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

path = '../../datasets/csv_data/'
cities = fnc.get_cities(path)   # Get list of cities with their features.
features = cities.columns.drop(['City', 'Longitude', 
                                'Latitude','tif_count_2000',
                                'tif_count_2021'])
k_means = fnc.cluster(cities, features, num_clusters=5)

cities['Cluster'] = k_means.labels_

fnc.plot_clusters(cities)

k_means.labels_
