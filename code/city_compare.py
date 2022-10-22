""" city_compare.py
Uses data aggregated in the csv files located at /../../datasets/csv_data/
"""

import functions as fnc
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
from itertools import combinations

np.random.seed(20)
path = '../../datasets/csv_data/'
cities = fnc.get_cities(path)   # Get list of cities with their features.
features = cities.columns.drop(['City', 'Longitude', 
                                'Latitude','tif_count_2000',
                                'tif_count_2021'])
b_iters = 1000     # clustering iterations for calculating baseline stability
iters = 10000      # clustering iterations for calculating jackknife stability
num_clusters = 5

# find baseline clusters to compare against
baseline = fnc.get_baselines(cities, features)
# Plotting the baseline clusters:
cities['Cluster'] = baseline
fnc.plot_clusters(cities)
# Baseline cluster stability:
b_stability = fnc.cluster_stability(cities, features, baseline, iters=b_iters)
cities['Baseline_Stability'] = b_stability
# randomly removing features, rows from the feature set
drop_feat_perc = 10 # percentage of features to randomly drop.
drop_feats = int(np.floor(drop_feat_perc*len(features))/100)
drop_rows_perc = 10  # percentage of rows to randomly drop.
drop_rows = int(np.floor(drop_rows_perc*len(cities))/100)

# initializing stability
stability = np.zeros(shape=(len(baseline)))
count = np.zeros(shape=(len(baseline)))
for i in range(iters):
    to_drop = np.random.randint(len(features), size=drop_feats)
    new_feats = features.drop(features[to_drop])
    to_drop = np.random.randint(len(cities), size=drop_rows)
    new_cities = cities.drop(to_drop,axis=0)
    count[new_cities.index] += 1
    stability += fnc.cluster_stability(new_cities, new_feats, baseline)
stability = stability / count
cities['Stability'] = stability
# Saving to CSV
cities[['City','Cluster','Baseline_Stability','Stability']].to_csv('../cluster_data/cluster_stabilities.csv',index=False)
cities.to_csv('../cluster_data/cities.csv',index=False)
