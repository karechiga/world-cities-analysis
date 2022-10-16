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
iters = 10000
num_clusters = 5
theshold = 0.5

# find baseline clusters to compare against
baseline = fnc.get_baselines(cities, features)
# Plotting the baseline clusters:
cities['Cluster'] = baseline
fnc.plot_clusters(cities)
# randomly removing features from the feature set
drop_feats = 3 # max number of features to randomly drop
# and clustering to determine the overall stability for each city.
stability = np.zeros(shape=(len(baseline)))
for i in range(iters):
    to_drop = np.random.randint(len(features), size=drop_feats)
    new_feats = features.drop(features[to_drop])
    stability += fnc.cluster_stability(cities, new_feats, baseline)
stability = stability / iters

df = pd.merge(cities['City'], pd.DataFrame(stability, columns=["Stability"]),left_index=True, right_index=True)
df.to_csv(path + 'csv_data/landscan_cities.csv',index=False)
