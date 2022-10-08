"""
Functions to be used in the city_compare.py script.
"""
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot as plt
import numpy as np

def get_cities(path):
    """
    Reads the csv data located at "path", aggregates the features
    into a single dataframe and returns the dataframe.
    """
    cities = pd.DataFrame()
    files = os.listdir(path)
    # boolean to specify that this is the first file in the iteration
    first_file = True
    for f in files:
        if f[-4:] == '.csv':
            if first_file:
                # Initialize the cities df
                cities = pd.read_csv(path + f)
                if not 'landscan' in f:
                    # Only using the Latitude and Longitude provided in the Landscan csv.
                    cities = cities.drop(['Latitude', 'Longitude', 'tif_count'],axis=1)
                first_file = False
            else:
                # add to the cities df by merging with the new csv
                new_csv = pd.read_csv(path + f)
                if not 'landscan' in f:
                    # Only using the Latitude and Longitude provided in the Landscan csv.
                    new_csv = new_csv.drop(['Latitude', 'Longitude', 'tif_count'],axis=1)
                cities = pd.merge(cities, new_csv,on='City')
    return cities

def cluster(df, features, num_clusters):
    """
    Returns KMeans clustering scikitlearn object
    """
    X = df[features].values   # returns an array of the feature values
    X_norm = preprocessing.normalize(X, norm='l2')
    return KMeans(n_clusters=num_clusters, random_state=0).fit(X_norm)

def plot_clusters(cities):
    
    unique, counts = np.unique(cities['Cluster'].values, return_counts=True)
    # plt.bar(unique,counts)  # Plots the number of occurrences for each cluster
    # plt.show()
    fig = plt.figure()
    ax = plt.subplot(111)
    points = []
    for k in unique:
        df = cities[cities['Cluster'] == k]
        points.append(ax.plot(df['Longitude'], df['Latitude'],
                            '.',markersize=1.5, alpha=0.7,label='Cluster {}'.format(k)))
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                    box.width, box.height * 0.9])
    ax.legend(handles=[points[k][0] for k in unique],loc='upper center', bbox_to_anchor=(0.5, 1.1),
            ncol=3, fancybox=True, shadow=True, markerscale=5)
    ax.set_title('Geospatial Representation of {} City Clusters'.format(len(unique)), x=0.5, y=1.11)
    ax.set_xlabel('Longitude (Degrees)')
    ax.set_ylabel('Latitude (Degrees)')
    plt.savefig('../figures/{}_city_clusters.png'.format(len(unique)))