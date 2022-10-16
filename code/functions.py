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
    return KMeans(n_clusters=num_clusters).fit(X_norm)

def plot_clusters(cities):
    unique, counts = np.unique(cities['Cluster'].values, return_counts=True)
    plt.figure()
    bar_ax = plt.subplot(111)
    bar_ax.bar(unique,counts)  # Plots the number of occurrences for each cluster
    bar_ax.set_title('City Cluster Counts')
    bar_ax.set_xlabel('Cluster Numbers')
    bar_ax.set_ylabel('Counts')
    plt.savefig('../figures/{}_city_clusters_counts.png'.format(len(unique)))

    # plt.show()
    plt.figure()
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

def get_baselines(cities, features, num_clusters=5):
    # run k-means clustering 5-10 times, then cluster those results to get the true baseline
    labels = np.zeros(shape=(10, len(cities.index)))
    for i in range(10):
        k_means = cluster(cities, features, num_clusters=num_clusters)
        labels[i] = k_means.labels_
    return KMeans(n_clusters=num_clusters).fit(labels.T).labels_

def cluster_stability(df, features, baseline, num_clusters=5, iters=1):
    """
    This function calculates the stability of each df element's cluster.
    In each iteration, the data is clustered, and the stability is calculated
    by determining the percentage of its old neighbors that remain in the new cluster.
    Args:
        df (DataFrame): DataFrame to be clustered.
        features (List): List of the features used in the df
        baseline (numpy 1d array): List of the baseline clusters that each city is in.
        num_clusters (int): number of clusters. Defaults to 5.
        iters (int) (optional): Number of clustering iterations. Defaults to 1.
    Returns numpy 1d vector representing the stability for each city.
    """
    # for each iteration, recluster , determine the error for each label
    # Calculate for each city how many old neighbors are in the same cluster still in each iteration
    # Average the stability out over the number of iterations. Weed out cities with stability less than threshold
    neighbors = np.zeros(shape=(len(baseline), len(baseline)))  # 6018 x 6018 array of zeros
    b_neighbors = np.zeros(shape=(len(baseline), len(baseline)))  # 6018 x 6018 array of zeros
    for j in range(len(baseline)): b_neighbors[j] = baseline[j] == baseline
    # neighbors = binary square matrix representing cities that are in the same cluster
    # b_neighbors = baseline square matrix representing cities that are in the same cluster for the baseline
    # if city i is in the same cluster as city j, old_neighbors[i,j] = 1
    stability = np.zeros(shape=(len(df.index),))
    for _ in range(iters):
        k_means = cluster(df, features, num_clusters=num_clusters)
        labels = k_means.labels_
        for j in range(len(neighbors)): neighbors[j] = labels[j] == labels
        stability += np.sum(neighbors * b_neighbors, axis=0) / np.sum(b_neighbors, axis=0)
        # stability => percent of baseline neighbors remaining in the same cluster as the city
    return stability / iters
