"""
Functions to be used in the city_compare.py script.
"""
import pandas as pd
import os
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn import preprocessing
from sklearn.decomposition import PCA
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
                    cities = cities.drop(['Region', 'Latitude', 'Longitude', 'tif_count'],axis=1)
                first_file = False
            else:
                # add to the cities df by merging with the new csv
                new_csv = pd.read_csv(path + f)
                if not 'landscan' in f:
                    # Only using the Latitude and Longitude provided in the Landscan csv.
                    new_csv = new_csv.drop(['Region', 'Latitude', 'Longitude', 'tif_count'],axis=1)
                cities = pd.merge(cities, new_csv,on='City')
    return cities

def cluster(df, features, num_clusters = 5, method = 'kmeans'):
    """
    Returns scikitlearn cluster object (default: uses kmeans)
    """
    X = df[features].values   # returns an array of the feature values
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    # X_norm = preprocessing.normalize(X_scaled, norm='l2', axis=1)
    if method.lower() == 'dbscan':
        return DBSCAN(eps=0.42, min_samples=350).fit(X_scaled)
    return KMeans(n_clusters=num_clusters).fit(X_scaled)

def pca_2d(df, features):
    """
    Performs Principle Components Analysis (PCA) on the given df[features]
    """
    X = df[features].values   # returns an array of the feature values
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    # X_norm = preprocessing.normalize(X_scaled, norm='l2', axis=1)
    pca = PCA(n_components=2)
    pc = pca.fit_transform(X_scaled)
    return pca, pc

def plot_pca(cities, method='kmeans'):
    """
    Input: "cities" dataframe should contain columns for "PC1" and "PC2".
            Optionally, should also include cluster column as "Cluster".
    """
    unique, counts = np.unique(cities['Cluster'].values, return_counts=True)
    plt.figure()
    ax = plt.subplot(111)
    points = []
    for k in range(len(unique)):
        df = cities[cities['Cluster'] == k+1]
        points.append(ax.plot(df['PC1'], df['PC2'],
                        '.',markersize=1.5, alpha=0.7,
                        label='Cluster {}'.format(unique[k])))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                    box.width, box.height * 0.9])
    ax.legend(handles=[points[k][0] for k in range(len(unique))],loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=3, fancybox=True, shadow=True, markerscale=5)
    ax.set_title('2D Principle Components of {} City Clusters ({})'.
                 format(len(unique), method.upper()), x=0.5, y=1.13)
    ax.set_xlabel('Principle Component 1')
    ax.set_ylabel('Principle Component 2')
    # plt.xlim([-180, 180])
    # plt.ylim([-90, 90])
    plt.savefig('../figures/2d_PCA_{}_{}.png'.format(method, len(unique)))

def plot_clusters(cities, method='kmeans'):
    unique, counts = np.unique(cities['Cluster'].values, return_counts=True)
    if len(unique) == 5:
        # For 5 clusters plot, set the five colors.
        colors = ['salmon', 'gold', 'limegreen', 'cornflowerblue', 'mediumorchid']
    else:
        colors = ['']*20
    plt.figure()
    bar_ax = plt.subplot(111)
    if len(unique) == 5:
        bar_ax.bar(unique,counts,color=colors)  # Plots the number of occurrences for each cluster
    else:
        bar_ax.bar(unique,counts)
    bar_ax.set_title('City Cluster Counts ({} clustering)'.format(method.upper()))
    bar_ax.set_xlabel('Cluster Numbers')
    bar_ax.set_ylabel('Counts')
    plt.savefig('../figures/{}_{}_city_clusters_counts.png'.format(method.upper(), len(unique)))

    # plt.show()
    plt.figure()
    ax = plt.subplot(111)
    points = []
    for k in range(len(unique)):
        df = cities[cities['Cluster'] == k+1]
        if len(unique) != 5:
            points.append(ax.plot(df['Longitude'], df['Latitude'],
                                '.',markersize=1.5, alpha=0.7,label='Cluster {}'.format(unique[k])))
        else:
            points.append(ax.plot(df['Longitude'], df['Latitude'],
                                '.',color=colors[k],markersize=1.5, alpha=0.7,label='Cluster {}'.format(unique[k])))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                    box.width, box.height * 0.9])
    ax.legend(handles=[points[k][0] for k in range(len(unique))],loc='upper center', bbox_to_anchor=(0.5, 1.1),
            ncol=3, fancybox=True, shadow=True, markerscale=5)
    ax.set_title('Geospatial Representation of {} City Clusters'.format(len(unique)), x=0.5, y=1.11)
    ax.set_xlabel('Longitude (Degrees)')
    ax.set_ylabel('Latitude (Degrees)')
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.savefig('../figures/{}_{}_city_clusters.png'.format(len(unique), method))

def get_baselines(cities, features, num_clusters=5, iters=1000):
    """
    Calculates baseline clusters for each city by running k-means
    clustering "iters" (default 1000) number of times. The cluster centers
    are averaged from each iteration, then cities are labeled based on the
    minimum Euclidean distance to each cluster.

    Args:
        cities (DataFrame): DataFrame of the cities and its features.
        features (Pandas Series of strings): The specified features to use
        for clustering.
        num_clusters (int, optional): Number of clusters. Defaults to 5.
        iters (int, optional): Number of clustering iterations. Defaults to 1000.

    Returns:
        labels (1D NumPy array): the 1D list of city cluster labels.
    """
    # run k-means clustering multiple times, then average cluster centers to get true baseline
    centers = np.zeros(shape=(num_clusters, len(features)))
    for i in range(iters):
        k_means = cluster(cities, features, num_clusters=num_clusters)
        # Cluster numbers are random in each iteration, so to keep the cluster labels
        # consistently ordered, sort by each cluster's sum of their center's features.
        sums = np.array([np.sum(x) for x in k_means.cluster_centers_])
        centers += k_means.cluster_centers_[np.argsort(sums)]
    centers = centers/iters   # average cluster centers
    scaler = preprocessing.StandardScaler().fit(cities[features].values)
    X_scaled = scaler.transform(cities[features].values)
    # normalized = preprocessing.normalize(cities[features].values, norm='l2')
    # Calculate Euclidean distance from average cluster centers
    # Find the minimum Euclidean distance for each city,
    # assign city to that cluster label.
    labels = []
    for i, n in enumerate(X_scaled):
        dist = np.linalg.norm(n - centers[0])
        labels.append(1)
        for j in range(1, num_clusters):
            d = np.linalg.norm(n - centers[j])
            if d < dist:
                dist = d
                labels[i] = j + 1
    return np.array(labels)

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
    neighbors = np.zeros(shape=(len(baseline), len(baseline)))  # array of zeros
    b_neighbors = np.zeros(shape=(len(baseline), len(baseline)))  # array of zeros
    labels = np.zeros(shape=(len(baseline),))  # 1D array, labels for each city
    for j in df.index: b_neighbors[j] = baseline[j] == baseline
    # neighbors = binary square matrix representing cities that are in the same cluster
    # b_neighbors = baseline square matrix representing cities that are in the same cluster for the baseline
    # if city i is in the same cluster as city j, old_neighbors[i,j] = 1
    stability = np.zeros(shape=(len(baseline),))
    for _ in range(iters):
        k_means = cluster(df, features, num_clusters=num_clusters)
        labels[df.index] = k_means.labels_ + 1
        # starting the labels at 1, not 0
        for j in df.index: neighbors[j] = labels[j] == labels
        stability += np.sum(neighbors * b_neighbors, axis=0) / np.sum(b_neighbors, axis=0)
        # stability => percent of baseline neighbors remaining in the same cluster as the city
    return stability / iters

def stability_analysis(cities, features, num_clusters=5, c_iters=10, b_stab_iters=100, 
                     stab_iters=100, drop_feat_perc=10, drop_rows_perc=10):
    """
    Reads in data features for all cities in the dataset and performs
    multiple K-Means clustering iterations to determine the stability of each cluster.
    """
    # find baseline clusters to compare against
    baseline = get_baselines(cities, features, num_clusters, c_iters)
    cities['Cluster'] = baseline
    # Baseline cluster stability:
    b_stability = cluster_stability(cities, features, baseline, iters=b_stab_iters)
    cities['Baseline_Stability'] = b_stability
    # randomly removing features, rows from the feature set
    drop_feats = int(np.floor(drop_feat_perc*len(features))/100)
    drop_rows = int(np.floor(drop_rows_perc*len(cities))/100)

    # initializing stability
    stability = np.zeros(shape=(len(baseline)))
    count = np.zeros(shape=(len(baseline)))
    for _ in range(stab_iters):
        to_drop = np.random.randint(len(features), size=drop_feats)
        new_feats = features.drop(features[to_drop])
        to_drop = np.random.randint(len(cities), size=drop_rows)
        new_cities = cities.drop(to_drop,axis=0)
        count[new_cities.index] += 1
        stability += cluster_stability(new_cities, new_feats, baseline)
    stability = stability / count
    cities['Stability'] = stability
    # Saving to CSV
    # cities[['City','Cluster','Baseline_Stability','Stability']].to_csv('../cluster_data/cluster_stabilities_{}.csv'.format(num_clusters),index=False)
    cities.to_csv('../cluster_data/cities_{}.csv'.format(num_clusters),index=False)
    return
