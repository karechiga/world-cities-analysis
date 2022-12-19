""" cities.py
Uses data aggregated in the csv files located at /../csv_data/
"""

import functions as fnc
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import optparse


def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-s', '--stability',action='store_true',
                            dest='stability',default=False,
                            help='Clusters the cities based on features located at "../csv_data/". '+
                                'Then calculates the cluster stability by running several iterations of clustering.')
    optParser.add_option('-c', '--cluster',action='store_true',
                            dest='cluster_plot',default=False,
                            help='Clusters city data located at "../csv_data/", then plots the clusters in a geographic representation.')
    optParser.add_option('-t', '--centers_to_csv',action='store_true',
                            dest='centers_to_csv',default=False,
                            help='Saves calculated cluster centroids to CSV file (use with python cities.py -c or -p).')
    optParser.add_option('-p', '--pca',action='store_true',
                            dest='pca',default=False,
                            help='Clusters then plots the cities with their first two Principle Components')
    optParser.add_option('-e', '--elbow',action='store_true',
                            dest='elbow',default=False,
                            help='Uses centroid CSV files for different k values, and Euclidean calculates distances between cities and clusters.\n'+
                                'Plots an Elbow plot and generates a csv for cluster distances and city distances from each other.')
    optParser.add_option('-k', '--num_clusters',action='store',
                            metavar="K", type='int',dest='num_clusters',default=6,
                            help='Number of clusters to partition the data in k-means clustering. Default: %default' )
    optParser.add_option('-i', '--cluster_iters',action='store',
                            metavar="C", type='int',dest='cluster_iters',default=10,
                            help='Number of clustering iterations to calculate baseline clusters. Default: %default')
    optParser.add_option('-b', '--baseline_stability_iters',action='store',
                            metavar="B", type='int',dest='b_stab_iters',default=100,
                            help='Number of iterations to calculate the stability of the baseline clusters. Default: %default' )
    optParser.add_option('-n', '--stability_iters',action='store',
                            metavar="N", type='int',dest='stab_iters',default=100,
                            help='Number of iterations to calculate the cluster stability for each city. Default: %default' )
    optParser.add_option('-f', '--drop_features',action='store',
                            metavar="F", type='int',dest='drop_feats_perc',default=10,
                            help='Percentage of features to drop at each iteration of stability calculations. Default: %default' )
    optParser.add_option('-d', '--drop_rows',action='store',
                            metavar="D", type='int',dest='drop_rows_perc',default=10,
                            help='Percentage of rows to drop at each iteration of stability calculations. Default: %default' )
    optParser.add_option('-r', '--random_seed',action='store',
                            metavar="R", type='int',dest='random_seed',default=10,
                            help='Random seed initialization. Default: %default' )

    opts, args = optParser.parse_args()

    return opts

if __name__ == '__main__':

    opts = parseOptions()
    path = '../csv_data/'
    start_cols = ['City', 'Region', 'Longitude',
                    'Latitude','tif_count_2000',
                    'tif_count_2021', 'Area']
    cities = fnc.get_cities(path)   # Get list of cities with their features.
    cities = cities.loc[:,start_cols + list(cities.columns.drop(start_cols))]
    features = cities.columns.drop(['City', 'Region', 'Longitude',
                                    'Latitude','tif_count_2000',
                                    'tif_count_2021'])
    if opts.elbow:
        # Calculates WSS and plots K-Means Elbow plot
        # Only works if cluster data has already been output already to CSV
        fnc.elbow_method(cities, features, k=opts.num_clusters)
    if opts.cluster_plot:
        np.random.seed(opts.random_seed)
        # find baseline clusters to compare against
        print("Clustering cities...")
        labels = fnc.get_baselines(cities, features, num_clusters=opts.num_clusters,
                                    iters=opts.cluster_iters, centers_to_csv=opts.centers_to_csv)
        # Plotting the clusters:
        cities['Cluster'] = labels
        fnc.plot_clusters(cities)
        plt.show()
    if opts.pca:
        # Principle Components plot
        np.random.seed(opts.random_seed)
        print("Clustering cities, then performing PCA.")
        labels = fnc.get_baselines(cities, features, num_clusters=opts.num_clusters,
                                    iters=opts.cluster_iters,centers_to_csv=opts.centers_to_csv)
        cities['Cluster'] = labels
        pca, components = fnc.pca_2d(cities, features)
        print('Explained variance - PC1: {}, PC2: {}.'.format(
                pca.explained_variance_ratio_[0],pca.explained_variance_ratio_[1]))
        cities['PC1'], cities['PC2'] = components.T
        fnc.plot_pca(cities, features, pca)
        plt.show()
    if opts.stability:
        np.random.seed(opts.random_seed)
        print("Analyzing the stability of the calculated baseline clusters...")
        fnc.stability_analysis(cities, features, num_clusters=opts.num_clusters, c_iters=opts.cluster_iters,
                             b_stab_iters=opts.b_stab_iters, stab_iters=opts.stab_iters,
                             drop_feat_perc=opts.drop_feats_perc, drop_rows_perc=opts.drop_rows_perc)