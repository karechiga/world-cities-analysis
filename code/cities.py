""" cities.py
Uses data aggregated in the csv files located at /../../datasets/csv_data/
"""

import functions as fnc
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import optparse


def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-a', '--analysis',action='store_true',
                            dest='analysis',default=True,
                            help='Clusters the cities based on features located at "../../datasets/csv_data/". '+
                                'Then calculates the cluster stability by running several iterations of clustering.')
    optParser.add_option('-c', '--cluster',action='store_true',
                            dest='cluster_plot',default=False,
                            help='Clusters city data located at "../../datasets/csv_data/", then plots the clusters in a geographic representation.')
    # optParser.add_option('-p', '--pca',action='store_true',
    #                         dest='pca',default=False,
    #                         help='Performs Principal Components Analysis (PCA).')
    optParser.add_option('-k', '--num_clusters',action='store',
                            metavar="K", type='int',dest='num_clusters',default=5,
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
    # num_clusters=5, c_iters=10, b_stab_iters=100, stab_iters=100,
    #                  random_seed=20, drop_feat_perc=10, drop_rows_perc=10
    
    opts, args = optParser.parse_args()

    return opts

if __name__ == '__main__':

    opts = parseOptions()
    path = '../../datasets/csv_data/'
    cities = fnc.get_cities(path)   # Get list of cities with their features.
    features = cities.columns.drop(['City', 'Region', 'Longitude',
                                    'Latitude','tif_count_2000',
                                    'tif_count_2021'])
    if opts.cluster_plot:
        np.random.seed(opts.random_seed)
        # find baseline clusters to compare against
        print("Clustering cities...")
        baseline = fnc.get_baselines(cities, features, num_clusters=opts.num_clusters, iters=opts.cluster_iters)
        # Plotting the baseline clusters:
        cities['Cluster'] = baseline
        fnc.plot_clusters(cities)
        plt.show()
    if not opts.cluster_plot:
        np.random.seed(opts.random_seed)
        print("Analyzing the stability of the calculated baseline clusters...")
        fnc.cluster_analysis(cities, features, num_clusters=opts.num_clusters, c_iters=opts.cluster_iters,
                             b_stab_iters=opts.b_stab_iters, stab_iters=opts.stab_iters,
                             drop_feat_perc=opts.drop_feats_perc, drop_rows_perc=opts.drop_rows_perc)