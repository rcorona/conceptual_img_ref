from sklearn.cluster import MiniBatchKMeans, DBSCAN
from pyitlib import discrete_random_variable as drv
import scipy
from collections import Counter
import pickle
import os
import argparse
import sys
import json
import numpy as np
import csv

def parse_args(): 

    parser = argparse.ArgumentParser(description='Run random search of hyperparameters.')

    parser.add_argument('-results_dir', default='.', type=str,
                        help='Parent directory where results should be stored.')
    parser.add_argument('-procedure', default='batch_experiments', type=str,
                        help='Routine to run with script. [batch_experiments, plot]')
    parser.add_argument('-figure_dir', default='./figs', type=str,
                        help='Folder to store figures in.')
    parser.add_argument('-n_points', default=10000, type=int,
                        help='Number of embeddings to use for clustering.')
    
    return parser.parse_args()

def cluster(args):

    # Load variant in order to get hyperparameters.
    variant_path = os.path.join(args.results_dir, 'variant.json')
    variant = json.load(open(variant_path, 'r'))

    n_clusters = variant['n_pops']
    
    # First load agend embeddings on which to perform clustering.
    embeddings_path = os.path.join(args.results_dir, 'agent_embeddings.pkl')
    embeddings = pickle.load(open(embeddings_path, 'rb'))

    # Run clustering experiment for each evaluation interval.
    results = {idx: {} for idx in embeddings}

    for idx in embeddings:

        agent_embeddings = embeddings[idx]['embeddings'][:args.n_points]
        labels = embeddings[idx]['labels'][:args.n_points]
        
        # Run k-means to get unsupervised cluster labels. 
        kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(agent_embeddings)
        km_predict = kmeans.predict(agent_embeddings)

        # Random baseline.
        rand_predict = np.random.choice(n_clusters, size=args.n_points)
        
        # Do the same with DBSCAN. 
        db_clustering = DBSCAN(eps=0.05).fit(agent_embeddings)
        db_predict = db_clustering.labels_

        iv_kmeans = drv.information_variation(km_predict, labels)
        iv_rand = drv.information_variation(rand_predict, labels)
        iv_db = drv.information_variation(db_predict, labels)
        print('Idx: {} IV kmeans: {} IV db: {} IV random: {}'.format(idx, iv_kmeans, iv_db, iv_rand))

        # Keep track of results so we can save them.
        results[idx]['kmeans'] = iv_kmeans
        results[idx]['db'] = iv_db
        results[idx]['rand'] = iv_rand
        
    # Store results in a file.
    csv_path = os.path.join(args.results_dir, 'cluster_results.csv')
    writer = csv.writer(open(csv_path, 'w'))
    writer.writerow(['idx', 'kmeans', 'db', 'rand'])

    for idx in results:
        row = [idx, results[idx]['kmeans'], results[idx]['db'], results[idx]['rand']]
        writer.writerow(row)

if __name__ == '__main__':
    
    args = parse_args()

    if args.procedure == 'cluster':
        cluster(args)
