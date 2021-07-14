from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def compute_update(centroids, recv_data, rank):
    '''
    Computes the distance matrix, assigns points to centroids and updates the centroids
    :param centroids: dataframe of centroids
    :param recv_data: dataframe of data points
    :return: returns the updated centroids
    '''
    centroid_dict = {}  # the dictionary stores list of data points for each centroid
    k = len(centroids)

    # Computes distance matrix
    dist_matrix = euclidean_distances(recv_data, centroids)

    # Assigns points to centroids
    closest_centroids = [np.argmin(i) for i in dist_matrix]

    # Updates centroids
    centroid_ids = range(0, k)
    for centroid_id in centroid_ids:
        members = [i for i, x in enumerate(closest_centroids) if x == centroid_id]
        centroid_dict[centroid_id] = members
        # update the corresponding centroid by taking mean of its members
        centroids.iloc[centroid_id] = recv_data.iloc[members].mean()

    return centroids

def find_membership(centroids, recv_data):
    '''
    Assigns data points to the cluster by finding the minimum distance
    :param centroids:
    :param recv_data:
    :return:
    '''

    centroid_dict = {}  # the dictionary stores list of data points for each centroid
    k = len(centroids)

    # Computes distance matrix
    dist_matrix = euclidean_distances(recv_data, centroids)

    # Assigns points to centroids
    closest_centroids = [np.argmin(i) for i in dist_matrix]

    centroid_ids = range(0, k)
    for centroid_id in centroid_ids:
        members = [i for i, x in enumerate(closest_centroids) if x == centroid_id]
        centroid_dict[centroid_id] = members

    # Calculating Silhouette score - Measure of similarity to its own cluster and dissimilarity from other clusters

    score = silhouette_score(recv_data, closest_centroids, metric = 'euclidean')
    print("silhouette_score-", score)

def plot_line(results_path="results_kmeans.csv"):
    df_plot = pd.read_csv(results_path)
    sns.lineplot("processes", "time", data=df_plot, hue="k",
                 palette="tab10", marker="o")
    plt.title("Time vs #Processes and K")
    #plt.show()
    plt.savefig("plot_time.png", dpi=720)
    plt.figure()
    sns.lineplot("processes", "Silhouette_score", data=df_plot, hue="k",
                 palette="tab10", marker="o")
    plt.title("Silhoutte score vs #Processes and K")
    #plt.show()
    plt.savefig("plot_score.png", dpi=720)
