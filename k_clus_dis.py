'''
Author - Jaishree Janu
Description - This script contains the code for distributed K-means clustering
'''

from mpi4py import MPI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from utilities import compute_update, find_membership, plot_line
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Global params
k = 2  # No. of clusters | possible values could be [1,2,3,4,6,8]
prev_distance = np.zeros((k,k))  ## stores previous displacement between new and old centroids


print("Entering into worker no. -- ", rank)
start_time = time.time()

if rank == 0:
    # Read csv data file into a dataframe
    df = pd.read_csv('Absenteeism_at_work.csv', delimiter=';')

    # Slicing the dataframes
    data_size = len(df)
    quo, rem = divmod(data_size, size)

    counts = [quo + 1 if p < rem else quo for p in range(size)]

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(size)]
    ends = [sum(counts[:p + 1]) for p in range(size)]

    # converts data into a list of arrays
    sliced_data = [df[starts[p]:ends[p]] for p in range(size)]

    # Randomly sample k points from df
    centroids = df.sample(k)
    centroids.reset_index(inplace = True, drop=True)
    converging_cond = False

else:
    sliced_data = None
    centroids = None
    converging_cond = None

recv_data = comm.scatter(sliced_data, root=0)  # needs to be scattered only once - kept outside of loop
centroids = comm.bcast(centroids, root=0)
converging_cond = comm.bcast(converging_cond, root=0)


iteration = 0

while not converging_cond:
    # Compute euclidean distance of each data point from every centroid

    print("Rank after broadcast",rank)
    up_centroids = compute_update(centroids, recv_data, rank)
    comm.barrier() # Wait to finish for all workers
    # Gather centroids from all workers
    all_centroids = comm.gather(up_centroids, root=0)


    # calculate global mean
    if rank == 0:

        # Randomly initialize new centroid of clusters
        list_cols = centroids.columns
        no_params = len(list_cols)
        data = np.random.randint(1, 100, size=(k, no_params))
        new_centroids = pd.DataFrame(data, columns=list_cols).astype(float)

        for centroid in range(0,len(centroids)):
            find_avg = pd.DataFrame([all_centroids[p].iloc[centroid] for p in range(0,size)])
            new_centroids.iloc[centroid] = find_avg.mean()
            # if no point assigned to this centroid, randomly choose another point
            if new_centroids.iloc[centroid] is None:
                new_centroids.iloc[centroid] = recv_data.sample(1)


        # Finding displacement of new_centroids with respect to previous centroids
        distance = euclidean_distances(new_centroids, centroids)

        if prev_distance.any():
            # The code for convergence criteria
            update_quantities = [distance[clus][clus]-prev_distance[clus][clus] for clus in range(0,k)]


            ## Heuristic, 1 is a minute displacement in this dataset
            heuristic_change = np.float32(1.0)
            check = [i for i in update_quantities if i >= heuristic_change]
            print("check", check)
            if len(check) == 0:
                converging_cond = True

        prev_distance = distance # Store the previous distance

        centroids = new_centroids

    # broadcast everyone about convergence, so that others dont do computation after convergence
    converging_cond = comm.bcast(converging_cond, root=0)
    centroids = comm.bcast(centroids, root=0)
    comm.barrier()  # It is used because next iteration must start only when updated centroids have reached to everyone
    iteration += 1


print("Converged after ", iteration, "iterations")

# To determine the goodness of the clusters formed
if rank==0:
    df = pd.read_csv('Absenteeism_at_work.csv', delimiter=';')
    # The function calculates distance of each data point from final centroids, finds minimum distane and calculates Silhouette score
    find_membership(centroids,df)
    # Call plot_line() function to draw plots
    plot_line()

end_time = time.time()
print("Time taken for rank ", rank,"--", end_time-start_time)
comm.barrier()


