import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

## method reads in and formats data
def read_data(filename):
    ## read data in using space as a delimeter
    seeds = pd.read_csv(filename, sep='\s+')
    ## drop the last column
    seeds = seeds.drop(seeds.columns[-1], axis=1)
    ## return the formatted data
    return seeds

## method calculates euclidian distance of two vectors
def e_distance(vector1, vector2):
    ## make the passed in data vectors
    vector1 = np.asarray(vector1)
    vector2 = np.asarray(vector2)
    
    ## return euclidian distance
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

## method separates the data frame by labels passed in 
def separate_data(labels, data):
    
    ## make dataframe to hold labels
    clustered_data = []

    ## loop through the unique cluster labels
    for cluster_label in np.unique(labels):
        ## find where the data labels match the cluster labels
        cluster_data = data[labels == cluster_label]
        ## append to list
        clustered_data.append(cluster_data)
    
    return clustered_data

## method returns the sses and means of each clustered data point to a centroid
def get_sses_and_mean(clustered_data, centroids):

    ## make a list for calculated sse and mean
    sses = []
    means = []

    ## loop through the clustered data
    for cluster_idx, cluster_data in enumerate(clustered_data):
        ## set sse  value to zero
        sse = 0
        ## loop through each point in the cluster
        for _, point in cluster_data.iterrows():
            ## get euclidean distance and square it
            sse += np.square(e_distance(point, centroids[cluster_idx].flatten()))
        
        ## append the sse to list
        sses.append(sse)

        ## get the mean of the cluster
        means.append(cluster_data.mean())

    return sses, means

## method calculates the k mean for every k that is passed in
def calculate_kmeans(data, k_array, type):
    ## make a list to hold the results
    clusteredData = []
    ## loop through the indexes provided
    for k in k_array:
        ## initialize the kmeans method
        if type == "b": 
            kmeans = KMeans(n_clusters=k, n_init=1)
        else:
            kmeans = KMeans(n_clusters=k, init='random', n_init=1)
        ## fit the data
        kmeans.fit(data)
        ## append the fitted data
        clusteredData.append(kmeans)

    return clusteredData

## method formats and prints out data about a basic kmeans instance
def print_data_basic(data, calculated_kmeans, k):

    ## get sum of squared errors for trial
    total_SSE = calculated_kmeans.inertia_
    ## get the cluster labels for each data point
    labels = calculated_kmeans.labels_
    ## get the centroids for each data point
    centroids = calculated_kmeans.cluster_centers_

    ## get the datapoints for each label
    clustered_data = separate_data(labels, data)
        
    ## get the sses and means for each cluster
    sses, means = get_sses_and_mean(clustered_data, centroids)
        
    ## begin printing
    print("KMean at ", k)
    print("Total SSE: ", total_SSE )

    ## loop through and format printing
    for cluster in range(0, len(clustered_data)):
        print("Cluster ", cluster + 1)
        print("Cluster SSE: ", sses[cluster])
        print("Cluster Mean: ", means[cluster])
        print("Centroid ID: ", centroids[cluster])
        print("Members of Cluster: ", clustered_data[cluster])

## method formats and prints out data about bisecting kmeans
def print_data_bisecting(clusters):
   ## for total sse 
   total_SSE = 0
   
   ## sort the clusters
   sorted_clusters = dict(sorted(clusters.items()))

   ## loop through dictionary
   for key, value in sorted_clusters.items():
       ## add to total SSE
       total_SSE += value['SSE']
       
       ## print out values
       print("Cluster ", key)
       print("Cluster SSE: ", value['SSE'])
       print("Cluster Mean: ", value['Mean'])
       print("Centroid ID: ", value['Centroid'])
       print("Members of Cluster: ", value['Datapoints'])

   print("Total SSE: ", total_SSE)
   
## method performs all of the calculations and printing for basic kMeans algorithm (3.2)
def basic_kMeans(data, k_array):

    ## run the calcuclations
    calculated_kmeans = calculate_kmeans(data, k_array, "b")

    ## make a list for the sses
    sees = []
    ## print the data
    ## loop through the indexes provided
    for i in range(len(k_array)):
        print_data_basic(data, calculated_kmeans[i], k_array[i])
        sees.append(calculated_kmeans[i].inertia_)
        

    plt.plot(k_array, sees, marker='o')
    plt.title('Basic KMeans')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

## method performs all of calculations and printing for bisecting kMeans algorithm (3.3)
def bisecting_kMeans(data, k, number_of_trials): 
    
    ## define a cluster of all datapoints
    clusters = {}
    next_cluster_id = 2
    firstCluster = calculate_kmeans(data, [1], "n")
    firstData = separate_data(firstCluster[0].labels_, data)
    firstsse, firstmean = get_sses_and_mean(firstData, firstCluster[0].cluster_centers_)
    clusters[1] = {
        'SSE': firstCluster[0].inertia_,
        'Datapoints': firstData[0],
        'Mean': firstmean, 
        'Centroid': firstCluster[0].cluster_centers_[0]
    }

    ## until we reach our k value
    while len(clusters) < k:

        ## define variables for control and work
        max_SSE = -float('inf')
        cluster_to_split_key = None

        ## find the dictionary value with the highest SSE
        ## loop through dictionary
        for key, value in clusters.items():
            ## if this is the largest SSE
            if value['SSE'] >= max_SSE:
                ## assign the appropriate variables
                max_SSE = value['SSE']
                cluster_to_split_key = key
        
        ## get the data of the highest SSE
        working_data = clusters[cluster_to_split_key]['Datapoints']
        
        ## define variables for control and work
        min_SSE = float('inf')
        best_split = None

        ## perform the set number of trials
        for i in range(0, number_of_trials):

            ## calculate the kmeans
            perform_split = calculate_kmeans(pd.DataFrame(working_data), [2], "n")
            current_split = perform_split[0] ## since only one trial was performed, need the first returned model

            ## determine if this split is best
            if current_split.inertia_ <= min_SSE:
                min_SSE = current_split.inertia_
                best_split = current_split
        
        ## take the best split and add it to the dictionary
        labelled_data = separate_data(best_split.labels_, working_data)
        current_sses, current_means = get_sses_and_mean(labelled_data, best_split.cluster_centers_)

        del clusters[cluster_to_split_key]

        clusters[cluster_to_split_key] = {
            'SSE': current_sses[0],
            'Datapoints': labelled_data[0],
            'Mean': current_means[0], 
            'Centroid': best_split.cluster_centers_[0]
        }

        clusters[next_cluster_id] = {
            'SSE': current_sses[1],
            'Datapoints': labelled_data[1],
            'Mean': current_means[1], 
            'Centroid': best_split.cluster_centers_[1]
        }

        # Increment cluster ID for the next unique cluster
        next_cluster_id += 1

    
    print_data_bisecting(clusters)

## method performs the single link min hierarchical clustering algorithm
def hierarchical_clustering(data, k):

    ## make the model specifying the single method and euclidean metric
    model = linkage(data, method='single', metric='euclidean')

    ## set figure size
    plt.figure(figsize=(10, 5))
    ## model turned into dendrogram, specify the options to cut it at level 6
    dendrogram(model, truncate_mode='level', p=k, show_leaf_counts=True)
    plt.title(f"Dendrogram (Cutoff at {k} Clusters)")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()

    ## divide into clusters
    cluster_labels = fcluster(model, k, 'maxclust')

    ## get data points for each cluster
    labelled_data = separate_data(cluster_labels, data)

    ## get centroids of labelled data
    centroids = []
    for i in range(0, len(labelled_data)):
        centroids.append(labelled_data[i].mean())

    ## get the sses and means
    sses, means = get_sses_and_mean(labelled_data, np.array(centroids))

    total_SSE = 0

    for i in range(0, len(labelled_data)):
        ## add to total SSE
        total_SSE += sses[i]
       
        ## print out values
        print("Cluster ", i + 1)
        print("Cluster SSE: ", sses[i])
        print("Cluster Mean: ", means[i])
        print("Centroid ID: ", centroids[i])
        print("Members of Cluster: ", labelled_data[i])

    print("Total SSE: " , total_SSE)


    
def main():

    ## make array of clusters to make
    k_array = [2, 3, 4, 5, 6]

    ## read in data
    data = read_data('seeds_dataset.txt')
    ## call basic algorithm
    basic_kMeans(data, k_array)
    ## call bisecting algorithm
    bisecting_kMeans(data, 6, 3)
    ## call hierarchical clustering 
    hierarchical_clustering(data, 6)
    
    

if __name__=="__main__":
    main()

