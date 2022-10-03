import time
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import metrics # for calculating Silhouette score
import matplotlib.pyplot as plt

# APPLICAZIONE K_MEANS
def k_means_alg(data, X, k_clusters):
    start = time.time()
    k_means = KMeans(n_clusters=k_clusters, verbose=False)
    k_means.fit(X)
    song_cluster_labels = k_means.predict(X)
    data['cluster_label'] = song_cluster_labels
    # print(silhouette_score(X[:], song_cluster_labels))
    end = time.time()
    print("Il tempo di esecuzione dell'algoritmo K-Means è: ", round(end - start, 3), "s")
    return data

# APPLICAZIONE K_MEDOIDS
def k_medoids_alg(data, X, k_clusters):
    X = X.to_numpy()
    start = time.time()
    k_means = KMedoids(n_clusters=k_clusters)
    k_means.fit(X)
    song_cluster_labels = k_means.predict(X)
    data['cluster_label'] = song_cluster_labels
    # print(silhouette_score(X[:], song_cluster_labels))
    end = time.time()
    print("Il tempo di esecuzione dell'algoritmo KMedoids è: ", round(end - start, 3), "s")
    return data

# APPLICAZIONE DBSCAN
def dbscan_alg(data, X):
    ''' I valori di eps e min_samples sono stati impostati rispettivamente
    mediante il k-distance graph e la regola empirica (minPts >= num_features+1)'''
    start = time.time()
    clusters = DBSCAN(eps=2.02, min_samples=16)
    song_cluster_labels = clusters.fit_predict(X)
    data['cluster_label'] = song_cluster_labels

    # Contiamo il numero di noise points
    noisePoints = data[data.iloc[:, -1] == -1].value_counts()
    print("Il numero di noise point è: ", len(noisePoints))

    # Rimozione noise points
    data = data[data.iloc[:, -1] != -1]
    # print(silhouette_score(X[song_cluster_labels != -1], song_cluster_labels[song_cluster_labels != -1]))
    end = time.time()
    print("Il tempo di esecuzione dell'algoritmo DBSCAN è: ", round(end - start, 3), "s")
    return data


# APPLICAZIONE GAUSSIAN MIXTURE MODEL
def gmm_alg(data, X):
    start = time.time()
    K = range(2, 7)
    S=[]
    for k in K:
        # Set the model and its parameters
        model = GaussianMixture(n_components=k, n_init=20, init_params='kmeans')
        # Fit the model
        labels = model.fit_predict(X)
        # Calculate Silhoutte Score and append to a list
        S.append(metrics.silhouette_score(X, labels, metric='euclidean'))

    # Plot the resulting Silhouette scores on a graph
    plt.figure(figsize=(16, 8), dpi=300)
    plt.plot(K, S, 'bo-', color='black')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Identify the number of clusters using Silhouette Score')
    plt.show()
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X)
    song_cluster_labels = gmm.predict(X)
    data['cluster_label'] = song_cluster_labels
    end = time.time()
    print("Il tempo di esecuzione dell'algoritmo GMM è: ", round(end - start, 3), "s")

    return data