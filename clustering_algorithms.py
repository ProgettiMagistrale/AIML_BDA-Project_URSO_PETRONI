import time
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


# APPLICAZIONE K_MEANS
def k_means_alg(data, X, k_clusters):
    start = time.time()
    k_means = KMeans(n_clusters=k_clusters, verbose=False)
    k_means.fit(X)
    song_cluster_labels = k_means.predict(X)
    end = time.time()

    data['cluster_label'] = song_cluster_labels

    print("Il tempo di esecuzione dell'algoritmo K-Means è: ", round(end - start, 3), "s")
    return data

# APPLICAZIONE K_MEDOIDS
def k_medoids_alg(data, X, k_clusters):
    X = X.to_numpy()
    start = time.time()
    k_means = KMedoids(n_clusters=k_clusters)
    k_means.fit(X)
    song_cluster_labels = k_means.predict(X)
    end = time.time()

    data['cluster_label'] = song_cluster_labels

    print("Il tempo di esecuzione dell'algoritmo KMedoids è: ", round(end - start, 3), "s")
    return data

# APPLICAZIONE DBSCAN
def dbscan_alg(data, X):
    ''' I valori di eps e min_samples sono stati impostati rispettivamente
    mediante il k-distance graph e la regola empirica (minPts >= num_features+1)'''
    start = time.time()
    clusters = DBSCAN(eps=1.9, min_samples=25)
    song_cluster_labels = clusters.fit_predict(X)
    end = time.time()

    data['cluster_label'] = song_cluster_labels

    # Contiamo il numero di noise points
    noisePoints = data[data.iloc[:, -1] == -1].value_counts()
    print("Il numero di noise point è: ", len(noisePoints))

    # Rimozione noise points
    data = data[data.iloc[:, -1] != -1]

    print("Il tempo di esecuzione dell'algoritmo DBSCAN è: ", round(end - start, 3), "s")
    return data


# APPLICAZIONE GAUSSIAN MIXTURE MODEL
def gmm_alg(data, X):

    start = time.time()
    gmm = GaussianMixture(n_components=4)
    gmm.fit(X)
    song_cluster_labels = gmm.predict(X)
    end = time.time()

    data['cluster_label'] = song_cluster_labels

    print("Il tempo di esecuzione dell'algoritmo GMM è: ", round(end - start, 3), "s")

    return data