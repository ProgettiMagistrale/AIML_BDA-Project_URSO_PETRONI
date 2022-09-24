from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from kmedoids import KMedoids

# APPLICAZIONE K_MEANS
def k_means_alg(data, X, k_clusters):
    k_means = KMeans(n_clusters=k_clusters, verbose=False)
    k_means.fit(X)
    song_cluster_labels = k_means.predict(X)
    data['cluster_label'] = song_cluster_labels
    return data

# APPLICAZIONE K_MEDOIDS
def k_medoids_alg(data, X, k_clusters):
    X = X.to_numpy()
    k_means = KMedoids(n_cluster=k_clusters)
    k_means.fit(X)
    song_cluster_labels = k_means.prediction(X)
    data['cluster_label'] = song_cluster_labels
    return data

# APPLICAZIONE DBSCAN
def dbscan_alg(data, X):
    ''' I valori di eps e min_samples sono stati impostati rispettivamente
    mediante il k-distance graph e la regola empirica (minPts >= num_features+1)'''
    clusters = DBSCAN(eps=2.5, min_samples=18)
    song_cluster_labels = clusters.fit_predict(X)
    data['cluster_label'] = song_cluster_labels
    return data