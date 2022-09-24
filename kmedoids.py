import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

#Algoritmo K-Medoids
class KMedoids(object):
    def __init__(self, n_cluster=2, dist=euclidean_distances, random_state=42):
        self.n_cluster = n_cluster
        self.dist = dist
        self.rstate = np.random.RandomState(random_state)
        self.cluster_centers = []
        self.indices = []

    def fit(self, X):
        rint = self.rstate.randint
        self.indices = [rint(X.shape[0])]
        for _ in range(self.n_cluster -1):
            i = rint(X.shape[0])
            while i in self.indices:
                i = rint(X.shape[0])
            self.indices.append(i)
        self.cluster_centers = X[self.indices, :]

        self.y_pred = np.argmin(self.dist(X, self.cluster_centers), axis=1)
        cost, _ = self.cost_function(X, self.indices)
        new_cost = cost
        new_y_preds = self.y_pred.copy()
        new_indices = self.indices[:]
        init = True
        while (new_cost < cost) | init:
            init = False
            cost = new_cost
            self.y_pred = new_y_preds
            self.indices = new_indices
            for k in range(self.n_cluster):
                for r in [i for i, x in enumerate(new_y_preds == k) if x]:
                    if r not in self.indices:
                        indices_temp = self.indices[:]
                        indices_temp[k] = r
                        new_cost_temp, y_preds_temp = self.cost_function(X, indices_temp)
                        if new_cost_temp < new_cost:
                            new_cost = new_cost_temp
                            new_y_preds = y_preds_temp
                            new_indices = indices_temp


    def cost_function(self, data, indices):
        y_preds = np.argmin(self.dist(data, data[indices, :]), axis=1)
        cost = np.sum([np.sum(self.dist(data[y_preds == i], data[[indices[i]], :]))
                       for i in set(y_preds)])
        return cost, y_preds

    def prediction(self, X):
        return np.argmin(self.dist(X, self.cluster_centers), axis=1)