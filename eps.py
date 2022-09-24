from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

#Costruzione del k-distance graph per il calcolo del valore ottimale di eps
def choose_eps(stdDf):
    nn = NearestNeighbors(n_neighbors=20).fit(stdDf)
    distances, indices = nn.kneighbors(stdDf)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.figure(figsize=(10,8))
    plt.plot(distances)

    plt.show()