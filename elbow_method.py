import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture

def elbow(df):
    distortions = []
    K = range(1,15)
    for k in K:
        kmeanModel = GaussianMixture(n_components=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inerti)

    #plot
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
