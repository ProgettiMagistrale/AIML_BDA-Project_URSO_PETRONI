import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

def elbow(df):
    distortions = []
    K = range(2,9)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        #kmedoidsModel = KMedoids(n_clusters=k)

        kmeanModel.fit(df)
        #kmedoidsModel.fit(df)

        distortions.append(kmeanModel.inertia_)
        #distortions.append(kmedoidsModel.inertia_)

    #plot
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
