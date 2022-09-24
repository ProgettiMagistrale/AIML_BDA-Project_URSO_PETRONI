import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Song_Recommender import Song_Recommender
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings
from clustering_algorithms import *
warnings.filterwarnings("ignore")
from elbow_method import elbow
from eps import choose_eps

#Lettura del dataset
data = pd.read_csv("data/data.csv")

## Rimozione parentesi e apostrofo
data["artists"] = data["artists"].str.replace("[", "")
data["artists"] = data["artists"].str.replace("]", "")
data["artists"] = data["artists"].str.replace("'", "")

#Verifica del numero di missing values
print("Il numero dei missing values nel dataset è: ",data.isna().sum().sum())

#Selezione delle features numeriche
X = data.select_dtypes(np.number)

#Normalizzazione mediante z-score
st = StandardScaler()
X = pd.DataFrame(st.fit_transform(X), columns=X.columns)

'''Applicazione dell'elbow method per il calcolo del valore ottimale di K (n° di clusters)
OSS: dall'applicazione dell'elbow method il K ottimale è pari a 4'''
#elbow(X)

'''Plot del k-distance graph per la scelta del valore ottimale di eps (dim. del raggio)
OSS: dal risultato del plot, il valore ottimale per eps risulta essere 2.5'''
#choose_eps(X)


#Istruzioni per MENU'
print("Specificare l'algoritmo da applicare:")
print("1) K-MEANS")
print("2) K-MEDOIDS")
print("3) DBSCAN")
scelta = input('Inserisci: ')

def switch(data, scelta):
    #settato numero di cluster pari a 4 dopo aver applicato l'Elbow method
    k_clusters = 4
    if scelta == "1":
        data = k_means_alg(data, X,k_clusters)
        return data

    elif scelta == "2":
        data = k_medoids_alg(data, X, k_clusters)
        return data

    elif scelta == "3":
        data = dbscan_alg(data, X)
        return data

data = switch(data, scelta)


#Richiamo della clase SongReccomender fornendo i parametri richiesti
input_song= "something just like this"
recommender = Song_Recommender(data)
result = recommender.get_recommendations(input_song, 10)
print("La canzone in riprodzione (input) è: ", input_song)
print(result[["name","artists"]])

''' Visualizzazione dei cluster ottenuti dall'applicazione degli algoritmi di clustering, 
utilizzando l'algoritmo di dimensionality reduction PCA'''

#Applicazione del PCA
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

#Visualizzazione bidimensionale dei clusters
p = sns.scatterplot(data=projection, x="x", y="y", hue=data['cluster_label'], legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.15, 1), title='Clusters')
plt.show()