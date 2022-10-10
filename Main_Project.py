import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from matplotlib.pyplot import figure
from Song_Recommender import Song_Recommender
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings
from clustering_algorithms import *
warnings.filterwarnings("ignore")
from elbow_method import elbow
from eps import choose_eps
from sklearn.manifold import TSNE



#Lettura del dataset
data = pd.read_csv("data/playlist_dataframe.csv")
# Rimozione dei samlpes duplicati
#data.drop_duplicates(keep=False, inplace=True)

## Rimozione parentesi e apostrofo
data["artists"] = data["artists"].str.replace("[", "")
data["artists"] = data["artists"].str.replace("]", "")
data["artists"] = data["artists"].str.replace("'", "")

#Verifica del numero di missing values
print("Il numero dei missing values nel dataset è: ",data.isna().sum().sum())

playlists = data["playlist"]
data.drop(['playlist'], axis = 1, inplace = True)

#Selezione delle features numeriche
X = data.select_dtypes(np.number)

#Feauture selection, mediante l'information gain
'''figure(figsize=(11, 6), dpi=80)
importances = mutual_info_classif(X, playlists)
feat_importances=pd.Series(importances, X.columns[0:len(X.columns)])
feat_importances.plot(kind='barh', color='teal')
plt.show()'''

'''Poichè dal risultato precedente, la feature con meno importanza (IG più basso) sono key e popularity, 
non la consideriamo per la costruzione del modello'''
data.drop(['key'], axis = 1, inplace = True)
data.drop(['popularity'], axis = 1, inplace = True)

#Seleziono le features numeriche escludendo la feature 'key'
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
print("4) GMM")
scelta = input('Inserisci: ')

def switch(data, scelta):
    #settato numero di cluster pari a 7 dopo aver applicato l'Elbow method
    if scelta == "1":
        k_clusters = 7
        data = k_means_alg(data, X,k_clusters)
        return data

    elif scelta == "2":
        k_clusters = 5
        '''A causa dell'elevata capacità computazionale richiesta, in termini di memoria neccessaria
        per l'implementazione del k-medoids è stato ridotto il numero di samples utilizzati da 47.000 a 40.000'''
        data = k_medoids_alg(data.head(40000), X.head(40000), k_clusters)
        return data

    elif scelta == "3":
        data = dbscan_alg(data, X)
        return data

    elif scelta == "4":
        data = gmm_alg(data, X)
        return data

data = switch(data, scelta)

'''Riduzione dell'array playlist nel caso in cui il metodo scelta sia il k-medoids'''
if scelta == "2":
    playlists = playlists.head(40000)


#Richiamo della clase SongReccomender fornendo i parametri richiesti
input_song= "Call me maybe"
recommender = Song_Recommender(data)

#Numero canzoni playlist consigliata
n_song = 25

#Calcolo della playlist mediante il metodo get_recommendations() della classe SongRecommender()
result = recommender.get_recommendations(input_song, n_song)

#Estrazione dell'indice per la selezione della playlist ideale della canzone in input
index_song_input = recommender.get_index_input().values[0]
playlist_song_input = playlists[index_song_input]

#Estraggo dal dataset la playlist della input_song
playlist_songs = []
i = 0
for elem in data['name']:
    if playlists[i] == playlist_song_input and elem !=input_song:
        playlist_songs.append(elem)
    i = i+1
print("La canzone in riproduzione (input) è: ", input_song)
print(result[["name","artists"]])

#Calcolo del miss classification error necessaio per ottenere Error rate e accuracy
missClassErr=0
for elem in result['name']:
    if elem not in playlist_songs:
        missClassErr = missClassErr+1

print("Il numero di miss classified sample è: ", missClassErr)

print("L'error rate sulle prime ",n_song, "canzoni è pari al ", (missClassErr/n_song)*100, "%")
print("L'accuracy sulle prime ",n_song, "canzoni è pari al ", (1-(missClassErr/n_song))*100, "%")

''' Visualizzazione dei cluster ottenuti dall'applicazione degli algoritmi di clustering, 
utilizzando l'algoritmo di dimensionality reduction PCA'''

#Applicazione del PCA
figure(figsize=(9, 6), dpi=80)
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

p = sns.scatterplot(data=projection, x="x", y="y", hue=data['cluster_label'], legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.13, 1), title='Clusters')
plt.show()



# Visualizing the Clusters with t-SNE
figure(figsize=(9, 6), dpi=80)
tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
song_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

#Visualizzazione bidimensionale dei clusters
p = sns.scatterplot(data=projection, x="x", y="y", hue=data['cluster_label'], legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.13, 1), title='Clusters')
plt.show()