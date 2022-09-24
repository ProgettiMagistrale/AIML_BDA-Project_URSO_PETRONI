import numpy as np
from tqdm import tqdm

class Song_Recommender():
    def __init__(self, data):
        self.data_ = data

    def get_recommendations(self, song_name, n_top):
        distances = []

        #Estraiamo i valori corrispondenti alla Canzone in input (quella in riproduzione)
        song = self.data_[(self.data_.name.str.lower() == song_name.lower())].head(1).values[0]
        #Prelevo il cluster di appartenenza della canzone in input
        cluster_input = song[-1]

        #Estrazione valori di tutte le altre canzoni
        rem_data = self.data_[self.data_.name.str.lower() != song_name.lower()]

        # Estrazione delle canzoni appartenenti allo stesso cluster della canzone in input (in riproduzione)
        rem_data = rem_data[rem_data.iloc[:,-1] == cluster_input]

        #Per ogni canzone nel dataset (eccetto quella in input)
        for r_song in tqdm(rem_data.values):
            dist = 0
            for col in np.arange(len(rem_data.columns)):
                #Non consideriamo le colonne numerate in basso (le colonne categoriche)
                if not col in [3,8,14,16]:
                    #Valore di correlazione tra la canzone in riproduzione (input) e ciascuna delle altre canzoni presenti nel dataset
                    dist = dist + np.absolute(float(song[col]) - float(r_song[col]))
            distances.append(dist)

        #Aggiungiamo le distanze ad ogni canzone nel dataset
        rem_data['distance'] = distances

        #Ordiniamo il dataset in base alle distanze calcolate
        rem_data = rem_data.sort_values('distance')

        #Estrazione colonne di interesse
        columns = ['name', 'artists','id']

        return rem_data[columns][:n_top]