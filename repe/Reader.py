import torch
import numpy as np
from sklearn.decomposition import PCA

class RepReader:
    def __init__(self,n_components=1):
        self.directions={}
        self.n_components=n_components
        self.direction_means={}

    def extract_directions(self,diff_hidden_states,raw_hidden_states,train_labels=None):
        #1. 방향 추출
        #2. 부호 보정 
        for layer,diff_data in diff_hidden_states.items():
            pca=PCA(n_components=self.n_components)
            pca.fit(diff_data)
            direction=pca.components_[0]
            H=raw_hidden_states[layer] #(samples,hidden)
            scores=H.dot(direction)
            threshold=scores.mean()
            predictions=(scores>threshold).astype(int)
            if train_labels==None:
                train_labels=np.array([1,0]*(len(scores)//2))
            accuracy=np.mean(predictions==train_labels)

            if accuracy<0.5:
                direction=-1*direction
                scores=-1*scores
                threshold=scores.mean()

            self.directions[layer]=direction
            self.direction_means[layer]=threshold

        return self.directions

    def transform(self,raw_hidden_states,hidden_layers):
        scores={}
        for layer in hidden_layers:
            if layer not in self.directions.keys():
                print(f"해당 layer{layer}의 direction이 없음.")
                continue
            direction=self.directions[layer]
            H=raw_hidden_states[layer]
            scores[layer] = H @ direction - self.direction_means[layer]

        return scores
