import torch
import numpy as np
from sklearn.decomposition import PCA

def projection(H,direction):
    return

def recenter(x,mean=None):
    return 

class RepReader:
    def __init__(self,n_components=1):
        self.directions={}
        self.n_components=n_components

    def extract_directions(self,diff_hidden_states,raw_hidden_states,labels):
        #1. 방향 추출
        #2. 부호 보정 
        for layer,diff_data in diff_hidden_states.items():
            pca=PCA(n_components=self.n_components)
            pca.fit(diff_data)
            direction=pca.components_[0]
            H=raw_hidden_states[layer] #(samples,hidden)
            score=H.dot(direction)
            threshold=score.mean()
            predictions=(score>threshold)
            true_label_score=score*labels

        return self.directions

        

    def transform(self):
        '''
        train이 아니라 가지고 있는 direction을 가지고 scoring
        '''
        pass



