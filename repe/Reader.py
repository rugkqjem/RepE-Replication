import torch
import numpy as np
from sklearn.decomposition import PCA

class RepReader:
    def __init__(self,n_components=1):
        self.directions={}
        self.n_components=n_components
        self.scores_means={}
        self.scores_std={}

    def extract_directions(self,diff_hidden_states,raw_hidden_states,train_labels=None,mode="",component_index=0):
        #1. 방향 추출
        #2. 부호 보정 
        for layer,diff_data in diff_hidden_states.items():
            pca=PCA(n_components=self.n_components)
            pca.fit(diff_data)
            direction=pca.components_[component_index]
            H=raw_hidden_states[layer] #(samples,hidden)
            scores=H.dot(direction)
            mean=scores.mean()
            std=scores.std()

            if mode=="":
                predictions=(scores>mean).astype(int)
                if train_labels==None:
                    train_labels=np.array([1,0]*(len(scores)//2))
                accuracy=np.mean(predictions==train_labels)

                if accuracy<0.5:
                    direction=-1*direction
                    scores=-1*scores
                    mean=scores.mean()

            elif mode=="comparing":
                p_sign=0
                n_sign=0
                for i in range(0,len(scores),2):
                    if scores[i]>scores[i+1]:
                        p_sign+=1
                    else:
                        n_sign+=1
                if p_sign<n_sign:
                    direction=-1*direction
                    scores=-1*scores
                    mean=scores.mean()
            else:
                raise ValueError("존재하지 않는 RepReader의 extract_directions mode 입니다.")
            self.directions[layer]=direction
            self.scores_means[layer]=mean
            self.scores_std[layer] = std + 1e-8

        return self.directions


    # 표준화 적용 Scoring (Z-score Normalization)
    def transform(self,raw_hidden_states,hidden_layers):
        scores={}
        for layer in hidden_layers:
            if layer not in self.directions.keys():
                print(f"해당 layer{layer}의 direction이 없음.")
                continue
            direction=self.directions[layer]
            H=raw_hidden_states[layer]
            scores[layer] = ( H @ direction - self.scores_means[layer] ) / self.scores_std[layer]
            #scores[layer] = H @ direction

        return scores
