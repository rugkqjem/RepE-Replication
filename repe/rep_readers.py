from abc import ABC, abstractmethod
import torch
import numpy as np
from sklearn.decomposition import PCA

#concept readers ---------------------------------------

class RepReader(ABC):
    def __init__(self):
        self.direction_method=None 
        self.directions={}          
        self.direction_signs={}      
    
    @abstractmethod
    def get_rep_directions(self,hidden_states,hidden_layers,**kwargs):
        '''
        Args:
            hidden_states (dict) : {layer : torch.Tensor(n_samples,hidden_dim)}
            hidden_layers : Layers to consider 

        Returns : 
            directions : 레이어별 mapping된 concept에 대한 direction
        '''
        pass
    
    def get_signs(self,hidden_states,hidden_layers,lables):
        '''
        Args: 
            hidden_states(dict) :소량의 train sample data들의 hidden_states 
            hidden_layers(List) : Layers to consider 
            labels(List/torch.Tensor) :(예) 0= Dishonest, 1 = Honest(Target concept))
        
        Returns : 
            direction_signs (Dict[layer(int):(int -1 or 1)])
        '''
        pass 
    
    def transform(self,hidden_states,hidden_layers):
        '''
        입력된 hidden_states를 학습된 방향에 투영하여 점수 반환
        Returns : 
            scores: (n_samples,) 각 sample 의 개념 활성 점수
        '''
        pass

class PCARepReader(RepReader):
    '''
    PCA를 사용하여 Representation Direction을 추출하는 Reader class.
    '''
    def __init__(self,n_components=1):
        super().__init__()
        #주성분 추출 축 1개로 설정
        self.n_components=n_components

    def get_rep_directions(self, hidden_states, hidden_layers, **kwargs):
        for layer in hidden_layers:
            if layer not in hidden_states:
                print(f"layer_{layer} not in hidden_states")
            H = hidden_states[layer]
            n_samples=H.shape[0]
            perm=torch.randperm(n_samples)
            H_shuffled=H[perm]
            
            if n_samples%2!=0:
                H_shuffled=H_shuffled[:-1]
                n_samples-=1
            
            half=n_samples//2
            H_1=H_shuffled[:half]
            H_2=H_shuffled[half:]
            train_diff=(H_1-H_2).detach().cpu().numpy()

            pca=PCA(n_components=self.n_components)
            pca.fit(train_diff)
            self.directions[layer]=torch.tensor(pca.components_,dtype=torch.float32) #Tensor[1,hidden_dim]
            
            self.direction_signs[layer]=1.0  #초기 sign은 1.0으로 임의 설정

        return self.directions
    
    def get_signs(self,hidden_states,hidden_layers,labels):
        ''''
        추출된 Direction의 방향 (+/-)을 라벨 정보와 일치하도록 보정.
        '''
        if not isinstance(labels,torch.Tensor):
            labels=torch.tensor(labels)
        
        labels=labels.cpu()

        for layer in hidden_layers:
            H=hidden_states[layer].float().cpu()
            direction=self.directions[layer][0]
            scores=torch.matmul(H,direction) #shape(N,)
            mean_score_class_0=torch.mean(scores[labels==0])
            mean_score_class_1=torch.mean(scores[labels==1])

            if mean_score_class_1>mean_score_class_0:
                self.direction_signs[layer]=1.0
            else:
                self.direction_signs[layer]=-1.0
        
        return self.direction_signs

    def transform(self, hidden_states, hidden_layers):
        score_map={}
        '''
        추출된 direction과 hidden_state를 내적하여 점수 계산
        '''
        for layer in hidden_layers:
            if layer not in hidden_states or layer not in self.directions:
                continue
            H=hidden_states[layer]
            direction=self.directions[layer][0]
            sign=self.direction_signs[layer]
            scores=torch.matmul(H,direction.to(H.device))
            scores*=sign
            score_map[layer]=scores

        return score_map


#class ClusterMeanRepReader(RepReader):
#class RandomRepReader(RepReader):

