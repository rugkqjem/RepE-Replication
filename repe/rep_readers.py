from abc import ABC, abstractmethod
import torch
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict,List,Optional,Union

#concept readers ---------------------------------------

class RepReader(ABC):
    def __init__(self,n_components:int=1):
        self.direction_method=None 
        self.directions : Dict[int,torch.Tensor]={}          
        self.direction_signs :Dict[int,float]={}
        self.n_components=n_components      
    
    @abstractmethod
    def get_rep_directions(self,*args,**kwargs):
        '''
        Args:
            hidden_states (dict) : {layer : torch.Tensor(n_samples,hidden_dim)}
            hidden_layers : Layers to consider 

        Returns : 
            directions : 레이어별 mapping된 concept에 대한 direction
        '''
        pass
    
    def get_signs(self,hidden_states,hidden_layers,labels):
        '''
        [기능] 추출된 Direction의 방향 (+/-)을 라벨 정보와 일치하도록 보정.

        Args: 
            hidden_states(Dict[int,torch.Tensor]) :소량의 train sample data들의 hidden_states 
            hidden_layers(List[int]) : Layers to consider 
            labels(List/torch.Tensor) :(예) 0= Dishonest, 1 = Honest(Target concept))
        
        Returns : 
            direction_signs (Dict[layer(int):(float -1.0 or 1.0)])
        '''
        if not isinstance(labels,torch.Tensor):
            labels=torch.tensor(labels)
        
        labels=labels.cpu()

        for layer in hidden_layers:
            if layer not in self.directions:
                continue
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
        [기능] 추출된 direction과 hidden_state를 내적하여 점수 계산

        Args:
            hidden_states : Dict[int,torch.Tensor]
            hidden_layers : List[int]

        Returns:
            score_map : Dict[int,torch.Tensor]
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

class ConceptRepReader(RepReader):
    '''
    - encoder : {topic} token 의 Activations
    - decoder : last token 의 Activaitons
    sample data random 하게 Pair 만들어 difference 추출하여 PCA 추출 
    '''

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
    

class FunctionRepReader(RepReader):
    '''
    prompt(+/-),Answer text -> Causal Masking 된 All Token Activations 추출하여 
    pos_activation(set) <-> neg_activation(set) 랜덤하게 서로 difference vector -> PCA 추출 

    '''
    def get_rep_directions(self, pos_hidden_states,neg_hidden_states,hidden_layers):
        '''
        Args:
            pos_hidden_states: {layer : (batch,Sequence_length,hidden_dim)} - 진실 모드 데이터
            neg_hidden_states: {layer : (batch,Sequence_length,hidden_dim)} - 거짓 모드 데이터
        '''
        for layer in hidden_layers:
            if layer not in pos_hidden_states or layer not in neg_hidden_states:
                continue
            H_pos=pos_hidden_states[layer].reshape(-1,pos_hidden_states[layer].shape[-1])
            H_neg=pos_hidden_states[layer].reshape(-1,pos_hidden_states[layer].shape[-1])

            min_len=min(H_pos.shape[0],H_neg.shape[0])
            pos_indices=torch.randperm(H_pos.shape[0])[:min_len]
            neg_indices=torch.randperm(H_neg.shape[0])[:min_len]

            H_pos_selected=H_pos[pos_indices]
            H_neg_selected=H_neg[neg_indices]

            train_diff=(H_pos_selected-H_neg_selected).detacth().cpu().numpy()
            pca=PCA(n_components=self.n_components)
            pca.fit(train_diff)

            self.directions[layer]=torch.tensor(pca.components_,dtype=torch.float32)
            self.direction_signs[layer]=1.0

        return self.directions
    
#class ClusterMeanRepReader(RepReader):
#class RandomRepReader(RepReader):

