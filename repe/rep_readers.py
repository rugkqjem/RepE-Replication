from abc import ABC, abstractmethod
import torch


#utils-------------------------------------------------
# def project_onto_direction(H,direction)
# def recenter



#concept readers ---------------------------------------
'''
RepReader return transformed_hidden_states
PCARepReader return signs -> 방향 확인필요
ClusterMeanRepReader return directions 
RandomRepReader(RepReader) return
'''

#이전의 pipeline에서 layer별 마지막 hidden_states 뽑아서 전달
class RepReader(ABC):
    def __init__(self):
        self.direction_method=None 
        self.directions={} #shape(layer,hidden_size)
        self.direction_sign={} 
    
    @abstractmethod
    def get_rep_directions(self,hidden_states,hidden_layers,**kwargs):
        '''
        Args:
            hidden_states(num_layers,hidden_dim) : Hidden states(token) of the model on the training data (per layer)
            hidden_layers : Layers to consider 

        Returns : 
            directions : 레이어별 mapping된 concept에 대한 direction
        '''
        pass
    
    def get_signs(self,hidden_states,hidden_layers,lables):
        '''
        direction과 labeled samples를 가지고 direction의 방향(+/-) 결정
        Args: 
            hidden_states : 소량의 train sample data들의 hidden_states
            hidden_states : Layers to consider 
        '''
        pass 
    
    def transform(self,hidden_states,layer_idx):
        '''
        추출한 direction과 label이 있는 (같은 layer) hidden_state 투영하여 Scoring
        '''
        pass

class PCARepReader(RepReader):
    def __init__(self,n_compoents=1):
        super().__init__()
        #주성분 추출 축 1개로 설정
        self.n_components=n_compoents

#class ClusterMeanRepReader(RepReader):
#class RandomRepReader(RepReader):




#function readers----------------------------------------