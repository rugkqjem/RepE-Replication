import torch
import numpy as np
from sklearn.decomposition import PCA 
from tqdm import tqdm
from Reader import RepReader

class ReadingPipelien:
    def __init__(self,model,tokenizer):
        self.model=model
        self.tokenizer=tokenizer
    
    #배치로 train dataset에서 (hidden_layers들의 rep_token의 hidden_state 추출) 
    def get_hidden_states(self,text_inputs,batch_size=8,hidden_layers=[],rep_token=-1):
        '''
        데이터(List[str])를 입력 받아 hidden_layers 들의 rep_token의 hidden_state를 추출하는 함수
        '''
        self.model.eval()
        buffer={layer: [] for layer in hidden_layers}
        
        diff_hidden_states={}
        raw_hidden_states={}

        for i in tqdm(range(0,len(text_inputs),batch_size),desc="Extracting Hiddens"):
            batch_inputs=text_inputs[i:i+batch_size]

            encoded=self.tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            with torch.no_grad():
                outputs=self.model(**encoded,output_hidden_states=True)

            for layer in hidden_layers:
                vec=outputs.hidden_states[layer][:,rep_token,:].detach().cpu().numpy()
                buffer[layer].append(vec)

        for layer in hidden_layers:
            raw_hidden_states[layer]=np.vstack(buffer[layer])
            diff_hidden_states[layer]=raw_hidden_states[layer][::2]-raw_hidden_states[layer][1::2]

        return raw_hidden_states,diff_hidden_states
    
    def get_direction(self,raw_hidden_states,diff_hidden_states,hidden_layers,labels):
        difference_hidden_states={}

        #function의 경우 text_inputs리스트 입력할때부터 긍정/부정 번갈아가며 데이터셋이 구성되어야함
        reader=RepReader()
        

        