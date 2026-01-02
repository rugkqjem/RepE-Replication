import torch
from tqdm import tqdm
from typing import List,Dict,Union,Optional 
from functools import partial
from rep_readers import ConceptRepReader,FunctionRepReader

class RepReadingPipeline:
    def __init__(self,model,tokenizer, reader_type : str="concept",**kwargs):
        '''
        Args:
            reader_type: "concept" / "function"
        '''
        self.model=model
        self.num_layers=len(self.model.model.layers)
        self.tokenizer=tokenizer

        DIRECTION_FINDERS={
            "concept" : ConceptRepReader,
            "function" : FunctionRepReader
        }

        if reader_type not in DIRECTION_FINDERS:
            raise ValueError(f"Invalid reader_type:{reader_type}. Choose 'concept' or 'function'.")

        self.reader_type=reader_type
        self.reader=DIRECTION_FINDERS[reader_type](**kwargs)

        if self.tokenizer.padding_side !='left':
            self.tokenizer.paddig_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token 
    
    def _extract_hidden_states(self,data,batch_size=8,extract_mode="concept"):
        '''
        모델에서 hidden_states를 추출하는 함수.
        #layer list 필요없다. 아예 hidden 모든 레이어 {layer:{samples,hidden_dim}} 으로 reader한테 넘길거 .
        Args:
            data: 
                - concept mode : List[(prompt,stimulus)] (완성된 prompt list)
                - function mode : List[(prompt,stimulus(Question,Answer))]
            extract_mode : 'concept' (Last token) / 'function' (Answer tokens) 
        '''
        self.model.eval() 
        num_layers=len(self.model.model.layers)
        all_layer_indices=list(range(num_layers))

        collected_acts={layer:[] for layer in all_layer_indices}
        current_batch_acts={}

        def hook_fn(module,input,output,layer_idx):
            if isinstance(output,tuple):
                act=output[0]

