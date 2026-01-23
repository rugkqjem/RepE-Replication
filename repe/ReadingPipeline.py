import torch
import numpy as np
from sklearn.decomposition import PCA 
from tqdm import tqdm
from Reader import RepReader

class RepPipeline:
    def __init__(self,model,tokenizer):
        self.model=model
        self.tokenizer=tokenizer
        self.reader=None 
    
    #배치로 train dataset에서 (hidden_layers들의 rep_token의 hidden_state 추출) 
    def _get_hidden_states(self,text_inputs,batch_size=8,hidden_layers=[],rep_token=-1):
        '''
        데이터(List[str])를 입력 받아 hidden_layers 들의 rep_token의 hidden_state를 dict{layer:[hidden_states]}추출하는 함수
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
    
    def get_direction(self,train_inputs,train_labels,hidden_layers,rep_token=-1,batch_size=8,mode=""):
        """
        mode: reader 의 extract_direction 함수에 들어가는 인자로, ("" 또는 "comparing")
        """
        raw_hidden_states,diff_hidden_states=self._get_hidden_states(
            text_inputs=train_inputs,batch_size=batch_size,hidden_layers=hidden_layers,rep_token=rep_token
        )
        #function의 경우 text_inputs리스트 입력할때부터 긍정/부정 번갈아가며 데이터셋이 구성되어야함
        self.reader=RepReader()
        directions=self.reader.extract_directions(raw_hidden_states=raw_hidden_states,diff_hidden_states=diff_hidden_states,train_labels=train_labels,mode=mode)
        return directions
    
    def predict(self,test_inputs,test_labels,hidden_layers,rep_token=-1,batch_size=8,mode="binary",group_sizes=None,reader=None):
        """
        mode: "binary" 또는 "multi_choice" 또는 "comparing"
        group_size: "multi_choice" 모드일 때, 각 문제당 보기 개수
        """
        if reader is not None:
            self.reader=reader

        if self.reader is None:
            raise ValueError("get_directions(학습)되지 않았음.")
    
        raw_hidden_states,_=self._get_hidden_states(text_inputs=test_inputs,batch_size=batch_size,hidden_layers=hidden_layers,rep_token=rep_token)
        scores=self.reader.transform(raw_hidden_states,hidden_layers)
        test_labels = torch.tensor(test_labels).to(self.model.device)
        results={}

        for layer in hidden_layers:
            if mode=="binary":
                score=torch.tensor(scores[layer]).to(self.model.device)
                preds=(score>0).long()
                accuracy = (preds == test_labels).float().mean()
                results[layer]=accuracy.item()

            elif mode=="comparing":
                if len(test_inputs)%2!=0:
                    raise ValueError("Comparing 모드에서는 test data 셋이 짝수이어야한다.")
                correct_count=0
                score=scores[layer]
                for i in range(0,len(test_inputs),2):
                    if score[i]>score[i+1]:
                        correct_count+=1
                acc=correct_count/(len(test_inputs)//2)
                results[layer]=acc

            elif mode=="multi_choice":
                if group_sizes is None:
                    raise ValueError("multi_choice 모드에서 group_sizes 필요")
                correct_count=0
                total_questions=len(group_sizes)
                current_idx=0
                score=np.array(scores[layer])
                for size in group_sizes:
                    chunk_scores = score[current_idx: current_idx + size]
                    chunk_labels=test_labels[current_idx:current_idx+size].cpu().numpy()
                    predicted_idx=np.argmax(chunk_scores)
                    true_idx=np.argmax(chunk_labels)
                    if predicted_idx==true_idx:
                        correct_count+=1
                    current_idx+=size
                acc=correct_count/total_questions
                results[layer]=acc

        return results


        