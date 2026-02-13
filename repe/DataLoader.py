import pandas as pd
import numpy as np
import json
import random
from torch.utils.data import Dataset
from datasets import load_dataset

class FunctionDatasetLoader:
    def __init__(self,tokenizer,user_tag="USER:",assistant_tag="ASSISTANT:",seed=0):
        self.tokenizer=tokenizer
        self.seed=seed
        self.user_tag=user_tag
        self.assistant_tag=assistant_tag

        
        random.seed(seed)
        np.random.seed(seed)
    
    def load_data(self,file_path,n_train=512):
        #data를 pf으로 뽑기 
        if file_path.endswith('.jsonl'):
            data_from="file"
            df=pd.read_json(file_path,lines=True)
        elif file_path.endswith('.csv'):
            data_from="file"
            df=pd.read_csv(file_path)
        elif file_path=="alpaca":
            data_from="dataset"

        else:
            raise ValueError("지원하지 않는 파일형식")

        honest_statements=[]
        untruthful_statements=[]
        
        if (data_from!="dataset") and ('Question' in df.columns or'question' in df.columns):
            data_type='qa'
            df=df[df['label']==1]
            template="You answer the question {type}, question is"
            questions=df['question'].values.tolist()
            statements=df['answer'].values.tolist()

        elif (data_from!="dataset") and ('Statement'in df.columns or'statement' in df.columns):
            data_type='statement'
            df=df[df['label']==1]
            template="Pretend you're {type} person making statements about the world."
            statements=df['statement'].values.tolist()
            questions=[None]*len(statements)

        elif data_from=="dataset" : 
            data_type="dataset"
            ds=load_dataset('tatsu-lab/alpaca')
            ds=ds.filter(lambda x:x['input']=='')
            questions=ds["train"]["instruction"] #사실은 instruction (아래 형식 맞추기 위해 questions이라는 변수로 설정)
            statements=ds['train']['output']        #사실은 outputs ( 아래 형식 맞추기 위해 statements라는 변수로 설정)
            template="{user_tag} {instruction} {type} {assistant_tag} {response}"

        else: 
            raise ValueError("해당 데이터 형식으로는 데이터 가공 불가")
        
        for statement,question in zip(statements,questions):
            tokens=self.tokenizer.tokenize(statement)
            #아예 정답까지 모두 주면 모델이 너무 정답에만 따르기 때문에 정답이 있을법한 끝부분은 조금 자름
            if len(tokens)<=5:
                ranges=[len(tokens)]
            else:
                ranges=range(1,len(tokens)-5)

            for idx in ranges:
                if idx==len(tokens):
                    truncated_statement=statement
                else:
                    truncated_tokens=tokens[:idx]
                    truncated_statement=self.tokenizer.convert_tokens_to_string(truncated_tokens)

                if data_type=='statement':
                    honest_statements.append(f"{self.user_tag} {template.format(type='an honest')} {self.assistant_tag} "+truncated_statement)
                    untruthful_statements.append(f"{self.user_tag} {template.format(type='a dishonest')} {self.assistant_tag} "+truncated_statement)
                elif data_type=='qa' :
                    honest_statements.append(f"{self.user_tag} {template.format(type='honestly')} {question} {self.assistant_tag} "+truncated_statement)
                    untruthful_statements.append(f"{self.user_tag} {template.format(type='untruthfully')} {question} {self.assistant_tag} "+truncated_statement)
                elif data_type=="dataset":
                    honest_statements.append(template.format(user_tag=self.user_tag,instruction=question,type="a truthful",assistant_tag=self.assistant_tag,response=truncated_statement))
                    untruthful_statements.append(template.format(user_tag=self.user_tag,instruction=question,type="an untruthful",assistant_tag=self.assistant_tag,response=truncated_statement))
        
        combined_data=[[honest,untruthful] for honest,untruthful in zip(honest_statements,untruthful_statements)]
        #random.shuffle(combined_data)
        train_data=np.concatenate(combined_data[:n_train]).tolist()
        train_labels=[1,0]*n_train

        #test data reshaping ----------------------------------------------
        reshaped_data=[[honest,untruthful] for honest,untruthful in zip(honest_statements[1:],untruthful_statements[:-1])]
        test_data=np.concatenate(reshaped_data[-n_train:]).tolist()

        #test_data=np.concatenate(combined_data[n_train:n_train*2]).tolist()
        test_labels=[1,0]*(len(test_data)//2)

        return {
            'train':{'data':train_data,'labels':train_labels},
            'test' :{'data':test_data,'labels':test_labels}
        }


class ConceptDatasetLoader:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        #TruthfulQA Template---------------------------------------------------------------------------------------------------------------
        #self.concept_template="Consider the amount of truthfulness in the following answer: \n {} \n The amount of truthfulness in the answer is"
        #ARC--------------------------------------------------------------------------------------------------------------------------------
        self.concept_template="Consider the correctness of the answer to the following question:\n{}\nThe probability of the answer being correct is"
    
    def load_data(self,file_path,template="",n_train=25,seed=0):
        random.seed(seed)
        np.random.seed(seed)

        if template!="":
            self.concept_template=template
        if file_path.endswith('.jsonl'):
            df=pd.read_json(file_path,lines=True)
        elif file_path.endswith('.csv'):
            df=pd.read_csv(file_path)
        else:
            raise ValueError("지원하지 않는 파일 형식")
        
        if 'question' in df.columns:
            df["statement"]=(
                "Question: " + df["question"].astype(str).str.strip()+"\n"+ 
                "Answer: " + df["answer"].astype(str).str.strip()
            )

        elif 'statement' not in df.columns:
            raise ValueError("데이터에 statement columns 존재하지 않음")
        
        df['final_input']=df['statement'].apply(lambda x:self.concept_template.format(x))

        true_inputs=df[df["label"]==1]['final_input'].tolist()
        false_inputs=df[df["label"]==0]['final_input'].tolist()

        min_len=min(len(true_inputs),len(false_inputs))
        true_inputs=true_inputs[:min_len]
        false_inputs=false_inputs[:min_len]
        combined_data=[[t,f] for t,f in zip(true_inputs,false_inputs)]
        random.shuffle(combined_data)
        combined_data_size=len(combined_data)
        #---train sample 개수(n_train) 명시 하는 경우----------------------------------- 
        train_data=np.concatenate(combined_data[:n_train]).tolist()
        train_labels=[1,0]*n_train
        
        #train에 모든 dataset 사용하는 경우 (val을 위해서 train dataset 재활용할 수 밖에 없음)
        if (combined_data_size==n_train) or (combined_data_size<n_train*2):
            test_data=train_data
            test_labels=train_labels
        else:
            reshaped_data=[[t,f] for t,f in zip(true_inputs[:-1],false_inputs[1:])]
            test_data=np.concatenate(reshaped_data[n_train:n_train*2]).tolist()
            test_labels=[1,0]*n_train
        
        return{
           "train":{"data":train_data,"labels":train_labels},
           "test" :{"data":test_data,"labels":test_labels}
           }