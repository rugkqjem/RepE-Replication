import pandas as pd
import numpy as np
import json
import random

class FunctionDatasetLoader:
    def __init__(self,tokenizer,user_tag="USER:",assistant_tag="ASSISTANT:",seed=0):
        self.tokenizer=tokenizer
        self.seed=seed
        self.user_tag=user_tag
        self.assistant_tag=assistant_tag
        
        random.seed(seed)
        np.random.seed(seed)
    
    def load_data(self,file_path):
        #data를 pf으로 뽑기 
        if file_path.endswith('.jsonl'):
            df=pd.read_json(file_path,lines=True)
        elif file_path.endswith('.csv'):
            df=pd.read_csv(file_path)
        else:
            raise ValueError("지원하지 않는 파일형식")
        
        #참인 데이터만 뽑기 
        df=df[df['label']==1]
        honest_statements=[]
        untruthful_statements=[]
        
        if 'Question' in df.columns or'question' in df.columns:
            data_type='qa'
            template="You answer the question {type} question is"
            questions=df['question'].values.tolist()
            statements=df['answer'].values.tolist()

        elif 'Statement'in df.columns or'statement' in df.columns:
            data_type='statement'
            template="Pretend you're {type} person making statements about the world."
            statements=df['statement'].values.tolist()
            questions=[None]*len(statements)


        else: 
            raise ValueError("해당 데이터 형식으로는 데이터 가공 불가")
        
        for statement,question in zip(statements,questions):
            tokens=self.tokenizer.tokenize(statement)
            #아예 정답까지 모두 주면 모델이 너무 정답에만 따르기 때문에 정답이 있을법한 끝부분은 은근히 자름
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
                    untruthful_statements.append(f"{self.user_tag} {template.format(type='an untruthful')} {self.assistant_tag} "+truncated_statement)
                else :
                    honest_statements.append(f"{self.user_tag} {template.format(type='honestly')} {question} {self.assistant_tag} "+truncated_statement)
                    untruthful_statements.append(f"{self.user_tag} {template.format(type='untruthfully')} {question} {self.assistant_tag} "+truncated_statement)

        combined_data=[[honest,untruthful] for honest,untruthful in zip(honest_statements,untruthful_statements)]
        random.shuffle(combined_data)
        combined_data_size=len(combined_data)
        combined_train_size=int(combined_data_size*0.8)
        combined_test_size=combined_data_size-combined_train_size
        train_data=np.concatenate(combined_data[:combined_train_size]).tolist()
        test_data=np.concatenate(combined_data[combined_train_size:]).tolist()
        train_labels=[1,0]*combined_train_size
        test_labels=[1,0]*combined_test_size

        return {
            'train':{'data':train_data,'labels':train_labels},
            'test' :{'data':test_data,'labels':test_labels}
        }


class ConceptDatasetLoader:
    def __init__(self,tokenizer,seed=0):
        self.tokenizer=tokenizer
        self.seed=seed
        random.seed(seed)
        np.random.seed(seed)
        self.concept_template="Consider the amount of truthfulness in the following: {} The amount of concept is"
    
    def load_data(self,file_path):

        if file_path.endswith('.jsonl'):
            df=pd.read_json(file_path,lines=True)
        elif file_path.endswith('.csv'):
            df=pd.read_csv(file_path)
        else:
            raise ValueError("지원하지 않는 파일 형식")
        
    
        if 'question' in df.columns:
            df["statement"]=("The answer to the question("+df["question"].astype(str)+") is "+ df["answer"].astype(str))

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
        combined_train_size=int(combined_data_size*0.8)
        combined_test_size=combined_data_size-combined_train_size
        train_data=np.concatenate(combined_data[:combined_train_size]).tolist()
        test_data=np.concatenate(combined_data[combined_train_size:]).tolist()
        train_labels=[1,0]*combined_train_size
        test_labels=[1,0]*combined_test_size


        return {
            'train':{'data':train_data,'labels':train_labels},
            'test' :{'data':test_data,'labels':test_labels}
        }