import torch
import numpy as np
from datasets import load_dataset
from DataLoader import ConceptDatasetLoader
from ReadingPipeline import RepPipeline
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import pandas as pd

#template 입력 가능
def train_predict(model,tokenizer,hidden_layers,file_path,n_train,n_trials,template=""):
    dataset_loader=ConceptDatasetLoader(tokenizer=tokenizer)
    total_results={ layer:[] for layer in hidden_layers}
    best_reader=None
    best_reader_per_layer={layer:{"best_acc":-1,"best_reader":None} for layer in hidden_layers}
    #best val accuracy reader (direction)    
    for seed in range(n_trials):
        print(f"======{seed+1}번째 학습 진행 중======")
        concept_dataset=dataset_loader.load_data(file_path,template=template,n_train=n_train,seed=seed)
        train_inputs=concept_dataset["train"]["data"]
        train_labels=concept_dataset["train"]["labels"]
        val_inputs=concept_dataset["test"]["data"]
        val_labels=concept_dataset["test"]["labels"]
        reading_pipeline=RepPipeline(model=model,tokenizer=tokenizer)
        concept_directions=reading_pipeline.get_direction(
            train_inputs=train_inputs,
            train_labels=train_labels,
            hidden_layers=hidden_layers
        )
        print(f"======{seed+1}번째 평가 진행 중======")
        results=reading_pipeline.predict(
            test_inputs=val_inputs,
            test_labels=val_labels,
            hidden_layers=hidden_layers
        )
        current_reader=reading_pipeline.reader
        for layer in hidden_layers:
            acc=results[layer]
            total_results[layer].append(acc)
            if acc>best_reader_per_layer[layer]["best_acc"]:
                best_reader_per_layer[layer]["best_acc"]=acc
                best_reader_per_layer[layer]["best_reader"]=copy.deepcopy(current_reader)

    layer_stats=[]
    for layer,accuracies in total_results.items():
        mean_acc=np.mean(accuracies)
        std_acc=np.std(accuracies)
        layer_stats.append({"Layer":layer,"Mean_Acc":mean_acc,"Std_Dev":std_acc})
    
    df=pd.DataFrame(layer_stats)
    best_layer_row=df.loc[df["Mean_Acc"].idxmax()]
    best_layer=int(best_layer_row["Layer"])
    best_reader=best_reader_per_layer[best_layer]["best_reader"]
    print(f"Best Layer:{best_layer}층")
    print(f"Accuracy:{best_layer_row['Mean_Acc']:.4f}±{best_layer_row['Std_Dev']:.4f}")
    
    return (best_reader,best_layer,df)
    
def plotting(df,stimulus_number):
    layers=df["Layer"]
    means=df['Mean_Acc']
    stds=df["Std_Dev"]
    plt.figure(figsize=(10,6))
    plt.plot(layers,means,label="Validation Accuracy",color="blue",marker='o')
    plt.fill_between(layers,means-stds,means+stds,color="blue",alpha=0.2,label="standard Deviation")
    plt.title(f"Layer-wise Reading Accuracy (Stimulus{stimulus_number})",fontsize=14)
    plt.xlabel('Layer Index',fontsize=12)
    plt.ylabel('Accuracy',fontsize=12)
    plt.grid(True,linestyle='--',alpha=0.6)
    plt.legend()
    plt.show()


def evaluate_truthfulqa_mc1(model,tokenizer,hidden_layers,reader,val_layer):
    """
    TruthfulQA MC1 평가함수
    Args:
        model: 평가모델,
        tokenizer,
        hidden_layers,
        reader,
    Returns:
        Dict[layer][float]
    """
    dataset=load_dataset("truthful_qa","multiple_choice",split="validation")
    template="Consider the amount of truthfulness in the following answer:\nQuestion: {q}\nAnswer: {a}\nThe amount of truthfulness in the answer is"

    flat_inputs=[]
    flat_labels=[]
    group_sizes=[]

    model.eval()
    for item in tqdm(dataset):
        question=item["question"]
        choices=item["mc1_targets"]["choices"]
        labels=item["mc1_targets"]["labels"]
        group_sizes.append(len(choices))

        for choice,label in zip(choices,labels):
            input_text=template.format(q=question,a=choice)
            flat_inputs.append(input_text)
            flat_labels.append(label)

    results={}
    reading_pipeline=RepPipeline(model=model,tokenizer=tokenizer)
    reading_pipeline.reader=reader
    results=reading_pipeline.predict(
        test_inputs=flat_inputs,
        test_labels=flat_labels,
        hidden_layers=hidden_layers,
        mode="multi_choice",
        group_sizes=group_sizes
    )
    best_layer=max(results,key=results.get)
    best_acc=results[best_layer]
    print(f"Best Layer:{best_layer}층")
    print(f"Best Accuracy:{best_acc:.4f}({best_acc*100:.2f}%)")
    print(f"Val Layer:{val_layer}층")
    print(f"Val Layer Accuracy:{results[val_layer]:.4f}({results[val_layer]*100:.2f}%)")