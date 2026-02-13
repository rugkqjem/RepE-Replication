import torch
import numpy as np
from datasets import load_dataset
from DataLoader import ConceptDatasetLoader
from ReadingPipeline import RepPipeline
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

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


def plot_detection_results(input_ids,score_list,start_answer_token=":",threshold=0):
    cmap=LinearSegmentedColormap.from_list('rg',["r",(255/255,255/255,224/255),"g"],N=256)
    colormap=cmap

    words=[token.replace('▁',' ') for token in input_ids]
    fig,ax=plt.subplots(figsize=(12.8,3),dpi=100)

    ax.axis('off')
    # Set limits for the x and y axes
    xlim = 1000
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 5)

    # Remove ticks and labels from the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Starting position of the words in the plot
    x_start, y_start = 1, 8
    y_pad = 0.3
    # Initialize positions and maximum line width
    x, y = x_start, y_start
    max_line_width = xlim

    y_pad = 0.3
    word_width = 0

    iter = 0

    
    start_idx=0
    found = False
    
    # words 리스트 전체를 돌면서 확인
    for i in range(len(words) - 2): # 뒤에 2개 더 봐야 하니까 -2
        current_word = words[i]
        next_word = words[i+1]
        next_next_word = words[i+2]
        
        # 조건: 현재 토큰이 '/' 이고, 바로 뒤가 'INST' 인지 확인
        # (Mistral 토크나이저는 보통 ['[', '/', 'INST', ']'] 이렇게 자름
        if '/' in current_word and 'INST' in next_word:
            start_idx = i + 3 
            found = True
            break

    if not found:
        print("Warning: [/INST] 태그를 찾지 못했습니다. 강제로 5번 인덱스로 설정합니다.")
        start_idx = 5
    
    print(f"Found start_idx: {start_idx} (Word: {words[start_idx]})") # 확인용 출력


    full_mean=np.median(score_list)
    full_std=np.std(score_list)


    answer_scores=score_list[start_idx:]
    answer_words=words[start_idx:]
    print(answer_words)

    if len(answer_scores)<2:
        answer_scores=score_list[5:]
        answer_words=words[5:]

    answer_scores=np.array(answer_scores)
    answer_words=np.array(answer_words)

    mean=np.median(answer_scores)
    print("score mean:",mean)
    print("full score mean:",full_mean)
    std=np.std(answer_scores)
    print("std:",std)
    print("full_std:",full_std)
    answer_scores[(answer_scores>mean+5*std)|(answer_scores<mean-5*std)]=mean
    
    mag = max(0.3, np.abs(answer_scores).std() / 10)
    min_val, max_val = -mag, mag
    norm = Normalize(vmin=min_val, vmax=max_val)

    #입력 데이터 기준 정규화 
    answer_scores=[score-threshold for score in answer_scores]


    if np.std(answer_scores)==0:
        answer_scores=answer_scores
    else:
        answer_scores=answer_scores/np.std(answer_scores)
    answer_scores=np.clip(answer_scores,-mag,mag)
    answer_scores=np.clip(answer_scores,-np.inf,0)
    answer_scores[answer_scores==0]=mag

    x, y = x_start, y_start
    max_line_width = xlim

    for word,score in zip(answer_words,answer_scores):
        if start_answer_token in word:
            continue
        color=colormap(norm(score))
        if x + word_width>max_line_width:
            x=x_start
            y-=3

        text=ax.text(x,y,word,fontsize=13)
        word_width = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width
        word_height = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).height

        if iter:
            text.remove()

        text=ax.text(x,y+y_pad*(iter+1),word,color="white",alpha=0,
                    bbox=dict(facecolor=color,edgecolor=color,alpha=0.8,boxstyle=f"round,pad=0",linewidth=0),
                    fontsize=13)
        
        x+=word_width+0.1 
    iter+=1
    


def get_choice_log_likelihood(model,tokenizer,user_tag,assistant_tag,question,choices):
    """
    질문과 선택을 받아, 가장 확률이 높은 선택지의 인덱스 반환
    """
    model.eval()
    losses=[]
    with torch.no_grad():
        for choice in choices:
            prompt=f"{user_tag} Question:{question}\nAnswer:{assistant_tag}"
            full_text=prompt+" "+choice
            input_ids=tokenizer(full_text,return_tensors="pt").input_ids.to(model.device)
            prompt_ids=tokenizer(prompt,return_tensors="pt").input_ids.to(model.device)
            labels=input_ids.clone()
            prompt_len=prompt_ids.shape[1]
            labels[:,:prompt_len]=-100
            outputs=model(input_ids,labels=labels)
            neg_log_likelihood=outputs.loss.item()
            losses.append(neg_log_likelihood)

    predicted_index=np.argmin(losses)
    return predicted_index

def standard_truthfulQA_evaluation(model,tokenizer,user_tag,assistant_tag):
    ds=load_dataset("truthful_qa","multiple_choice",split="validation")
    correct_count=0
    total_count=0
    
    for item in tqdm(ds):
        question=item['question']
        choices=item['mc1_targets']['choices']
        labels=item['mc1_targets']['labels']
        ground_truth_idx=labels.index(1)
        predict_idx=get_choice_log_likelihood(
                    model=model,tokenizer=tokenizer,
                    user_tag=user_tag,assistant_tag=assistant_tag,
                    question=question,choices=choices)
        if predict_idx==ground_truth_idx:
            correct_count+=1
        total_count+=1
    
    accuracy=correct_count/total_count
    print(f"\nfinal accuracy:{accuracy:.4f}")
