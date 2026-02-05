# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import logging
import pathlib
import typing
import os
import json
import gc
from typing import Dict, Optional, Sequence

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig, deepspeed
import torch
from data_preprocessing import AlpacaSupervisedDataset, load_tqa_sentences, load_arc_sentences, get_logprobs_accuracy
import pickle

from args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)

#tokenize 된 inputs (for train) -> return (target layer별 val )
def compute_loss(self,model,inputs,taret_layers,alpha,beta,max_res_len,return_outputs=False,**kwargs):
    input_ids=inputs.get('input_ids')
    attention_mask=inputs.get("attention_mask")

    orig_input_ids=input_ids[:,0]
    pos_inputs_ids=input_ids[:,1]
    neg_inputs_ids=input_ids[:,2]
    
    orig_attention_mask=attention_mask[:,0]
    pos_attention_mask=attention_mask[:,1]
    neg_attention_mask=attention_mask[:,2]

    min_length=max_res_len
    #layer 차원까지 확장해야해서 (shape(layer,batch,seq,hidden))
    response_attention_mask=orig_attention_mask[:,-min_length:].repeat(len(target_layers),1,1).unsqueeze(-1)

    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            orig_outputs=model(
                input_ids=orig_inputs_ids,
                attention_mask=orig_attention_mask,
                output_hidden_states=True
            )["hidden_states"]
            orig_hidden=[orig_outputs[l][:,-min_length:].detach() for l in target_layers]
            pos_outputs=model(
                input_ids=pos_inputs_ids,
                attention_mask=pos_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            neg_outputs=model(
                input_ids=neg_inputs_ids,
                attention_mask=neg_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            direction_hidden=[pos_outputs[l][:,-min_length:].detach() - neg_outputs[l][:,-min_length:].detach() for l in target_layers]
            target_hidden=torch.stack([orig_hidden[i]+alpha*direction_hidden[i] for i in range(len(taret_layers))]) * response_attention_mask

            del orig_outputs,pos_outputs,neg_outputs,orig_hidden,direction_hidden
            gc.collect()
            torch.cuda.empty_cache()
        
    #LoRA모델 학습----------------------------------------------------------------------------
    model.train()
    lora_outputs=model(
        input_ids=orig_input_ids,
        attention_mask=orig_attention_mask,
        output_hidden_states=True
    )['hidden_states']

    lora_hidden=torch.stack([lora_outputs[l][:,-min_length:] for l in target_layers])*response_attention_mask
    loss_fct=torch.nn.MSELoss()  #(layer , sample)
    loss=torch.norm(lora_hidden-target_hidden,dim=-1,p=2,dtype=torch.float).nanmean() #모든 레이어의 오차를 다 더해 평균 계산
    return (loss,lora_hidden) if return_outputs else loss 


def train():
    parser=transformers.HfArgumentParser(
        (ModelArguments,TrainingArguments,LoraArguments,LorraArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        lorra_args,
    )=parser.parse_args_into_dataclasses()

    device_map="auto"
    
    compute_dtype=(
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model=transformers.AutoModelForCausalLM.from_pretrained(
        model=model_args.model_name_or_path,
        device_map=device_map
    )

    lorra_target_layers=[int(layer) for layer in lorra_args.target_layers.split(",")]
    lora_layers_to_transform=list(range(lorra_target_layers[-1]+1))

    lora_config=LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_layers_to_transform,
        task_type="CAUSAL_LM"
    )

    model=get_peft_model(model,lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer=transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False, 
    )

    tokenizer.pad_token=tokenizer.unk_token

    train_dataset=AlpacaSupervisedDataset(tokenizer=tokenizer,num_examples=10000,lorra_args=lorra_args)
    if training_args.do_eval:
        val_datasets={
            "tqa" : load_tqa_sentences(lorra_args.user_tag,lorra_args.assistant_tag),
            "arc-c" : load_arc_sentences(),
        }
        batch_size=training_args.per_device_eval_batch_size
    else:
        val_datasets={}

    class CustomTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False):
            return compute_loss(self,
                                model,
                                inputs,
                                target_layers=lorra_target_layers,
                                alpha=lorra_args.lorra_alpha,
                                beta=lorra_args.lorra_beta,
                                max_res_len=lorra_args.max_res_len,
                                return_outputs=return_outputs)

        def evaluate(self,eval_dataset=None,ignore_keys=None,sanity_check=False,**kwargs):
            self.model.eval()

            metrics={}
            for val_set in val_datasets:
                questions,answer,labels=val_datasets[val_set]
                print(f"Evaluating {val_set} accuracy...")
                with torch.no_grad():
                    acc=get_logprobs_accuracy(self.model,self.tokenizer,questions,answer,labels,batch_size)
                    acc_key="acc" if val_set=="tqa" else "acc_norm"
                    metrics[f"{val_set}_accuracy"]=acc[acc_key]
            self.model.train()
            print("===Eval results===")
            print(metrics)
            return metrics

    trainer=CustomTrainer(
        model=model,tokenizer=tokenizer,args=training_args,train_dataset=train_dataset
    )
    model.config.use_cache=False
    trainer.evaluate(eval_dataset=val_datasets,sanity_check=True)
    trainer.train()
    trainer.save_state()

    if training_args.local_rank==0:
        merged_model=model.merge_and_unload()
        merged_model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

if __name__=="__main__":
    train()