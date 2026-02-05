from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import transformers
import typing

@dataclass
class LorraArguments:
    user_tag : str = 'USER:'
    assistant_tag : str = 'ASSISTANT:'
    pos_type : str = "a truthful"
    neg_type : str = "an untruthful"
    target_layers : str = '10,12,14,16,18,20'
    control_template : str = "Give a {type} answer"
    lorra_alpha : float = 5
    lorra_beta : float = 0 
    max_res_len : int = 64

@dataclass
class LoraArguments:
    lora_r:int=8
    lora_alpha:int=16
    lora_dropout: float=0.05
    lora_target_modules: yping.List[str] = ["q_proj","v_proj"]
    lora_weight_path : str =""
    lora_bias: str = "none"
    q_lora : bool = False 

@dataclass
class ModelArguments:
    model_name_or_path : Optional[str] = "meta-llama/Llama-2-7b-chat-hf"
    adapter_name_or_path:str=field(default=None, metadata={"help":"Adapater name"})
    use_lora :field(default=False,metadata={"help":"Use LoRA(default:False)"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir : Optional[str] = field(default=None)
    optim : str = field(default="adamw_torch")
    model_max_length : int = 512
    grouped_to_max_length : bool = False