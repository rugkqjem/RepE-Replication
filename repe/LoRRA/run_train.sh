#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train_lorra.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --user_tag '[INST]' \
    --assistant_tag '[/INST]' \
    --pos_type "a truthful" \
    --neg_type 'an untruthful' \
    --control_template "Give {type} answer." \
    --target_layers "10,12,14,16,18,20" \
    --lorra_alpha 5 \
    --lorra_beta 0 \
    --lora_r 8 \
    --lora_dropout 0.05 \
    --output_dir ./lorra_tqa_7b \
    --overwrite_output_dir \
    --max_steps 70 \
    --bf16 True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --evaluation_strategy "steps" \
    --save_total_limit 0 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --q_lora False \
    --gradient_checkpointing True \
    --report_to none
