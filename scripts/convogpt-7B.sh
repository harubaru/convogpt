#!/bin/sh

MODEL=models/base-7B
OUTPUT_MODEL=models/convo-7B-002
DATA=dataset/convo-dataset-v0.jsonl
LR=1.2e-5

deepspeed --num_gpus 8 src/training/chatml.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir $OUTPUT_MODEL \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --seed 69 \
    --deepspeed ds_config.json

