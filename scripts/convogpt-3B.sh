#!/bin/sh

MODEL=models/base-3B
OUTPUT_MODEL=models/convo-3B-008
DATA=dataset/convo-dataset-v0.jsonl
LR=1.6e-5

torchrun --nproc_per_node=8 --master_port=9800 src/training/chatml.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir $OUTPUT_MODEL \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --model_max_length 2048 \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
