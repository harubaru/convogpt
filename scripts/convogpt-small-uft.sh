#!/bin/bash

BASE_MODEL="EleutherAI/gpt-neo-125M"
DATASET="convo-dataset-neo.tokens"
OUTPUT_DIR="models/convogpt-small-uft"
EPOCHS=2
BATCH_SIZE=1
SAVE_STEPS=50
LEARNING_RATE=1e-5

accelerate launch src/training/uft.py \
    --model $BASE_MODEL \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --learning_rate $LEARNING_RATE