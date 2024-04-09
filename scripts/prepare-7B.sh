#!/bin/bash

BASE_MODEL="01-ai/Yi-6B"
OUTPUT_MODEL="models/base-7B"

python3 src/training/prepare.py \
    --model=$BASE_MODEL \
    --output=$OUTPUT_MODEL
