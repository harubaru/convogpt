#!/bin/bash

BASE_MODEL="openlm-research/open_llama_3b_v2"
OUTPUT_MODEL="models/base-3B"

python3 src/training/prepare.py \
    --model=$BASE_MODEL \
    --output=$OUTPUT_MODEL
