#!/bin/bash

export PYTHONPATH=$(pwd)

set -x


model_prefix="gemini-3-flash-preview"
model_type="gemini"
seed=42
eval_set="test"
result_dir="experiments/results_gemini"
mkdir -p ${result_dir}

context_mode="post_top_10"
icl_mode="three_shot"
python evaluation/evaluate.py \
    --task "all" \
    --result_dir "${result_dir}" \
    --model_prefix "${model_prefix}" \
    --model_type "${model_type}" \
    --model_id "" \
    --lora_dir "" \
    --context_mode "${context_mode}" \
    --icl_mode "${icl_mode}" \
    --eval_set "${eval_set}" \
    --seed "${seed}"

