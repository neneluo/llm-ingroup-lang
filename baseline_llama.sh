#!/bin/bash

export PYTHONPATH=$(pwd)

set -x

model_id="meta-llama/Llama-3.1-8B-Instruct"
model_prefix="Llama-3.1-8B-Instruct"
model_type="llama"

seed=42
model_dir="experiments/models_seed${seed}"
result_dir="experiments/results_seed${seed}"
mkdir -p ${model_dir}
mkdir -p ${result_dir}

context_mode="post_top_10"
icl_mode="three_shot"
for eval_set in valid test; do
    python evaluation/evaluate.py \
        --task "all" \
        --result_dir "${result_dir}" \
        --model_prefix "${model_prefix}" \
        --model_type "${model_type}" \
        --model_id "${model_id}" \
        --lora_dir "" \
        --context_mode "${context_mode}" \
        --icl_mode "${icl_mode}" \
        --eval_set "${eval_set}" \
        --seed "${seed}"        
done

