#!/bin/bash

export PYTHONPATH=$(pwd)

set -x

model_id="meta-llama/Llama-3.1-8B-Instruct"
model_type="llama"
model_prefix="Llama-3.1-8B-Instruct"
data_start=1
data_end=15000
seed=42

simulation_dir="experiments/simulation"
mkdir -p ${simulation_dir}

special_token_mask="False"
sampling_strategies="npmi"
simulation_data="${simulation_dir}/simulation.single_turn_${sampling_strategies}_${model_prefix}_${data_start}_${data_end}.jsonl"
python models/simulation.py \
    --output_simulation_data "${simulation_data}" \
    --model_id "${model_id}" \
    --model_type "${model_type}" \
    --data_start "${data_start}" \
    --data_end "${data_end}" \
    --sampling_strategies "${sampling_strategies}" \
    --special_token_mask "${special_token_mask}" \
    --seed "${seed}"
