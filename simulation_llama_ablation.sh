#!/bin/bash

export PYTHONPATH=$(pwd)

set -x

stage="main"
data_start=1
data_end=1000
seed=42

simulation_dir="experiments/simulation_ablation"
mkdir -p ${simulation_dir}

if [ "$stage" = "main" ]; then
    special_token_mask="False"

    model_id="meta-llama/Llama-3.2-1B-Instruct"
    model_type="llama"
    model_prefix="Llama-3.2-1B-Instruct"

    sampling_strategies="random"
    simulation_data="${simulation_dir}/simulation_inno.single_turn_${sampling_strategies}_${model_prefix}_${data_start}_${data_end}.jsonl"
    python models/simulation.py \
        --output_simulation_data "${simulation_data}" \
        --model_id "${model_id}" \
        --model_type "${model_type}" \
        --data_start "${data_start}" \
        --data_end "${data_end}" \
        --sampling_strategies "${sampling_strategies}" \
        --special_token_mask "${special_token_mask}" \
        --seed "${seed}"

    sampling_strategies="npmi"
    simulation_data="${simulation_dir}/simulation_inno.single_turn_${sampling_strategies}_${model_prefix}_${data_start}_${data_end}.jsonl"
    python models/simulation.py \
        --output_simulation_data "${simulation_data}" \
        --model_id "${model_id}" \
        --model_type "${model_type}" \
        --data_start "${data_start}" \
        --data_end "${data_end}" \
        --sampling_strategies "${sampling_strategies}" \
        --special_token_mask "${special_token_mask}" \
        --seed "${seed}"

    model_id="meta-llama/Llama-3.1-8B-Instruct"
    model_type="llama"
    model_prefix="Llama-3.1-8B-Instruct"

    sampling_strategies="random"
    simulation_data="${simulation_dir}/simulation_inno.single_turn_${sampling_strategies}_${model_prefix}_${data_start}_${data_end}.jsonl"
    python models/simulation.py \
        --output_simulation_data "${simulation_data}" \
        --model_id "${model_id}" \
        --model_type "${model_type}" \
        --data_start "${data_start}" \
        --data_end "${data_end}" \
        --sampling_strategies "${sampling_strategies}" \
        --special_token_mask "${special_token_mask}" \
        --seed "${seed}"

    sampling_strategies="npmi"
    simulation_data="${simulation_dir}/simulation_inno.single_turn_${sampling_strategies}_${model_prefix}_${data_start}_${data_end}.jsonl"
    python models/simulation.py \
        --output_simulation_data "${simulation_data}" \
        --model_id "${model_id}" \
        --model_type "${model_type}" \
        --data_start "${data_start}" \
        --data_end "${data_end}" \
        --sampling_strategies "${sampling_strategies}" \
        --special_token_mask "${special_token_mask}" \
        --seed "${seed}"
fi

if [ "$stage" = "main" ]; then
    special_token_mask="False"

    model_id="meta-llama/Llama-3.1-70B-Instruct"
    model_type="llama"
    model_prefix="Llama-3.1-70B-Instruct"

    sampling_strategies="random"
    simulation_data="${simulation_dir}/simulation_inno.single_turn_${sampling_strategies}_${model_prefix}_${data_start}_${data_end}.jsonl"
    python models/simulation.py \
        --output_simulation_data "${simulation_data}" \
        --model_id "${model_id}" \
        --model_type "${model_type}" \
        --data_start "${data_start}" \
        --data_end "${data_end}" \
        --sampling_strategies "${sampling_strategies}" \
        --special_token_mask "${special_token_mask}" \
        --seed "${seed}"

    sampling_strategies="npmi"
    simulation_data="${simulation_dir}/simulation_inno.single_turn_${sampling_strategies}_${model_prefix}_${data_start}_${data_end}.jsonl"
    python models/simulation.py \
        --output_simulation_data "${simulation_data}" \
        --model_id "${model_id}" \
        --model_type "${model_type}" \
        --data_start "${data_start}" \
        --data_end "${data_end}" \
        --sampling_strategies "${sampling_strategies}" \
        --special_token_mask "${special_token_mask}" \
        --seed "${seed}"
fi
