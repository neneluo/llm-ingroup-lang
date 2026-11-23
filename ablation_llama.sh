#!/bin/bash

export PYTHONPATH=$(pwd)

set -x

stage="ablation"
simulation_data="experiments/simulation/simulation.single_turn_npmi_Llama-3.1-8B-Instruct_1_60000.jsonl"
model_id="meta-llama/Llama-3.1-8B-Instruct"
model_name="Llama-3.1-8B-Instruct"
model_type="llama"
context_mode="post_top_10"
icl_mode="three_shot"
eval_set="valid"

seed=42
training_data_dir="experiments/training_data_ablation_seed${seed}"
model_dir="experiments/models_ablation_seed${seed}"
result_dir="experiments/results_ablation_seed${seed}"
mkdir -p ${training_data_dir}
mkdir -p ${model_dir}
mkdir -p ${result_dir}

if [ "$stage" = "ablation" ]; then
    # synthetic data scale
    for scale in 7000 5000 3000 1000; do
        model_prefix="single_turn_npmi.${model_name}.synonym.${scale}"
        python models/instruction_ft.py  \
        --input_simulation_data "${simulation_data}" \
        --output_training_data "${training_data_dir}/training.${model_prefix}.jsonl" \
        --special_token_mask "False" \
        --data_amount "${scale}" \
        --model_id "${model_id}" \
        --output_dir "${model_dir}/training.${model_prefix}" \
        --seed "${seed}"
    done 

    # lexical
    scale=9000
    model_prefix="single_turn_npmi.${model_name}.synonym_mask.${scale}"
    python models/instruction_ft.py  \
        --input_simulation_data "${simulation_data}" \
        --output_training_data "${training_data_dir}/training.${model_prefix}.jsonl" \
        --special_token_mask "True" \
        --data_amount "${scale}" \
        --model_id "${model_id}" \
        --output_dir "${model_dir}/training.${model_prefix}" \
        --seed "${seed}"

    scale=9000
    model_prefix="single_turn_npmi.${model_name}.synonym_random.${scale}"
    python models/instruction_ft_random.py  \
        --input_simulation_data "${simulation_data}" \
        --output_training_data "${training_data_dir}/training.${model_prefix}.jsonl" \
        --special_token_mask "True" \
        --data_amount "${scale}" \
        --model_id "${model_id}" \
        --output_dir "${model_dir}/training.${model_prefix}" \
        --seed "${seed}"
fi

if [ "$stage" = "ablation" ]; then
    # synthetic data scale
    for scale in 7000 5000 3000 1000; do
        model_prefix="single_turn_npmi.${model_name}.synonym.${scale}"
        python evaluation/evaluate.py \
            --task "all" \
            --result_dir "${result_dir}" \
            --model_prefix "${model_prefix}" \
            --model_type "${model_type}" \
            --model_id "${model_id}" \
            --lora_dir "${model_dir}/training.${model_prefix}" \
            --context_mode "${context_mode}" \
            --icl_mode "${icl_mode}" \
            --seed "${seed}" \
            --eval_set "${eval_set}"
    done 

    # lexical
    scale=9000
    model_prefix="single_turn_npmi.${model_name}.synonym_mask.${scale}"
    python evaluation/evaluate.py \
        --task "all" \
        --result_dir "${result_dir}" \
        --model_prefix "${model_prefix}" \
        --model_type "${model_type}" \
        --model_id "${model_id}" \
        --lora_dir "${model_dir}/training.${model_prefix}" \
        --context_mode "${context_mode}" \
        --icl_mode "${icl_mode}" \
        --seed "${seed}" \
        --eval_set "${eval_set}"

    scale=9000
    model_prefix="single_turn_npmi.${model_name}.synonym_random.${scale}"
    python evaluation/evaluate.py \
        --task "all" \
        --result_dir "${result_dir}" \
        --model_prefix "${model_prefix}" \
        --model_type "${model_type}" \
        --model_id "${model_id}" \
        --lora_dir "${model_dir}/training.${model_prefix}" \
        --context_mode "${context_mode}" \
        --icl_mode "${icl_mode}" \
        --seed "${seed}" \
        --eval_set "${eval_set}"
fi
