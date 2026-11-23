#!/bin/bash

export PYTHONPATH=$(pwd)

set -x

stage="main"
simulation_data="experiments/simulation/simulation.single_turn_npmi_Llama-3.1-8B-Instruct_1_60000.jsonl"
model_id="meta-llama/Llama-3.1-8B-Instruct"
model_name="Llama-3.1-8B-Instruct"
model_type="llama"
context_mode="post_top_10"
icl_mode="three_shot"

seed=42
training_data_dir="experiments/training_data_seed${seed}"
model_dir="experiments/models_seed${seed}"
result_dir="experiments/results_seed${seed}"
mkdir -p ${training_data_dir}
mkdir -p ${model_dir}
mkdir -p ${result_dir}

if [ "$stage" = "main" ]; then
    scale=9000
    # supervised fine-tuning in game-based simulation
    model_prefix="single_turn_npmi.${model_name}.synonym.${scale}"
    python models/instruction_ft.py  \
        --input_simulation_data "${simulation_data}" \
        --output_training_data "${training_data_dir}/training.${model_prefix}.jsonl" \
        --special_token_mask "False" \
        --data_amount "${scale}" \
        --model_id "${model_id}" \
        --output_dir "${model_dir}/training.${model_prefix}" \
        --seed "${seed}"
    
    # next token prediction on contexts
    model_prefix="${model_name}.ntp.${scale}"
    python models/context_ft.py  \
        --output_training_data "${training_data_dir}/training.${model_prefix}.jsonl" \
        --data_amount "${scale}" \
        --model_id "${model_id}" \
        --output_dir "${model_dir}/training.${model_prefix}" \
        --seed "${seed}"
    
    # supervised fine-tuning in instruction-based simulation
    model_prefix="single_turn_npmi.${model_name}.synonym_instruct.${scale}"
    python models/simu_instruct_ft.py  \
        --input_simulation_data "${simulation_data}" \
        --output_training_data "${training_data_dir}/training.${model_prefix}.jsonl" \
        --data_amount "${scale}" \
        --model_id "${model_id}" \
        --output_dir "${model_dir}/training.${model_prefix}" \
        --seed "${seed}"
fi

if [ "$stage" = "main" ]; then
    eval_set="valid"
    scale=9000
    # supervised fine-tuning in game-based simulation
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

    # next token prediction on contexts
    model_prefix="${model_name}.ntp.${scale}"
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

    # supervised fine-tuning in instruction-based simulation
    model_prefix="single_turn_npmi.${model_name}.synonym_instruct.${scale}"
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

if [ "$stage" = "main" ]; then
    eval_set="test"
    scale=9000
    # supervised fine-tuning in game-based simulation
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

    # next token prediction on contexts
    model_prefix="${model_name}.ntp.${scale}"
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

    # supervised fine-tuning in instruction-based simulation
    model_prefix="single_turn_npmi.${model_name}.synonym_instruct.${scale}"
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


if [ "$stage" = "main" ]; then
    eval_set="hold_out"
    scale=9000
    # supervised fine-tuning in game-based simulation
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

    # next token prediction on contexts
    model_prefix="${model_name}.ntp.${scale}"
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

    # supervised fine-tuning in instruction-based simulation
    model_prefix="single_turn_npmi.${model_name}.synonym_instruct.${scale}"
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
