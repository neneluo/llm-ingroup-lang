#!/bin/bash

export PYTHONPATH=$(pwd)

set -x
eval_set="valid"

# main
for seed in 19 42 999; do
    result_dir="experiments/results_seed${seed}"
    model_name="Llama-3.1-8B-Instruct"
    result_prefix="${result_dir}/explanation.${eval_set}.${model_name}.post_top_10.three_shot"
    python evaluation/score.py "${result_prefix}.jsonl" "${result_prefix}.llm_judge.jsonl" "${seed}"

    result_prefix="${result_dir}/explanation.${eval_set}.${model_name}.ntp.9000.post_top_10.three_shot"
    python evaluation/score.py "${result_prefix}.jsonl" "${result_prefix}.llm_judge.jsonl" "${seed}"

    result_prefix="${result_dir}/explanation.${eval_set}.single_turn_npmi.${model_name}.synonym.9000.post_top_10.three_shot"
    python evaluation/score.py "${result_prefix}.jsonl" "${result_prefix}.llm_judge.jsonl" "${seed}"

    result_prefix="${result_dir}/explanation.${eval_set}.single_turn_npmi.${model_name}.synonym_instruct.9000.post_top_10.three_shot"
    python evaluation/score.py "${result_prefix}.jsonl" "${result_prefix}.llm_judge.jsonl" "${seed}"
done

# ablation
for seed in 19 42 999; do
    result_dir="experiments/results_ablation_seed${seed}"
    model_name="Llama-3.1-8B-Instruct"
    result_prefix="${result_dir}/explanation.${eval_set}.single_turn_npmi.${model_name}.synonym_mask.9000.post_top_10.three_shot"
    python evaluation/score.py "${result_prefix}.jsonl" "${result_prefix}.llm_judge.jsonl" "${seed}"

    result_prefix="${result_dir}/explanation.${eval_set}.single_turn_npmi.${model_name}.synonym_random.9000.post_top_10.three_shot"
    python evaluation/score.py "${result_prefix}.jsonl" "${result_prefix}.llm_judge.jsonl" "${seed}"
done
