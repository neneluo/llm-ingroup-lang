#!/bin/bash

set -x

seed=$1
stage="all"

if [ "$stage" = "all" ]; then
    echo "stage: main ..."
    context_mode="post_top_10"
    icl_mode="three_shot"
    scale=9000
    eval_set="valid"

    model_name="gemini-3-flash-preview"
    result_dir="experiments/results_gemini_seed${seed}"
    sh metric.sh "${result_dir}" "${model_name}" "${context_mode}" "${icl_mode}" "${eval_set}"

    
    model_name="Llama-3.1-8B-Instruct"
    result_dir="experiments/results_seed${seed}"
    sh metric.sh "${result_dir}" "${model_name}" "${context_mode}" "${icl_mode}" "${eval_set}"

    model_prefix="${model_name}.ntp.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"

    result_dir="experiments/results_seed${seed}"
    model_prefix="single_turn_npmi.${model_name}.synonym_instruct.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"

    result_dir="experiments/results_seed${seed}"
    model_prefix="single_turn_npmi.${model_name}.synonym.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"
fi

if [ "$stage" = "all" ]; then
    echo "stage: ablation ..."
    eval_set="valid"
    context_mode="post_top_10"
    icl_mode="three_shot"
    
    model_name="Llama-3.1-8B-Instruct"
    result_dir="experiments/results_ablation_seed${seed}"

    for scale in 1000 3000 5000 7000; do
        model_prefix="single_turn_npmi.${model_name}.synonym.${scale}"
        sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"
    done

    scale=9000
    model_prefix="single_turn_npmi.${model_name}.synonym_mask.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"
    model_prefix="single_turn_npmi.${model_name}.synonym_random.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"
fi


if [ "$stage" = "all" ]; then
    echo "stage: test set and holdout set ..."
    eval_set="test"

    model_name="gemini-3-flash-preview"
    result_dir="experiments/results_gemini_seed${seed}"
    sh metric.sh "${result_dir}" "${model_name}" "${context_mode}" "${icl_mode}" "${eval_set}"


    model_name="Llama-3.1-8B-Instruct"
    result_dir="experiments/results_seed${seed}"
    sh metric.sh "${result_dir}" "${model_name}" "${context_mode}" "${icl_mode}" "${eval_set}"

    model_prefix="${model_name}.ntp.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"

    model_prefix="single_turn_npmi.${model_name}.synonym_instruct.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"

    model_prefix="single_turn_npmi.${model_name}.synonym.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"


    eval_set="hold_out"
    model_name="gemini-3-flash-preview"
    result_dir="experiments/results_gemini_seed${seed}"
    sh metric.sh "${result_dir}" "${model_name}" "${context_mode}" "${icl_mode}" "${eval_set}"


    model_name="Llama-3.1-8B-Instruct"
    result_dir="experiments/results_seed${seed}"
    sh metric.sh "${result_dir}" "${model_name}" "${context_mode}" "${icl_mode}" "${eval_set}"

    model_prefix="${model_name}.ntp.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"

    model_prefix="single_turn_npmi.${model_name}.synonym_instruct.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"

    model_prefix="single_turn_npmi.${model_name}.synonym.${scale}"
    sh metric.sh "${result_dir}" "${model_prefix}" "${context_mode}" "${icl_mode}" "${eval_set}"
fi
