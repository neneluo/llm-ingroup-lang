#!/bin/bash

export PYTHONPATH=${pwd}

set -x

result_dir=$1
model_prefix=$2
context_mode=$3
icl_mode=$4
eval_set=$5

python evaluation/metrics.py \
    --result_json ${result_dir}/explanation.${eval_set}.${model_prefix}.${context_mode}.${icl_mode}.jsonl \
    --task_type explain_bert_score

python evaluation/metrics.py \
    --result_json ${result_dir}/explanation.${eval_set}.${model_prefix}.${context_mode}.${icl_mode}.llm_judge.jsonl \
    --task_type explain_llm_judge

python evaluation/metrics.py \
    --result_json ${result_dir}/wsd.${eval_set}.${model_prefix}.${context_mode}.jsonl \
    --task_type wsd_from_conversation

python evaluation/metrics.py \
    --result_json ${result_dir}/ner_wnut.${model_prefix}.${context_mode}.jsonl \
    --task_type ner \
    --exclude_o Fasle

python evaluation/metrics.py \
    --result_json ${result_dir}/ner_wnut.${model_prefix}.${context_mode}.jsonl \
    --task_type ner \
    --exclude_o True
