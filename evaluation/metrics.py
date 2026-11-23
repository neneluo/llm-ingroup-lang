import jsonlines
import torch
from bert_score import score
import csv
import argparse
import string
from collections import Counter


def compute_success_rate(result_json):
    simulation_round = 0.0
    gen_success_round = 0.0
    com_em_success_round = 0.0
    com_cos_success_round = 0.0
    similarity_t = 0.7
    with jsonlines.open(result_json, "r") as reader:
        for item in reader:
            simulation_round += 1
            gen_success_round += int(item["gen_outcome"])
            com_em_success_round += int(item["com_outcome_em"])
            similarity = float(item["word_similarity"])
            if similarity > similarity_t:
                com_cos_success_round += 1
    gen_success_rate = (gen_success_round / simulation_round) * 100
    com_em_success_rate = (com_em_success_round / simulation_round) * 100
    com_cos_success_rate = (com_cos_success_round / simulation_round) * 100
    print(f"Simulation rounds: {simulation_round:.0f}")
    print(f"Generation success rate: {gen_success_rate:.2f}%")
    print(f"Communication EM success rate: {com_em_success_rate:.2f}%")
    print(f"Communication CS success rate: {com_cos_success_rate:.2f}%")


def compute_bert_score(result_json):
    # comparing between reference and model generation
    refs = []
    hyps = []
    num_empty = 0
    rows = []
    with jsonlines.open(result_json, mode="r") as reader:
        for line in reader:
            community = line["community"]
            term = line["term"]
            definition = line["description"].lower()
            model_generation = line["model_generation"].lower()

            if definition.strip() == '':
                print(f"Empty definition: {line}")
                continue

            if model_generation.strip() == '':
                # skip generation for empty sentence as it throws tokenization error 
                # but record number for calculating mean
                num_empty += 1
            else:
                refs.append(definition)
                hyps.append(model_generation)
                rows.append([community, term, definition, model_generation])

    assert len(hyps) == len(refs)
    print(f"Scoring {result_json}, {len(hyps)} instances")
    P, R, F1 = score(cands=hyps, refs=refs, lang='en', verbose=True, rescale_with_baseline=True)
    P_valid = torch.cat([P, torch.zeros(num_empty)])
    R_valid = torch.cat([R, torch.zeros(num_empty)])
    F1_valid = torch.cat([F1, torch.zeros(num_empty)])
    print(f"System level Precision: {P_valid.mean():.2f}")
    print(f"System level Recall: {R_valid.mean():.2f}")
    print(f"System level F1 score: {F1_valid.mean():.2f}")

    output_writer = open("analyse.csv", mode="w", newline='')
    output_csv_writer = csv.writer(output_writer)
    header = ["community", "term", "definition", "model_generation", "P", "R", "F1"]
    output_csv_writer.writerow(header)
    for i, row in enumerate(rows):
        row.append(P_valid[i].item())
        row.append(R_valid[i].item())
        row.append(F1_valid[i].item())
        output_csv_writer.writerow(row)
    output_writer.close()


def compute_llm_score(result_json):
    good = 0
    bad = 0
    borderline = 0
    amount = 0
    result_reader = jsonlines.open(result_json, mode="r")
    for line in result_reader:
        label = line["label"]
        if label.strip().lower() == "good":
            good += 1
        if label.strip().lower() == "bad":
            bad += 1
        if label.strip().lower() == "borderline":
            borderline += 1
        amount += 1
    result_reader.close()
    print(f"LLM-as-a-judge Evaluation: {result_json}")
    print(f"Good: {good/amount*100:.2f}")
    print(f"Bad: {bad/amount*100:.2f}")
    print(f"Borderline: {borderline/amount*100:.2f}")


def compute_wsd_f1(result_json):
    result_reader = jsonlines.open(result_json, mode="r")
    f1_list = []
    for line in result_reader:
        f1 = float(line["f1"])
        f1_list.append(f1)
    
    print(f"Average F1: {sum(f1_list) / len(f1_list)}")
    result_reader.close()


def normalize_answer(s):
    """Normalize QA answer
    codes adapted from: https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py#L14

    Args:
        s (str): a string to normalize
    
    Returns:
        str: a normalized string
    """
    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    return white_space_fix(handle_punc(lower(s))).strip()

def compute_f1(prediction, ground_truth):
    """Compute F1 score.
    codes adapted from: https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py#L36

    Args:
        pred (str): model prediction
        gt_list (list): a list of ground truth labels

    Returns:
        float: f1 score
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_wsd_f1_read_from_conversation(result_json):
    result_reader = jsonlines.open(result_json, mode="r")
    f1_list = []
    for line in result_reader:
        conversation = line["conversation"][-1]
        assert conversation["role"] == "assistant"
        prediction = conversation["content"]
        if prediction is None:
            print(f"wsd predition None: {line['description']}")
            prediction = ""
        prediction_strip = prediction.removeprefix("1. ").removeprefix("2. ").removeprefix("3. ").removeprefix("4. ")
        ground_truth = line["description"]
        f1 = compute_f1(prediction=prediction_strip, ground_truth=ground_truth)
        f1_list.append(f1)

    result_reader.close()
    print(f"Average f1: {sum(f1_list) / len(f1_list)}")


def compute_ner_acc(result_json, exclude_o="False"):
    if exclude_o in ["true", "True"]:
        print("Excluding O categories ...")
    
    result_reader = jsonlines.open(result_json, mode="r")
    predictions = []
    labels = []
    correct_tag = 0
    amount = 0
    for line in result_reader:
        ner_tag = line["ner_tag"]
        model_predicted_tag = line["ner_tag (model_generated)"]

        if exclude_o in ["true", "True"]:
            if ner_tag == "O":
                continue
        
        predictions.append(model_predicted_tag)
        labels.append(ner_tag)

        if model_predicted_tag.lower() == ner_tag.lower():
            correct_tag += 1
        amount += 1

    result_reader.close()
    print(f"Evaluating {result_json}.")
    print(f"NER Tagging Sentence Amount: {amount}")
    print(f"NER Tagging Acc: {correct_tag / amount * 100:.2f}%")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(prog='metrics.py')
    parser.add_argument('--result_json', type=str, help="Path to result.jsonl.")
    parser.add_argument('--task_type', type=str, help="Task type: simulation/explain_bert_score/explain_llm_judge/wsd/ner")
    parser.add_argument('--exclude_o', type=str, default="False", help="Exclude O category in NER.")
    
    args = parser.parse_args()
    print(args)

    if args.task_type == "simulation":
        compute_success_rate(result_json=args.result_json)
    
    if args.task_type == "explain_bert_score":
        compute_bert_score(result_json=args.result_json)
    if args.task_type == "explain_llm_judge":
        compute_llm_score(result_json=args.result_json)
    if args.task_type == "wsd":
        compute_wsd_f1(result_json=args.result_json)
    if args.task_type == "wsd_from_conversation":
        compute_wsd_f1_read_from_conversation(result_json=args.result_json)
    if args.task_type == "ner":
        compute_ner_acc(
            result_json=args.result_json, 
            exclude_o=args.exclude_o
        )