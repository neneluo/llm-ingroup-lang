import torch
import jsonlines
from bert_score import score
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
import csv
import string
from collections import Counter
import argparse
from preprocessing.path import *
from models.prompts import *
from models.agent import Agent, inference_gemini


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


def explanation(agent, glossary_path, result_jsonl, context_mode, icl_mode, model_mode):
    # glossary: term, community, definition
    glossary_reader = jsonlines.open(glossary_path, mode="r")
    results_writer = jsonlines.open(result_jsonl, mode="w", flush=True)
    refs = []
    cands = []
    num_empty = 0
    for line in glossary_reader:
        community = line["subreddit"]
        term = line["term"]
        description = line["definition"]
        post_random_1 = line["post_random_1"]
        post_top_1 = line["post_top_1"]
        post_top_10 = line["post_top_10"]

        if context_mode == "post_random_1":
            context = post_random_1
        elif context_mode == "post_top_1":
            context = post_top_1
        elif context_mode == "post_top_10":
            context = post_top_10
        else:
            print(f"Error: unsupported context mode {context_mode}.")
            exit(0)

        if icl_mode == "zero_shot":
            evaluation_prompt = evaluation_free_explanation
        elif icl_mode == "one_shot":
            evaluation_prompt = evaluation_free_explanation_one_shot
        elif icl_mode == "three_shot":
            evaluation_prompt = evaluation_free_explanation_three_shot

        content = evaluation_prompt.format(
            term=term, community=community, context=context
        )
        
        if model_mode == "gemini":
            model_response = inference_gemini(content=content)
            conversation = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": model_response}
            ]
        elif model_mode == "llama":
            conversation = [
                {"role": "user", "content": content},
            ]
            model_response = agent.inference_llama(conversation)
            conversation.append({"role": "assistant", "content": model_response})
        else:
            print(f"Error: {model_mode} not supported!")
            exit(0)

        try:
            model_gen_description = model_response.split("Answer:")[1].strip()
        except:
            model_gen_description = model_response
            if model_response is None:
                model_gen_description = '' # placeholder to replace empty string

        if description.strip() == '':
            print(f"Empty description: {line}")
            continue

        if model_gen_description.strip() == '':
            # skip generation for empty sentence as it throws tokenization error 
            # but record number for calculating mean
            num_empty += 1
        else:
            refs.append(description.lower())
            cands.append(model_gen_description.lower())

        outline = {
            "community": community,
            "term": term,
            "description": description,
            "model_generation": model_gen_description,
            "context": context,
            "conversation": conversation
        }
        results_writer.write(outline)

    glossary_reader.close()
    results_writer.close()

    assert len(cands) == len(refs)
    print(f"Scoring {len(cands)} instances.")
    P, R, F1 = score(cands=cands, refs=refs, lang='en', verbose=True, rescale_with_baseline=True)
    P_valid = torch.cat([P, torch.zeros(num_empty)])
    R_valid = torch.cat([R, torch.zeros(num_empty)])
    F1_valid = torch.cat([F1, torch.zeros(num_empty)])
    print(f"System level Precision: {P_valid.mean():.2f}")
    print(f"System level Recall: {R_valid.mean():.2f}")
    print(f"System level F1 score: {F1_valid.mean():.2f}")
    

def word_sense_diambiguation(agent, wsd_jsonl, result_jsonl, context_mode, model_mode):
    wsd_json_reader = jsonlines.open(wsd_jsonl, mode="r")
    result_writer = jsonlines.open(result_jsonl, mode="w", flush=True)

    f1_list = []
    for line in wsd_json_reader:
        community = line["subreddit"]
        term = line["term"]
        description = line["description"]
        post_top_1 = line["post_top_1"]
        post_top_10 = line["post_top_10"]
        post_random_1 = line["post_random_1"]
        wsd_options = line["wsd_options"]

        if context_mode == "post_top_1":
            context = post_top_1
        elif context_mode == "post_top_10":
            context = post_top_10
        elif context_mode == "post_random_1":
            context = post_random_1
        else:
            print(f"Error: context mode {context_mode} not supported!")
            exit(0)
        
        content = evaluation_word_sense_diambiguation.format(
            term=term, community=community, context=context, options=wsd_options
        )

        if model_mode == "gemini":
            model_response = inference_gemini(content=content)
            conversation = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": model_response}
            ]
        elif model_mode == "llama":
            conversation = [
                {"role": "user", "content": content},
            ]
            model_response = agent.inference_llama(conversation)
            conversation.append({"role": "assistant", "content": model_response})
        else:
            print(f"Error: {model_mode} not supported!")
            exit(0)

        if model_response is None:
            prediction = '' # placeholder to replace empty string
        else:
            prediction = model_response.removeprefix("1. ").removeprefix("2. ").removeprefix("3. ").removeprefix("4. ")

        f1 = compute_f1(prediction, description)
        f1_list.append(f1)

        outline = {
            "community": community,
            "term": term,
            "description": description,
            "model_prediction": prediction,
            "f1": str(f1),
            "context": context,
            "conversation": conversation
        }
        result_writer.write(outline)
    
    print(f"Average F1 over {len(f1_list)} instances: {sum(f1_list) / len(f1_list)}")
    wsd_json_reader.close()
    result_writer.close()


def named_entity_recognition(agent, ner_csv, output_ner_jsonl, context_mode, model_mode, glossary_path):
    # load contexts
    post_random_1_dict = {}
    post_top_1_dict = {}
    post_top_10_dict = {}
    glossary_reader = jsonlines.open(glossary_path, mode="r")
    for line in glossary_reader:
        community = line["subreddit"]
        term = line["term"]
        key = community + "_" + term
        post_random_1_dict[key] = line["post_random_1"]
        post_top_1_dict[key] = line["post_top_1"]
        post_top_10_dict[key] = line["post_top_10"]
    glossary_reader.close()

    correct_tag = 0
    amount = 0
    output_writer = jsonlines.open(output_ner_jsonl, mode="w", flush=True)
    predictions = []
    labels = []
    with open(ner_csv, "r") as csv_file:
        ner_csv_reader = csv.DictReader(csv_file, delimiter=',')
        for line in ner_csv_reader:
            amount += 1
            community = line["subreddit"]
            term = line["term"]
            ner_tag = line["NER_TAG"]

            if context_mode == "post_random_1":
                context = post_random_1_dict[community + "_" + term]
            elif context_mode == "post_top_1":
                context = post_top_1_dict[community + "_" + term]
            elif context_mode == "post_top_10":
                context = post_top_10_dict[community + "_" + term]
            else:
                print(f"Error: unsupported context mode {context_mode}.")
                exit(0)

            if ner_tag == "":
                break
            
            content = evaluation_named_entity_recognition_wnut.format(
                term=term, community=community, context=context
            )
            if model_mode == "gemini":
                model_response = inference_gemini(content=content)
                conversation = [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": model_response}
                ]
            elif model_mode == "llama":
                conversation = [
                    {"role": "user", "content": content},
                ]
                model_response = agent.inference_llama(conversation)
                conversation.append({"role": "assistant", "content": model_response})
            else:
                print(f"Error: {model_mode} not supported!")
                exit(0)

            if model_response is None:
                model_response = ''

            if ner_tag.lower().strip() == model_response.lower().strip():
                correct_tag += 1
                correctness = True
            else:
                correctness = False
            
            predictions.append(model_response.lower().strip())
            labels.append(ner_tag.lower().strip())
            
            outline = {
                "community": community,
                "term": term,
                "ner_tag": ner_tag,
                "ner_tag (model_generated)": model_response,
                "acc": str(int(correctness)),
                "context": context
            }
            output_writer.write(outline)
    output_writer.close()
    print(f"NER Tagging Sentence Amount: {amount}")
    print(f"NER Tagging Acc: {correct_tag / amount * 100:.2f}%")


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # parse arguments
    parser = argparse.ArgumentParser(prog='evaluate.py')
    parser.add_argument('--task', type=str, help="Task type: all/explain/wsd/ner")
    parser.add_argument('--result_dir', type=str, help="Task type: all/explain/wsd/ner")
    parser.add_argument('--model_prefix', type=str, help="Model prefix.")
    parser.add_argument('--model_type', type=str, help="Model type: gemini/llama")
    parser.add_argument('--model_id', type=str, help="Model ID.")
    parser.add_argument('--lora_dir', type=str, help="LoRA dir.")
    parser.add_argument('--context_mode', type=str, help="Context mode: post_random_1/post_top_1/post_top_10")
    parser.add_argument('--icl_mode', type=str, help="ICL mode: zero_shot/one_shot/three_shot")
    parser.add_argument('--seed', type=int, help="Set seed.")
    parser.add_argument('--eval_set', type=str, help="Eval set: valid/test")
    
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    model_prefix = args.model_prefix
    model_id = args.model_id
    lora_dir = args.lora_dir
    context_mode = args.context_mode
    icl_mode = args.icl_mode

    if args.model_type == "gemini":
        agent = None
    elif args.model_type == "llama":
        cache = "/tmp"
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            clean_up_tokenization_spaces=False
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache
        )
        if lora_dir != "":
            fine_tuned_model = PeftModel.from_pretrained(base_model, lora_dir)
            agent = Agent(tokenizer, fine_tuned_model)
            print(fine_tuned_model)
        else:
            agent = Agent(tokenizer, base_model)
            print(base_model)
    else:
        print(f"Error: unsupported model {args.model}.")
        exit(0)

    glossary_path = f"{EXPERIMENTAL_DATA}/reddit_{args.eval_set}_set_w_vt_em_con3_contexts_10.jsonl"
    if args.task in ["explain", "all"]:
        result_jsonl = f"{args.result_dir}/explanation.{args.eval_set}.{model_prefix}.{context_mode}.{icl_mode}.jsonl"
        explanation(
            agent=agent, 
            glossary_path=glossary_path, 
            result_jsonl=result_jsonl, 
            context_mode=context_mode, 
            icl_mode=icl_mode,
            model_mode=args.model_type
        )
    if args.task in ["wsd", "all"]:
        wsd_jsonl = f"{EXPERIMENTAL_DATA}/wsd.{args.eval_set}_w_vt_em_con3_contexts_10.jsonl"
        output_wsd_jsonl = f"{args.result_dir}/wsd.{args.eval_set}.{model_prefix}.{context_mode}.jsonl"
        word_sense_diambiguation(
            agent=agent, 
            wsd_jsonl=wsd_jsonl, 
            result_jsonl=output_wsd_jsonl,
            context_mode=context_mode,
            model_mode=args.model_type
        )
    if args.eval_set == "valid" and args.task in ["ner", "all"]:
        ner_csv = f"{EXPERIMENTAL_DATA}/NER_wnut.csv"
        output_ner_jsonl = f"{args.result_dir}/ner_wnut.{model_prefix}.{context_mode}.jsonl"
        named_entity_recognition(
            agent=agent, 
            ner_csv=ner_csv, 
            output_ner_jsonl=output_ner_jsonl, 
            context_mode=context_mode,
            model_mode=args.model_type,
            glossary_path=glossary_path
        )