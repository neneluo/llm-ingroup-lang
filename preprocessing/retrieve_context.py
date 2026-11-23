from tqdm import tqdm
import jsonlines
import csv
import re
from sentence_transformers import CrossEncoder
import random
from datasets import concatenate_datasets
import hashlib
from preprocessing.utils import split_dataset
from preprocessing.path import *

def check_definition_exact_match(definition, text):
    if definition.lower() in text.lower():
        return True
    else:
        return False

def check_definition_window_match(definition, text, sliding_window):
    # check whether there is a consecutive word match of sliding_window size
    definition_words = re.findall(r'\b[\w.\'-]+\b', definition.lower())
    text_words = re.findall(r'\b[\w.\'-]+\b', text.lower())

    if len(definition_words) < sliding_window or len(text_words) < sliding_window:
        return False

    definition_words_set = set()
    for i in range(len(definition_words)-sliding_window+1):
        definition_words_set.add(tuple(definition_words[i:i+sliding_window]))

    for i in range(len(text_words)-sliding_window+1):
        word_combination = tuple(text_words[i:i+sliding_window])
        if word_combination in definition_words_set:
            return True
    return False

def rank_definition_context(definition, context_list, model, context_number):
    ranks = model.rank(definition, context_list, return_documents=True)
    ranked_context = []
    for i in range(len(context_list)):
        text = ranks[i]['text']
        ranked_context.append(text)
        if len(ranked_context) == context_number:
            break
    return ranked_context

def retrieve_reddit_data(input_csv, output_jsonl, model, context_number):
    # glossary: lexicon, community, definition
    subreddit_term_definition = {}
    subreddit_term_context = {}
    with open(input_csv, "r") as reader:
        csv_reader = csv.DictReader(reader)
        for line in csv_reader:
            subreddit = line["subreddit"].lower()
            term = line["term"].lower()
            description = line["description"]
            subreddit_term_definition.setdefault(subreddit, {})[term] = description
            subreddit_term_context.setdefault(subreddit, {})[term] = []

    print("Info:")
    for key, value in subreddit_term_definition.items():
        print(key, len(value))

    # select a subreddit
    train_set, valid_set, test_set = split_dataset()
    retrived_dataset = concatenate_datasets([valid_set, test_set])

    # retrieve duplicate posts
    dup_hashes = set()
    with open(f"{EXPERIMENTAL_DATA}/duplicate_reddit_post_hash.txt", "r") as dup_hash_reader:
        for line in dup_hash_reader:
            dup_hashes.add(line.strip())
    print(f"Duplicate post amount: {len(dup_hashes)}")

    # glossary: term, subreddit, description, context, seperate by "\t"
    # subreddit_term_definition: {subreddit: {term: description}}
    # subreddit_term_context: {subreddit: {term: [context1, context2, ...]}}
    json_writer = jsonlines.open(output_jsonl, mode="w", flush=True)
    for item in tqdm(retrived_dataset):
        sentence, subreddit = item['body'].lower().strip(), item['subreddit'].lower().strip()

        # skip if this post is duplicate
        text_hash = hashlib.md5(sentence.encode("utf-8")).hexdigest()
        if text_hash in dup_hashes:
            continue

        words = re.findall(r'\b[\w.\'-]+\b', sentence)
        if subreddit not in subreddit_term_definition:
            continue
        
        for term in subreddit_term_definition[subreddit].keys():
            if term in words:
                subreddit_term_context[subreddit][term].append(sentence)

    for subreddit, term_dict in subreddit_term_context.items():
        for term, context_list in term_dict.items():
            definition = subreddit_term_definition[subreddit][term]

            filtered_context_list = []
            # remove match context
            for text in context_list:
                em_match = check_definition_exact_match(definition=definition, text=text)
                con_match = check_definition_window_match(definition=definition, text=text, sliding_window=3)
                if em_match or con_match:
                    continue
                filtered_context_list.append(text)

            if len(filtered_context_list) < context_number:
                continue
            
            ranked_context_list = rank_definition_context(
                definition=definition,
                context_list=filtered_context_list,
                model=model,
                context_number=context_number
            )

            # convert to string
            post_top_10 = "\n".join(f"Post {i}: {content}" for i, content in enumerate(ranked_context_list))
            post_top_1 = ranked_context_list[0]
            post_random_1 = random.choice(context_list)
            
            outline = {
                "term": term,
                "subreddit": subreddit,
                "definition": subreddit_term_definition[subreddit][term],
                "post_random_1": post_random_1,
                "post_top_1": post_top_1,
                "post_top_10": post_top_10
            }
            json_writer.write(outline)
    
    json_writer.close()

if __name__ == "__main__":
    context_number = 10
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    input_csv = "data/lucy_glossary/valid.csv"
    output_jsonl = f"{EXPERIMENTAL_DATA}/reddit_valid_set_w_vt_em_con3_contexts_10.jsonl"
    retrieve_reddit_data(
        input_csv=input_csv, 
        output_jsonl=output_jsonl, 
        model=model, 
        context_number=context_number
    )

    input_csv = "data/lucy_glossary/test.csv"
    output_jsonl = f"{EXPERIMENTAL_DATA}/reddit_test_set_w_vt_em_con3_contexts_10.jsonl"
    retrieve_reddit_data(
        input_csv=input_csv, 
        output_jsonl=output_jsonl, 
        model=model, 
        context_number=context_number
    )

    input_csv = "data/lucy_glossary/hold_out.csv"
    output_jsonl = f"{EXPERIMENTAL_DATA}/reddit_hold_out_set_w_vt_em_con3_contexts_10.jsonl"
    retrieve_reddit_data(
        input_csv=input_csv, 
        output_jsonl=output_jsonl, 
        model=model, 
        context_number=context_number
    )
