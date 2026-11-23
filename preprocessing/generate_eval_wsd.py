import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import csv
import random
import jsonlines
from preprocessing.path import *

def retrieve_word_senses(word):
    definitions = []
    syns = wn.synsets(word)
    if len(syns) is None:
        return []
    
    for sense in syns:
        definitions.append(sense.definition())
    return definitions

def create_wsd_testsets(input_json, input_csv, output_json):
    random.seed(42)
    # glossary: term, community, definition
    subreddit_definition = {}
    with open(input_csv, "r") as reader:
        csv_reader = csv.DictReader(reader)
        for line in csv_reader:
            subreddit = line["subreddit"].lower()
            term = line["term"].lower()
            description = line["description"].lower()
            subreddit_definition.setdefault(subreddit, []).append(description)
    print("Info:")
    for key, value in subreddit_definition.items():
        print(key, len(value))

    input_json_reader = jsonlines.open(input_json, mode="r")
    wsd_writer = jsonlines.open(output_json, mode="w", flush=True)

    for line in input_json_reader:
        subreddit = line["subreddit"].lower()
        term = line["term"].lower()
        description = line["definition"].lower()
        post_random_1 = line["post_random_1"]
        post_top_1 = line["post_top_1"]
        post_top_10 = line["post_top_10"]

        options = [description]

        word_senses = retrieve_word_senses(term)
        if len(word_senses) >= 1:
            options.append(random.choice(word_senses))

        sense_same_subreddit = list(subreddit_definition[subreddit])
        sense_same_subreddit.remove(description)
        if len(sense_same_subreddit) >= 1:
            options.append(random.choice(sense_same_subreddit))

        subreddit_list = list(subreddit_definition.keys())
        subreddit_list.remove(subreddit)
        random_subreddit = random.choice(subreddit_list)
        sense_other_subreddits = list(subreddit_definition[random_subreddit])
        if len(sense_other_subreddits) >= 1:
            options.append(random.choice(sense_other_subreddits))
        subreddit_list.remove(random_subreddit)
        
        if len(options) != 4:
            random_subreddit = random.choice(subreddit_list)
            sense_other_subreddits = list(subreddit_definition[random_subreddit])
            if len(sense_other_subreddits) >= 1:
                options.append(random.choice(sense_other_subreddits))
            assert len(options) == 4

        wsd_options = ""
        random.shuffle(options)
        for i, option in enumerate(options):
            wsd_options += f"{i+1}. {option}\n"
            
        outline = {
            "term": term,
            "subreddit": subreddit,
            "description": description,
            "wsd_options": wsd_options,
            "post_random_1": post_random_1,
            "post_top_1": post_top_1,
            "post_top_10": post_top_10,
        }
        wsd_writer.write(outline)

    input_json_reader.close()
    wsd_writer.close()

if __name__ == "__main__":
    input_json = f"{EXPERIMENTAL_DATA}/reddit_valid_set_w_vt_em_con3_contexts_10.jsonl"
    input_csv = "data/lucy_glossary/valid.csv"
    output_json = f"{EXPERIMENTAL_DATA}/wsd.valid_w_vt_em_con3_contexts_10.jsonl"
    create_wsd_testsets(input_json=input_json, input_csv=input_csv, output_json=output_json)

    input_json = f"{EXPERIMENTAL_DATA}/reddit_test_set_w_vt_em_con3_contexts_10.jsonl"
    input_csv = "data/lucy_glossary/test.csv"
    output_json = f"{EXPERIMENTAL_DATA}/wsd.test_w_vt_em_con3_contexts_10.jsonl"
    create_wsd_testsets(input_json=input_json, input_csv=input_csv, output_json=output_json)

    input_json = f"{EXPERIMENTAL_DATA}/reddit_hold_out_set_w_vt_em_con3_contexts_10.jsonl"
    input_csv = "data/lucy_glossary/hold_out.csv"
    output_json = f"{EXPERIMENTAL_DATA}/wsd.hold_out_w_vt_em_con3_contexts_10.jsonl"
    create_wsd_testsets(input_json=input_json, input_csv=input_csv, output_json=output_json)
