import random
import nltk
from nltk.corpus import stopwords
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import torch.nn.functional as F 
import numpy as np
from collections import Counter
from preprocessing.utils import tokenize
from preprocessing.path import *

def filter_stop_words(words):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
    return filtered_words

def compute_words_npmi(community, words, stop_words):
    npmi_community = {}
    with open(f"{WORD_NPMI}/subreddit_npmi_{community}.txt", "r") as reader:
        for line in reader:
            lexicon, npmi = line.split("\t")
            npmi_community[lexicon] = float(npmi)
    
    npmi_candidates = {}
    for word in words:
        if word in stop_words:
            continue
        npmi_candidates[word] = npmi_community[word]
    return npmi_candidates

def word_sampling(text, community, mode, tokenizer, model):
    if mode == "random":
        tokens = tokenize(text)
        word = random.choice(tokens)
    elif mode == "npmi":
        words = tokenize(text) 
        stop_words = set(stopwords.words('english'))
        words_npmi = compute_words_npmi(
            community=community, 
            words=words,
            stop_words=stop_words
        )
        words_npmi_sorted = dict(sorted(words_npmi.items(), key=lambda item: item[1], reverse=True))
        buffer = 4 if len(words_npmi_sorted.keys()) > 4 else len(words_npmi_sorted.keys())
        idx = random.randint(0, buffer)
        word = list(words_npmi_sorted.keys())[idx]
    else:
        print("Error: sampling mode {m} not supported".format(m=mode))
    return word

def sampling_test():
    set_seed(42)
    text = "I bought an iPhone last week so I can cook."
    text = text.lower()
    community = "apple"

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    cache = "/tmp"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache
    )

    sampled_word_random = word_sampling(
        text=text, 
        community=community, 
        mode="random",
        tokenizer=tokenizer,
        model=model
    )
    print("sampled_word_random:", sampled_word_random)

    sampled_word_npmi = word_sampling(
        text=text, 
        community=community, 
        mode="npmi",
        tokenizer=tokenizer,
        model=model
    )
    print("sampled_word_npmi:", sampled_word_npmi)

if __name__ == "__main__":
    sampling_test()