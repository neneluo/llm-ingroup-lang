import os
import re
from collections import Counter
from tqdm import tqdm
import math
import hashlib
from preprocessing.path import *
from preprocessing.utils import tokenize, split_dataset

def compute_subreddit_freq():
    # retrieve duplicate posts
    dup_hashes = set()
    with open(f"{EXPERIMENTAL_DATA}/duplicate_reddit_post_hash.txt", "r") as dup_hash_reader:
        for line in dup_hash_reader:
            dup_hashes.add(line.strip())
    print(f"Duplicate post amount: {len(dup_hashes)}")

    subreddit_freq = {}
    total_subreddit = Counter()
    train_set, valid_set, test_set = split_dataset()
    
    for item in tqdm(train_set):
        sentence, community = item['body'].lower().strip(), item['subreddit'].lower().strip()

        # skip if this post is duplicate
        text_hash = hashlib.md5(sentence.encode("utf-8")).hexdigest()
        if text_hash in dup_hashes:
            continue

        tokens = tokenize(sentence)
        if community in subreddit_freq:
            subreddit_freq[community].update(tokens)
        else:
            subreddit_freq[community] = Counter(tokens)
        total_subreddit.update(tokens)

    for community in tqdm(subreddit_freq.keys()):
        with open(f"{WORD_STAT}/subreddit_freq_{community}.txt", "w") as writer:
            for word, count in subreddit_freq[community].most_common():
                writer.write(word + "\t" + str(count) + "\n")
    
    with open(f"{WORD_STAT}/general_freq_subreddit.txt", "w") as writer:
        for word, count in total_subreddit.most_common():
            writer.write(word + "\t" + str(count) + "\n")

def compute_npmi_by_community(community, general_freq):
    subreddit_freq = {}
    npmi_community = {}

    with open(f"{WORD_STAT}/subreddit_freq_{community}.txt", "r") as reader:
        for line in reader:
            word, freq = line.split("\t")
            subreddit_freq[word] = int(freq)

    word_count_general = float(sum(general_freq.values()))
    word_count_subreddit = float(sum(subreddit_freq.values()))

    # compute npmi of the given community
    for word in subreddit_freq.keys():
        p_w_given_c = subreddit_freq[word] / word_count_subreddit
        p_w = general_freq[word] / word_count_general
        pmi_w_c = math.log(p_w_given_c / p_w)
        p_w_c = subreddit_freq[word] / word_count_general
        npmi = - pmi_w_c / math.log(p_w_c)
        npmi_community[word] = npmi

    with open(f"{WORD_NPMI}/subreddit_npmi_{community}.txt", "w") as writer:
        for word, npmi in sorted(npmi_community.items(), key=lambda item: item[1], reverse=True):
            npmi = npmi_community[word]
            writer.write(word + "\t" + str(npmi) + "\n")

def compute_npmi():
    general_freq = {}
    with open(f"{WORD_STAT}/general_freq_subreddit.txt", "r") as reader:
        for line in reader:
            word, freq = line.split("\t")
            general_freq[word] = int(freq)

    subreddit_list = []
    for filename in sorted(os.listdir(WORD_STAT)):
        is_subreddit_freq = re.fullmatch(r'subreddit\_freq\_\S*\.txt', filename)
        if is_subreddit_freq is not None:
            subreddit = ".".join(filename.split("subreddit_freq_")[1].split('.')[:-1])
            subreddit_list.append(subreddit)

    print("length of subreddit_list", len(subreddit_list))
    for subreddit in tqdm(subreddit_list): 
        compute_npmi_by_community(community=subreddit, general_freq=general_freq)

if __name__ == "__main__":
    os.makedirs(WORD_STAT, exist_ok=True)
    os.makedirs(WORD_NPMI, exist_ok=True)

    compute_subreddit_freq()
    compute_npmi()
