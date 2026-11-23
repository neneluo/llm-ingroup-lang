from datasets import load_dataset
import hashlib
from tqdm import tqdm
import os
from preprocessing.path import *


def gen_duplicate_hash():
    dataset = load_dataset(
    'sentence-transformers/reddit-title-body', 
    split='train'
    )

    seen_hashes = set()
    duplicate_hashes = set()
    duplicate_content_amount = 0

    for item in tqdm(dataset):
        post = item["body"].lower().strip()
        text_hash = hashlib.md5(post.encode("utf-8")).hexdigest()
        
        if text_hash in seen_hashes:
            duplicate_hashes.add(text_hash)
            duplicate_content_amount += 1
        else:
            seen_hashes.add(text_hash)

    with open(f"{EXPERIMENTAL_DATA}/duplicate_reddit_post_hash.txt", "w") as dup_hash_writer:
        for hash in duplicate_hashes:
            dup_hash_writer.write(hash + "\n")

    print(f"Total content amount: {len(dataset)}")
    print(f"Duplicate content amount: {duplicate_content_amount}")

if __name__ == "__main__":
    os.makedirs(EXPERIMENTAL_DATA, exist_ok=True)

    gen_duplicate_hash()
