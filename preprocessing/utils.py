import re
from datasets import load_dataset

def tokenize(sentence):
    tokens = re.findall(r'\b[\w.\'-]+\b', sentence)
    return tokens

def split_dataset():
    dataset = load_dataset(
        'sentence-transformers/reddit-title-body', 
        split='train'
    )

    train_test_split = dataset.train_test_split(shuffle=True, seed=42, test_size=0.5)
    val_test_split = train_test_split["test"].train_test_split(shuffle=True, seed=42, test_size=0.5)

    train_set = train_test_split["train"]
    valid_set = val_test_split["train"]
    test_set = val_test_split["test"]
    return train_set, valid_set, test_set
