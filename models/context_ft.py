from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
from datasets import load_dataset
import torch
import jsonlines
import argparse
import hashlib
from preprocessing.utils import split_dataset
from preprocessing.path import *

def construct_sft_dataset(output_jsonl, amount):
    num = 1
    train_set, valid_set, test_set = split_dataset()

    # retrieve duplicate posts
    dup_hashes = set()
    with open(f"{EXPERIMENTAL_DATA}/duplicate_reddit_post_hash.txt", "r") as dup_hash_reader:
        for line in dup_hash_reader:
            dup_hashes.add(line.strip())
    print(f"Duplicate post amount: {len(dup_hashes)}")
    
    output_jsonl_writer = jsonlines.open(output_jsonl, mode="w", flush=True)
    for outline in train_set:

        sentence, community = outline['body'].lower().strip(), outline['subreddit'].lower().strip()
        # skip if this post is duplicate
        text_hash = hashlib.md5(sentence.encode("utf-8")).hexdigest()
        if text_hash in dup_hashes:
            continue

        output_jsonl_writer.write(outline)
        num += 1
        if num > amount:
            break
    output_jsonl_writer.close()

def train(training_jsonl, output_dir, model_id, cache):
    def tokenize(batch):
        return tokenizer(
            batch["body"],
            truncation=True,
            max_length=128000,
        )
    
    num_train_epochs = 3
    batch_size = 1
    gradient_accumulation_steps = 16

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]
    )

    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache,
            use_cache=False
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=training_jsonl, split="train")
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        optim="adamw_torch",
        learning_rate=1e-5,
        bf16=True,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(prog='ft.py')
    # data creation related args
    parser.add_argument('--output_training_data', type=str, help="Output: training data jsonl path.")
    parser.add_argument('--data_amount', type=int, help="Data amount.")

    # model training related args
    parser.add_argument('--model_id', type=str, help="Model ID")
    parser.add_argument('--output_dir', type=str, help="Model output dir.")
    parser.add_argument('--seed', type=int, help="Set seed.")
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    construct_sft_dataset(
        output_jsonl=args.output_training_data,
        amount=args.data_amount
    )
    train(
        output_dir=args.output_dir,
        model_id=args.model_id,
        training_jsonl=args.output_training_data,
        cache="/tmp"
    )
