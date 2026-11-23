from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from datasets import load_dataset
import torch
import argparse
import jsonlines
import random
import string
from preprocessing.path import *
from models.prompts import *

def generate_random_string(length):
    characters = string.ascii_letters
    return ''.join(random.choices(characters, k=length))

def construct_sft_dataset(input_jsonl, output_jsonl, mask, data_amount):
    # Conversational language modeling
    # {"messages": [{"role": "user", "content": "What color is the sky?"},
    #               {"role": "assistant", "content": "It is blue."}]}
    similarity_t = 0.7
    input_jsonl_reader = jsonlines.open(input_jsonl, mode="r")
    output_jsonl_writer = jsonlines.open(output_jsonl, mode="w", flush=True)
    amount = 0
    for line in input_jsonl_reader:
        gen_outcome = bool(line["gen_outcome"])
        if not gen_outcome:
            continue
        
        word = line["word"]
        word_guess = line["word_guess"]
        word_inno = line["word_innovation"]
        sentence_inno = line["sentence_innovation"]
        sentence = line["sentence"]
        community = line["community"]
        similarity = float(line["word_similarity"])

        if mask in ["True", "true"]:
            word_mask = generate_random_string(5)
            if gen_outcome and similarity > similarity_t:
                outline = {
                    "messages": [
                        {"role": "user", "content": training_prompt_synonym.format(
                            term=word_mask,
                            community=community,
                            context=sentence_inno.replace(word_inno, word_mask),
                        )},
                        {"role": "assistant", "content": f"{word}"}
                    ]
                }
                output_jsonl_writer.write(outline)
                amount += 1
        else:
            if gen_outcome and similarity > similarity_t:
                outline = {
                    "messages": [
                        {"role": "user", "content": training_prompt_synonym.format(
                            term=word_inno,
                            community=community,
                            context=sentence_inno,
                        )},
                        {"role": "assistant", "content": f"{word}"}
                    ]
                }
                output_jsonl_writer.write(outline)
                amount += 1
        
        if amount == data_amount:
            break

    input_jsonl_reader.close()
    output_jsonl_writer.close()


def train(model_id, output_dir, training_json):
    # codes adapted from: https://github.com/huggingface/trl/blob/main/examples/notebooks/sft_trl_lora_qlora.ipynb
    num_train_epochs=3
    batch_size=1
    gradient_accumulation_steps=16
    cache = "/tmp"

    train_dataset = load_dataset("json", data_files=training_json, split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.chat_template = r"""
    {% for message in messages %}
    {% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
    {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
    {% if message['role'] == 'assistant' %}
    {% generation %}{{ message['content'] | trim }}{% endgeneration %}{{ '<|eot_id|>' }}
    {% else %}
    {{ message['content'] | trim }}{{ '<|eot_id|>' }}
    {% endif %}
    {% endfor %}
    {% if add_generation_prompt %}
    {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
    {% endif %}
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        dtype=torch.bfloat16, # float16
        device_map="auto",
        cache_dir=cache,
        use_cache=False
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    )   
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        optim="adamw_torch",
        learning_rate=1e-5,
        max_length=128000,
        bf16=True,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        assistant_only_loss=True,
        report_to="wandb"
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(prog='instruction_ft.py')
    # data creation related args
    parser.add_argument('--input_simulation_data', type=str, help="Input: simulation data jsonl path.")
    parser.add_argument('--output_training_data', type=str, help="Output: training data jsonl path.")
    parser.add_argument('--special_token_mask', type=str, help="Whether to replace lexical innovation into a special token: [MASK].")
    parser.add_argument('--data_amount', type=int, help="Data amount.")

    # model training related args
    parser.add_argument('--model_id', type=str, help="Model ID")
    parser.add_argument('--output_dir', type=str, help="Model output dir.")
    parser.add_argument('--seed', type=int, help="Set seed.")
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    construct_sft_dataset(
        input_jsonl=args.input_simulation_data, 
        output_jsonl=args.output_training_data,
        mask=args.special_token_mask,
        data_amount=args.data_amount
    )

    train(
        model_id=args.model_id, 
        output_dir=args.output_dir, 
        training_json=args.output_training_data
    )
