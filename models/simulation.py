import re
import os
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from gensim.models.fasttext import load_facebook_model
from tqdm import tqdm
import hashlib
import argparse
from preprocessing.utils import split_dataset
from preprocessing.path import *
from models.sampling import word_sampling
from models.prompts import *
from models.agent import Agent


class Simulation:
    def __init__(self, agent_generator, agent_guesser):
        self.agent_generator = agent_generator
        self.agent_guesser = agent_guesser

    def prompt_engineering_generator(self, sentence, word, community):
        conversation = [
            {"role": "system", "content": system_prompt_ingroup_generator.format(community=community)},
            {"role": "user", "content": user_prompt.format(post=sentence, word=word)},
        ]
        return conversation
    
    def prompt_engineering_guesser(self, sentence_inno, word_inno, community):
        conversation = [
            {"role": "system", "content": system_prompt_ingroup_guesser.format(community=community)},
            {"role": "user", "content": user_prompt.format(post=sentence_inno, word=word_inno)},
        ]
        return conversation

    def extract_answer(self, response):
        try:
            answer = response.split("Answer:")[1].strip()
        except Exception as e:
            # print(e)
            answer = ""
        return answer

    def run_single_turn(self, mode, simulation_data, model_type, data_start, data_end, special_token_mask=False):
        train_set, valid_set, test_set = split_dataset()

        # retrieve duplicate posts
        dup_hashes = set()
        with open(f"{EXPERIMENTAL_DATA}/duplicate_reddit_post_hash.txt", "r") as dup_hash_reader:
            for line in dup_hash_reader:
                dup_hashes.add(line.strip())
        print(f"Duplicate post amount: {len(dup_hashes)}")

        amount = 0
        valid_amount = data_start

        results_writer = jsonlines.open(simulation_data, mode="w", flush=True)

        # load models for calculating cosine similarity
        fasttext_model = load_facebook_model(f"{EXPERIMENTAL_DATA}/cc.en.300.bin")
        
        for item in tqdm(train_set, total=data_end):
            amount += 1
            if amount < data_start:
                continue
            if valid_amount > data_end:
                break

            gen_succ = False
            com_succ = False
            sentence, community = item['body'].lower().strip(), item['subreddit'].lower().strip()
            valid_amount += 1

            # skip if this post is duplicate
            text_hash = hashlib.md5(sentence.encode("utf-8")).hexdigest()
            if text_hash in dup_hashes:
                continue

            try:
                word = word_sampling(
                    text=sentence, 
                    community=community, 
                    mode=mode,
                    tokenizer=self.agent_generator.tokenizer,
                    model=self.agent_generator.model
                )
            except Exception as e:
                print("\n======\nError: sampling error in mode ", mode)
                print("\nSentence:", sentence)
                continue

            conversation_generator = self.prompt_engineering_generator(sentence, word, community)
            if model_type == "llama":
                response_generator = self.agent_generator.inference_llama(conversation_generator)
            else:
                print(f"Error: {model_type} not supported!")
                exit(0)
            conversation_generator.append({"role": "assistant", "content": response_generator})
            
            word_inno = self.extract_answer(response_generator)
            if (word_inno != "") and (word_inno.lower() != word.lower()):
                if special_token_mask in ["True", "true"]:
                    word_inno = "[MASK]"
                sentence_inno = re.sub(r'\b' + word + r'\b', word_inno, sentence)
                conversation_guesser = self.prompt_engineering_guesser(sentence_inno, word_inno, community)
                if model_type == "llama":
                    response_guesser = self.agent_guesser.inference_llama(conversation_guesser)
                else:
                    print(f"Error: {model_type} not supported!")
                    exit(0)
                conversation_guesser.append({"role": "assistant", "content": response_guesser})
                word_guess = self.extract_answer(response_guesser)

                # compare the original word and guessed word
                gen_succ = True
                com_succ = (word.lower() == word_guess.lower())
                similarity = float(fasttext_model.wv.similarity(word.lower(), word_guess.lower()))
            else:
                word_guess = ""
                sentence_inno = ""
                conversation_guesser = []
                similarity = 0

            outline = {
                "simulation_round": amount, 
                "gen_outcome": int(gen_succ),
                "com_outcome_em": int(com_succ),
                "word_similarity": similarity,
                "community": community,
                "sentence": sentence,
                "word": word,
                "word_innovation": word_inno,
                "word_guess": word_guess,
                "sentence_innovation": sentence_inno,
                "conversation_generator": conversation_generator,
                "conversation_guesser": conversation_guesser
            }
            results_writer.write(outline)
        results_writer.close()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # parse arguments
    parser = argparse.ArgumentParser(prog='simulation.py')
    parser.add_argument('--output_simulation_data', type=str, help="Simulation data path.")
    parser.add_argument('--model_id', type=str, help="Model ID")
    parser.add_argument('--model_type', type=str, help="Model type: llama")
    parser.add_argument('--data_start', type=int, help="Data start number.")
    parser.add_argument('--data_end', type=int, help="Data end number.")
    parser.add_argument('--sampling_strategies', type=str, help="Sampling strategies: random/npmi")
    parser.add_argument('--special_token_mask', type=str, help="Whether to replace lexical innovation into a special token: [MASK].")
    parser.add_argument('--seed', type=int, help="Set seed.")
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        clean_up_tokenization_spaces=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        cache_dir="/tmp"
    )
    agent_generator = Agent(tokenizer, model)
    agent_guesser = Agent(tokenizer, model)

    simu = Simulation(agent_generator, agent_guesser)
    simu.run_single_turn(
        simulation_data=args.output_simulation_data, 
        data_start=args.data_start,
        data_end=args.data_end, 
        mode=args.sampling_strategies, 
        model_type=args.model_type,
        special_token_mask=args.special_token_mask
    )
