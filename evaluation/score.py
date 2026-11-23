from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import jsonlines
from models.prompts import *
from models.agent import Agent
import sys

def comparing_definition_llm(input_json, output_json):
    model_id = "Qwen/Qwen3-14B"
    cache = "/tmp"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto", 
        cache_dir=cache
    )
    agent = Agent(tokenizer, base_model)

    input_json_reader = jsonlines.open(input_json, mode="r")
    output_json_writer = jsonlines.open(output_json, mode="w", flush=True)
    
    good = 0
    bad = 0
    borderline = 0
    amount = 0
    for input_line in input_json_reader:
        term=input_line["term"]
        community=input_line["community"]
        definition=input_line["description"]
        model_generation=input_line["model_generation"]
        
        conversation = [
            {"role": "user", "content": evaluation_free_explanation_llm_judge.format(
                term=term,
                description=definition,
                model_generation=model_generation,
            )},
        ]

        label = agent.inference_qwen(conversation)

        if label.strip().lower() == "good":
            good += 1
        if label.strip().lower() == "bad":
            bad += 1
        if label.strip().lower() == "borderline":
            borderline += 1
        
        amount += 1
        outline = {
            "community": community,
            "term": term,
            "definition": definition,
            "model_generation": model_generation,
            "label": label
        }
        output_json_writer.write(outline)
    input_json_reader.close()
    output_json_writer.close()
    print(f"LLM-as-a-judge Evaluation: {output_json}")
    print(f"Good: {good/amount*100:.2f}")
    print(f"Bad: {bad/amount*100:.2f}")
    print(f"Borderline: {borderline/amount*100:.2f}")


if __name__ == "__main__":

    _, input_json, output_json, seed = sys.argv
    set_seed(int(seed))
    comparing_definition_llm(input_json, output_json)
