from google import genai
import time
from models.api import GKEY

class Agent:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def inference_llama(self, conversation):
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output = outputs[0][input_ids['input_ids'].shape[-1]:]
        response = self.tokenizer.decode(output, skip_special_tokens=True)
        return response

    def inference_qwen(self, conversation, thinking=False):
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer(
            [text], return_tensors="pt"
        ).to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        if thinking:
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return response

def inference_gemini(content, retry=3):
    client = genai.Client(api_key=GKEY)
    
    for attempt in range(retry):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview", contents=content
            )
            return response.text
        except:
            print(f"Warning: gemini inference error, retrying for the {attempt+1} time... ")
            time.sleep(3)
