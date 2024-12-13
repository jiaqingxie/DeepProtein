from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration

from DeepProtein.instruction import get_example
import torch
import re
class BioMistral():
    def __init__(self, dataset_name):
        super(BioMistral, self).__init__()
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B",  add_bos_token=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B",
                                                          quantization_config=self.bnb_config,
                                                          device_map="auto",trust_remote_code=True)
        self.instruction, self.aim = get_example(dataset_name)
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)
    def inference(self, data):
        ans = []
        for _data in data:
            inputs = f"{self.instruction} What is the {self.aim} of the given protein sequence {_data}?"
            inputs = self.tokenizer(inputs, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0,
                    top_p=1,
                    eos_token_id=self.newline_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            num = float(extract_num(response))
            ans.append(num)
        ans_tensor = torch.tensor(ans).unsqueeze(0).T
        return ans_tensor




class BioT5_plus():
    def __init__(self, dataset_name):
        super(BioT5_plus, self).__init__()
        self.instruction, self.aim = get_example(dataset_name)
        self.tokenizer = T5Tokenizer.from_pretrained("QizhiPei/biot5-plus-base-chebi20",
                                                     model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-plus-base-chebi20')
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)

    def inference(self, data):
        ans = []
        for _data in data:
            inputs = f"{self.instruction} What is the {self.aim} of the given protein sequence {_data}?"
            inputs = self.tokenizer(inputs, return_tensors="pt").to("cuda").input_ids

            generation_config = self.model.generation_config
            generation_config.max_length = 64
            generation_config.num_beams = 1

            outputs = self.model.generate(inputs, generation_config=generation_config)
            output_selfies = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(' ', '')
            num = float(extract_num(output_selfies ))
            ans.append(num)
        ans_tensor = torch.tensor(ans).unsqueeze(0).T
        return ans_tensor

class ChemLLM_7B():
    def __init__(self, dataset_name):
        super(ChemLLM_7B, self).__init__()
        self.instruction, self.aim = get_example(dataset_name)
        self.tokenizer = T5Tokenizer.from_pretrained("QizhiPei/biot5-plus-base-chebi20",
                                                     model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-plus-base-chebi20')
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)

    def InternLM2_format(self, instruction, prompt):
        prefix_template = [
            "<|im_start|>system\n",
            "{}",
            "<|im_end|>\n"
        ]
        prompt_template = [
            "<|im_start|>user\n",
            "{}",
            "<|im_end|>\n"
            "<|im_start|>assistant\n",
            "{}",
            "<|im_end|>\n"
        ]
        system = f'{prefix_template[0]}{prefix_template[1].format(instruction)}{prefix_template[2]}'
        # history = "".join([f'{prompt_template[0]}{prompt_template[1].format(qa[0])}{prompt_template[2]}{prompt_template[3]}{prompt_template[4].format(qa[1])}{prompt_template[5]}' for qa in history])
        prompt = f'{prompt_template[0]}{prompt_template[1].format(prompt)}{prompt_template[2]}{prompt_template[3]}'
        return f"{system}{prompt}"

    def generate_response(self, model, tokenizer, instruction, prompt, history=None, max_tokens=200):
        """
        Generates a response from the model using a formatted template.
        """
        # Format the input
        formatted_prompt = self.InternLM2_format(instruction, prompt)

        # Encode the input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0,
                top_p=1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and return the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def inference(self, data):
        ans = []
        for _data in data:
            prompt = f"What is the {self.aim} of the given protein sequence {_data}?"
            prompt = self.tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
            response = self.generate_response(
                self.model, self.tokenizer,
                instruction=self.instruction,
                prompt=prompt
            )
            num = float(extract_num(response ))
            ans.append(num)
        ans_tensor = torch.tensor(ans).unsqueeze(0).T
        return ans_tensor

def extract_num(input_string):

    numbers = re.findall(r'\d+', input_string)
    return numbers[-1] if numbers else 0
