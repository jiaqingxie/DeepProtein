from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from DeepProtein.instruction import get_example
import torch
import re
from tqdm import tqdm
import math, sys
import rdkit.Chem as Chem

Regression = ["fluorescence", "stability", "beta", "ppi_affinity", "tap", "sabdab_chen", "crispr"]
class BioMistral():
    def __init__(self, dataset_name):
        super(BioMistral, self).__init__()
        self.dataset_name = dataset_name
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B",  add_bos_token=True, trust_remote_code=True)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B",
                                                          quantization_config=self.bnb_config,
                                                          device_map="auto",trust_remote_code=True)
        self.instruction, self.aim = get_example(dataset_name)
        self.newline_token_id = self.tokenizer.encode("<b>", add_special_tokens=False)
    def inference(self, data, data_2=None):
        ans = []
        for _data in tqdm(data):
            inputs = f"What is the {self.aim} of the given protein sequence {_data}? {self.instruction}"
            if data_2 is not None:
                inputs = f"What is the {self.aim} between sequence <PROTEIN> {_data} </PROTEIN> and sequence <PROTEIN> {data_2} </PROTEIN>? {self.instruction}"

            inputs = self.tokenizer(inputs, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    top_p=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.newline_token_id
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(response)
                if self.dataset_name in Regression:
                    num = float(extract_num(response, True, False))
                else:
                    num = float(extract_num(response, False, True))

                ans.append(num)
        ans_tensor = torch.tensor(ans).unsqueeze(0).T
        return ans_tensor 



class BioT5_plus():
    def __init__(self, dataset_name):
        super(BioT5_plus, self).__init__()
        self.dataset_name = dataset_name
        self.instruction, self.aim = get_example(dataset_name)
        self.tokenizer = T5Tokenizer.from_pretrained("QizhiPei/biot5-plus-base-chebi20",
                                                     model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-plus-base-chebi20', device_map="auto")
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)

    def inference(self, data):
        ans = []
        for _data in data:
            inputs = f"What is the {self.aim} of the given protein sequence {_data}? {self.instruction}"
            inputs = self.tokenizer(inputs, return_tensors="pt").to("cuda")

            generation_config = self.model.generation_config
            generation_config.max_length = 64
            generation_config.num_beams = 1

            outputs = self.model.generate(**inputs, generation_config=generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(' ', '')
            print(response)
            if self.dataset_name in Regression:
                num = float(extract_num(response, True, False))
            else:
                num = float(extract_num(response, False, True))
            ans.append(num)
        ans_tensor = torch.tensor(ans).unsqueeze(0).T
        return ans_tensor

class ChemLLM_7B():
    def __init__(self, dataset_name):
        super(ChemLLM_7B, self).__init__()
        self.dataset_name = dataset_name
        self.instruction, self.aim = get_example(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained("AI4Chem/ChemLLM-7B-Chat-1_5-DPO", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained('AI4Chem/ChemLLM-7B-Chat-1_5-DPO', torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
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
                do_sample=True,
                max_new_tokens=max_tokens,
                temperature=0.05,
                top_p=1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and return the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def inference(self, data, data_2=None):
        ans = []
        idx = 0
        for _data in data:
            prompt = f"What is the {self.aim} of the given protein sequence {_data}? {self.instruction}"
            if data_2 is not None:
                prompt = f"What is the {self.aim} between sequence {_data} and sequence {data_2[idx]}? {self.instruction}"
            response = self.generate_response(
                self.model, self.tokenizer,
                instruction=self.instruction,
                prompt=prompt
            )
            print(response)
            if self.dataset_name in Regression:
                num = float(extract_num(response, True, False))
            else:
                num = float(extract_num(response, False, True))
            ans.append(num)
            idx += 1
        ans_tensor = torch.tensor(ans).unsqueeze(0).T
        return ans_tensor

class LlaSMol():
    def __init__(self, dataset_name):
        super(LlaSMol, self).__init__()
        self.dataset_name = dataset_name
        self.instruction, self.aim = get_example(dataset_name)
        from LlaSMol.generation import LlaSMolGeneration
        self.generator = LlaSMolGeneration('osunlp/LlaSMol-Mistral-7B')

    def inference(self, data, data_2=None):
        ans = []
        idx = 0
        for _data in tqdm(data):
            # In LlaSMol model, we should transform the protein sequence to SMILES string.
            # Which is like <SMILES> C1CCOC1 </SMILES> as presented in https://github.com/OSU-NLP-Group/LLM4Chem

            prompt = f"What is the {self.aim} of the given protein sequence <PROTEIN> {_data} </PROTEIN>? {self.instruction}"
            if data_2 is not None:
                prompt = f"What is the {self.aim} between sequence <PROTEIN> {_data} </PROTEIN> and sequence <PROTEIN> {data_2[idx]} </PROTEIN>? {self.instruction}"
            try:
                answer = self.generator.generate(prompt, max_input_tokens=len(prompt))[0]['output'][0]
                print(answer)
                if "insoluble" in answer:
                    answer = "0"
                elif "soluble" in answer:
                    answer = "1"
            except:
                answer = 0
            if self.dataset_name in Regression:
                num = float(extract_num(answer, True, False))
            else:
                num = float(extract_num(answer, False, True))
            ans.append(num)
            idx += 1
        ans_tensor = torch.tensor(ans).unsqueeze(0).T
        return ans_tensor

class ChemDFM():
    def __init__(self, dataset_name):
        super(ChemDFM, self).__init__()
        self.instruction, self.aim = get_example(dataset_name)
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained("OpenDFM/ChemDFM-v1.5-8B")
        self.model = LlamaForCausalLM.from_pretrained("OpenDFM/ChemDFM-v1.5-8B", torch_dtype=torch.float16, device_map="auto")
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)

        self.generation_config = GenerationConfig(
                do_sample=True,
                top_k=20,
                top_p=0.95,
                temperature=0.05,
                max_new_tokens=1024,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id
            )

    def inference(self, data):
        ans = []
        for _data in data:
            prompt = f"What is the {self.aim} of the given protein sequence {_data}? {self.instruction}"
            input_text = f"[Round 0]\nHuman: {prompt}\nAssistant:"
            inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):]
            print(generated_text)
            if self.dataset_name in Regression:
                num = float(extract_num(generated_text.strip(), True, False))
            else:
                num = float(extract_num(generated_text.strip(), False, True))
            ans.append(num)
        ans_tensor = torch.tensor(ans).unsqueeze(0).T
        return ans_tensor


import re



def extract_num(input_string, _float=False, _int=False):
    pattern = r'[+-]?\d+(?:\.\d+)?'
    numbers = re.findall(pattern, input_string)
    if numbers:
        value = float(numbers[0])
        if _float:
            return value
        if _int:
            if math.isinf(value):
                return sys.maxsize
            else:
                return int(value)

        return int(value)
    else:
        return 0