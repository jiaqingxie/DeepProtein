from transformers import AutoModel, AutoTokenizer
from instruction import *
import torch
import re
class BioMistral():
    def __init__(self, dataset_name):
        super(BioMistral, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
        self.model = AutoModel.from_pretrained("BioMistral/BioMistral-7B", torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
        self.instruction, self.aim = instruction_list[dataset_name]
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
                    pad_token_id=self.newline_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            num = float(self.extract_num(response))
            ans.append(num)
        ans_tensor = torch.tensor(ans).unsqueeze(0)  # Shape: [1, len(data)]
        return ans_tensor

    def extract_num(self, input_string):

        numbers = re.findall(r'\d+', input_string)
        return numbers[-1] if numbers else 0