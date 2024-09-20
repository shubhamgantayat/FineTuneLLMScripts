from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import re


class Inference:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
        self.config = PeftConfig.from_pretrained("shubhamgantayat/paper-finetune-model-phi1.5-gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5").to(self.device)
        self.model = PeftModel.from_pretrained(self.model, "shubhamgantayat/paper-finetune-model-phi1.5-gpt2").to(self.device)

    def predict(self, query):
        inputs = self.tokenizer([f"""### Question: 
        {query}

        ### Answer: """], return_tensors="pt").to(self.device)
        peft_model_outputs = self.model.generate(**inputs, max_new_tokens=1024)
        peft_model_text_output = self.tokenizer.batch_decode(peft_model_outputs)
        return peft_model_text_output[0]


inf = Inference()
while True:
    query = input("Enter Query: ")
    answer = inf.predict(query)
    end = re.search('<|endoftext|>', answer).span()[0]
    # start = len(query)
    print(answer[: end])
