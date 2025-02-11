import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cpu")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

while True:
    input_string = input("User: ")
    inputs = tokenizer(input_string, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=100)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)
