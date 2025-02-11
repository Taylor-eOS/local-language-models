import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.to("cpu")

def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    while True:
        prompt = input("User: ")
        response = generate_response(prompt)
        print("Model Response:", response)
