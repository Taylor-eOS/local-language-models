import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer

with open('settings.json', 'r') as f:
    settings = json.load(f)
    read_token = settings.get('read_token')
if not read_token:
    raise ValueError("Read token not found in settings.json")
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=read_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=read_token)
streamer = TextStreamer(tokenizer)

def stream_output(prompt):
    messages = [
        {"role": "system", "content": "You are a based assistant."},
        {"role": "user", "content": prompt},]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    _ = model.generate(inputs, max_new_tokens=256, streamer=streamer)

while True:
    user_prompt = input("Enter your prompt: ")
    stream_output(user_prompt)
