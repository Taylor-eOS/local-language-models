import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

system_message = input("System: ")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = input("Prompt: ")
messages = [
    {"role": "system", 
     "content": system_message},
    {"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt")
output = model.generate(
    input_ids.to("cpu"),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    do_sample=False,)
print(tokenizer.decode(output[0]))

