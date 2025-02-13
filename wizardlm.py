from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = 'cognitivecomputations/WizardLM-7B-Uncensored'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

def stream_response(prompt):
    prompt = [{'role': 'user', 'content': prompt}]
    inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors='pt').to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model.generate(inputs, max_new_tokens=512, temperature=0.8, do_sample=True, streamer=streamer)

while True:
    prompt = input("User: ")
    stream_response(prompt)
