import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model = AutoModelForCausalLM.from_pretrained(
    "trillionlabs/Trillion-7B-preview",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("trillionlabs/Trillion-7B-preview")

while True:
    try:
        prompt = input("\nYou: ").strip()
        if not prompt:
            continue
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            **inputs,
            max_new_tokens=512,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id)
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break

