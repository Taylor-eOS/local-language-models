import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = "trillionlabs/Trillion-7B-preview"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

while True:
    prompt = input("\nPrompt: ")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    _ = model.generate(
        model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=512,
        streamer=streamer,
    )

