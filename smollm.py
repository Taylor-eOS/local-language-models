from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
model.to(device)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    messages = [{"role": "user", "content": user_input}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(end="")
    model.generate(inputs, max_new_tokens=500, temperature=0.2, top_p=0.9, do_sample=True, streamer=streamer)
