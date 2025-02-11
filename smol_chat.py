from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
conversation_history = []
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    conversation_history.append({"role": "user", "content": user_input})
    input_text = tokenizer.apply_chat_template(conversation_history, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    model_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model:", model_response)
    conversation_history.append({"role": "assistant", "content": model_response})
