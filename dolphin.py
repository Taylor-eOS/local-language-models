import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_NAME = "cognitivecomputations/dolphin-2.1-mistral-7b"
DEVICE = "auto"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    while True:
        question = input("Enter your question: ")
        if question.lower() == 'exit':
            print("Exiting...")
            break
        print("Response:", end=" ")
        response = generate_text(question)
        print("\n")

if __name__ == "__main__":
    main()
