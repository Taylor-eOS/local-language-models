import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_NAME = "cognitivecomputations/Dolphin3.0-Llama3.2-3B"
DEVICE = "auto"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE)

def generate_text(system_prompt, user_message):
    full_prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_message}\n<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    system_prompt = input("Enter the system prompt: ")
    while True:
        user_message = input("Enter your message: ")
        if user_message.lower() == 'exit':
            break
        response = generate_text(system_prompt, user_message).replace("\n\n", "\n")
        print("")
        #print("Response:", response) #.replace("\n\n", "\n")

if __name__ == "__main__":
    main()
