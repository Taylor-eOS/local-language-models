import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

def main():
    # Get system prompt from user
    system_prompt = input("Enter system prompt (default: helpful assistant): ") \
                    or "You are a helpful AI assistant."
    
    messages = [{"role": "system", "content": system_prompt}]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        # Get user input
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Prepare input with chat template
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input}],  # Only current message
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        # Generate streamed response without chat history
        print("Assistant:", end=" ", flush=True)
        model.generate(
            inputs,
            max_new_tokens=500,
            temperature=0.0,
            do_sample=False,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False  # Critical fix for Phi-3 compatibility
        )

if __name__ == "__main__":
    main()
