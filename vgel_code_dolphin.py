import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Configuration
MODEL_NAME = "cognitivecomputations/dolphin-2.1-mistral-7b"
DEVICE = "auto"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE)

def generate_text(prompt, max_length=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Create a streamer object
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generate text with streaming
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        streamer=streamer,  # Pass the streamer to the generate function
    )
    
    # Decode the final output (optional, if you want to return the full text)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    while True:
        question = input("Enter your question (type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting...")
            break
        
        print("Response:", end=" ")
        response = generate_text(question)
        print("\n")

if __name__ == "__main__":
    main()
