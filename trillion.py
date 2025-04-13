import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model = AutoModelForCausalLM.from_pretrained(
    "trillionlabs/Trillion-7B-preview",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("trillionlabs/Trillion-7B-preview")

# Add stopping criteria or parameters to help the model know when to stop
generation_kwargs = {
    "max_new_tokens": 512,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.eos_token_id,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "stopping_criteria": None,  # You can add custom stopping criteria if needed
}

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
        
        # Create a streamer that will stop at appropriate points
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            # You can add a stopping pattern if the model has specific ending tokens
            stop_str=[tokenizer.eos_token, "\nYou:", "\nUser:"]
        )
        
        # Clear the GPU cache to prevent memory issues
        torch.cuda.empty_cache()
        
        # Generate the response
        _ = model.generate(
            **inputs,
            streamer=streamer,
            **generation_kwargs
        )
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
