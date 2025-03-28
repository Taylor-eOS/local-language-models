import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = "trillionlabs/Trillion-7B-preview"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize conversation history
messages = []

# Continuous conversation loop
while True:
    # Get user input
    prompt = input("\nPrompt: ")
    messages.append({"role": "user", "content": prompt})
    
    # Format conversation with chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Create streamer
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # Generate streamed response
    generated_ids = model.generate(
        model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=512,
        streamer=streamer,
    )
    
    # Extract new tokens (remove input context)
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Save response to conversation history
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    messages.append({"role": "assistant", "content": response})
