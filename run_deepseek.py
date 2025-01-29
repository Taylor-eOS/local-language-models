import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the pad token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as pad token

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt, max_length=100):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the generated tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test prompts
test_prompts = [
    #"Question: What is the capital of France? Answer:",
    #'German input: "Ich bin ein engländer." English translation: ',
    'German: Hier präsentiert Fernau eine "andere" Geschichte Amerikas. Anders deshalb, weil er bewußt die dunklen Seiten der Historie erhellt." English translation: ',
]

# Generate and print responses
for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    response = generate_response(prompt)
    print(f"Response: {response}\n")

