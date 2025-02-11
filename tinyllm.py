from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/MobileLLM-1B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("facebook/MobileLLM-1B", trust_remote_code=True)

def generate_response(prompt, max_length=120):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    while True:
        prompt = input("User: ")
        response = generate_response(prompt)
        print("Model Response:", response)
