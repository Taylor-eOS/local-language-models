from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "arcee-ai/Llama-3.1-SuperNova-Lite" #This is 8B
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
while True:
    prompt = input("User: ")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200,  
            num_return_sequences=1,  
            no_repeat_ngram_size=2,  
            top_k=50,  
            top_p=0.95,  
            temperature=0.7,  
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
