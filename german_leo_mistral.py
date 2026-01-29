import os
import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

_model = None
_tokenizer = None
_model_lock = threading.Lock()

def _load_components(model_name="jphme/em_german_leo_mistral"):
    global _model, _tokenizer
    with _model_lock:
        if _model is None or _tokenizer is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            try:
                _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
                _model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cpu")
            except OSError:
                _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                _model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="cpu")
    return _tokenizer, _model

def generate_text(system_prompt, user_message):
    if not system_prompt or not user_message:
        raise ValueError("system_prompt and user_message must be provided")
    tokenizer, model = _load_components()
    prompt = f"{system_prompt}\n\n{user_message}".strip()
    full_prompt = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9, streamer=streamer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response

def main():
    base = "Du bist ein Assistent. "
    behaviour = input("Verhalten: ") or "Du beantwortest fragen."
    if not behaviour.endswith(('.', '?', '!')):
        behaviour += '.'
    concise_part = input("Stil Instruktion: ") or " Du fässt dich kurz."
    system_prompt = base + behaviour + concise_part
    print(f"System: {system_prompt}")
    while True:
        user_message = input("Nutzer: ")
        if user_message.lower() == 'exit':
            break
        response = generate_text(system_prompt, user_message)
        print(response.replace("\n\n", "\n"))

if __name__ == "__main__":
    main()

