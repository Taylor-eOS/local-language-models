import os
import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

_model = None
_tokenizer = None
_model_lock = threading.Lock()

def _load_components():
    global _model, _tokenizer
    with _model_lock:
        if _model is None or _tokenizer is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            try:
                _tokenizer = AutoTokenizer.from_pretrained(
                    'Nanbeige/Nanbeige4.1-3B',
                    cache_dir=cache_dir,
                    local_files_only=True,
                    use_fast=False,
                    trust_remote_code=True
                )
                _model = AutoModelForCausalLM.from_pretrained(
                    'Nanbeige/Nanbeige4.1-3B',
                    cache_dir=cache_dir,
                    local_files_only=True,
                    torch_dtype='auto',
                    device_map='cpu',
                    trust_remote_code=True
                )
            except OSError:
                _tokenizer = AutoTokenizer.from_pretrained(
                    'Nanbeige/Nanbeige4.1-3B',
                    cache_dir=cache_dir,
                    use_fast=False,
                    trust_remote_code=True
                )
                _model = AutoModelForCausalLM.from_pretrained(
                    'Nanbeige/Nanbeige4.1-3B',
                    cache_dir=cache_dir,
                    torch_dtype='auto',
                    device_map='cpu',
                    trust_remote_code=True
                )
    return _tokenizer, _model

def generate_text(system_prompt, user_message):
    if not system_prompt or not user_message:
        raise ValueError("system_prompt and user_message must be provided")
    tokenizer, model = _load_components()
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_message}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
        streamer=streamer,
        eos_token_id=166101,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    base = ""
    behaviour = input("Custom behaviour (default helpful): ") or "You are a helpful assistant."
    if not behaviour.endswith(('.', '?', '!')):
        behaviour += '.'
    concise_part = input("Style instruction (default concise): ") or " Write concisely and continuously. Skip introductions, and get straight to the point. Be mindful that responses take a lot of time to process, so provide content in as few words as possible. Employ a great deal of creative discernment."
    system_prompt = base + behaviour + concise_part
    print(f"System prompt: {system_prompt}")
    while True:
        user_message = input("Enter user message: ")
        if user_message.lower() == 'exit':
            break
        print(generate_text(system_prompt, user_message).replace("\n\n", "\n"))

if __name__ == "__main__":
    main()

