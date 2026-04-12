import os
import threading
from llama_cpp import Llama

_model = None
_model_lock = threading.Lock()

def _load_model(repo_id="TheBloke/em_german_leo_mistral-GGUF", filename="em_german_leo_mistral.Q4_K_M.gguf"):
    global _model
    with _model_lock:
        if _model is None:
            model_path = os.path.expanduser(f"~/.cache/llama_cpp/{filename}")
            _model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=4096,
                n_threads=os.cpu_count(),
                verbose=False,
            )
    return _model

def generate_text(user_message):
    if not user_message:
        raise ValueError("user_message must be provided")
    model = _load_model()
    prompt = f"<|im_start|>user\n{user_message}\n<|im_end|>\n<|im_start|>assistant\n"
    stream = model(
        prompt,
        max_tokens=1024,
        temperature=0.6,
        top_p=0.9,
        stream=True,
        stop=["<|im_end|>", "</s>"],
        echo=False,
    )

    response_text = ""
    for chunk in stream:
        token = chunk["choices"][0]["text"]
        print(token, end="", flush=True)
        response_text += token
    print()
    return response_text

def main():
    print("Using model: TheBloke/em_german_leo_mistral-GGUF")
    while True:
        user_message = input("Enter user message (or 'exit' to quit): ")
        if user_message.lower() == 'exit':
            break
        generate_text(user_message)

if __name__ == "__main__":
    main()

