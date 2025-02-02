import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_NAME = "cognitivecomputations/dolphin-2.1-mistral-7b"
DEVICE = "auto"
REPLACEMENTS = ["\nWait, but", "\nHmm", "\nSo"]
MIN_THINKING_TOKENS = 16
PREFILL = ""

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
)

@torch.inference_mode
def reasoning_effort(question: str, min_thinking_tokens: int):
    # Prepare the input with chat template
    tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": PREFILL},
        ],
        return_tensors="pt",
    )
    tokens = tokens.to(model.device)
    n_thinking_tokens = 0

    while True:
        # Generate the next token
        out = model(input_ids=tokens, use_cache=True)
        next_token = torch.multinomial(
            torch.softmax(out.logits[0, -1, :], dim=-1), 1
        ).item()

        # If the model generates an EOS token, stop
        if next_token == tokenizer.eos_token_id:
            break

        # Decode and yield the token
        yield tokenizer.decode([next_token])

        # Update the tokens for the next iteration
        tokens = torch.tensor([[next_token]], device=tokens.device)
        n_thinking_tokens += 1

        # If we haven't reached the minimum thinking tokens, inject a replacement
        if n_thinking_tokens < min_thinking_tokens and random.random() < 0.3:  # 30% chance to inject a replacement
            replacement = random.choice(REPLACEMENTS)
            yield replacement
            replacement_tokens = tokenizer.encode(replacement, add_special_tokens=False)
            tokens = torch.tensor([replacement_tokens], device=tokens.device)
            n_thinking_tokens += len(replacement_tokens)

def main():
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting...")
            break

        for chunk in reasoning_effort(question, MIN_THINKING_TOKENS):
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()

