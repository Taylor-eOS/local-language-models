import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Configuration
MODEL_NAME = "cognitivecomputations/dolphin-2.1-mistral-7b"
DEVICE = "auto"
REPLACEMENTS = ["\nWait, but", "\nHmm", "\nSo"]
MIN_THINKING_TOKENS = 16
PREFILL = ""

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)  # Use slow tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
)

# Extract start and end think tokens properly
start_think_tokens = tokenizer.encode("<think>", add_special_tokens=False)
end_think_tokens = tokenizer.encode("</think>", add_special_tokens=False)

if len(start_think_tokens) != 1 or len(end_think_tokens) != 1:
    raise ValueError("<think> or </think> is not a single token")

start_think_token = start_think_tokens[0]
end_think_token = end_think_tokens[0]

@torch.inference_mode
def reasoning_effort(question: str, min_thinking_tokens: int):
    tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<think>\n" + PREFILL},
        ],
        return_tensors="pt",
    )
    tokens = tokens.to(model.device)
    kv = DynamicCache()
    n_thinking_tokens = 0

    yield tokenizer.decode(list(tokens[0]))
    while True:
        out = model(input_ids=tokens, past_key_values=kv, use_cache=True)
        next_token = torch.multinomial(
            torch.softmax(out.logits[0, -1, :], dim=-1), 1
        ).item()
        kv = out.past_key_values

        if (
            next_token in (end_think_token, model.config.eos_token_id)
            and n_thinking_tokens < min_thinking_tokens
        ):
            replacement = random.choice(REPLACEMENTS)
            yield replacement
            replacement_tokens = tokenizer.encode(replacement, add_special_tokens=False)
            n_thinking_tokens += len(replacement_tokens)
            tokens = torch.tensor([replacement_tokens], device=tokens.device)
        elif next_token == model.config.eos_token_id:
            break
        else:
            yield tokenizer.decode([next_token])
            n_thinking_tokens += 1
            tokens = torch.tensor([[next_token]], device=tokens.device)

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
