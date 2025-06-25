import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList

model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
    trust_remote_code=True)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_sequences):
        self.stop_sequences = stop_sequences
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in self.stop_sequences:
            if input_ids[0, -len(seq):].tolist() == seq:
                return True
        return False

stop_strings = ["\nYou:", "\nUser:", tokenizer.eos_token]
stop_sequences = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]

generation_kwargs = {
    "max_new_tokens": 2048,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.eos_token_id,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "stopping_criteria": StoppingCriteriaList([StopOnTokens(stop_sequences)]),}

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
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            stop_str=stop_strings)
        torch.cuda.empty_cache()
        _ = model.generate(
            **inputs,
            streamer=streamer,
            **generation_kwargs)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break

