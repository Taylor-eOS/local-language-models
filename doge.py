import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer

MODEL_NAME = "SmallDoge/Doge-60M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")

generation_config = GenerationConfig(
    max_new_tokens=100,
    use_cache=True,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.0,
)
streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

def main():
    print("Welcome")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        conversation = [{"role": "user", "content": user_input}]
        inputs = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        outputs = model.generate(
            input_ids=inputs,
            generation_config=generation_config,
            streamer=streamer,
        )

if __name__ == "__main__":
    main()

