import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Writer/camel-5b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16)
instruction = input("User: ")
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),}

text = (
    PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
    if not input
    else PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input))

model_inputs = tokenizer(text, return_tensors="pt").to("cpu")
output_ids = model.generate(
    **model_inputs,
    max_length=256,
)
output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
clean_output = output_text.split("### Response:")[1].strip()

print(clean_output)

