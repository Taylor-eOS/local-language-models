import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pysbd

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "auto"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
)

# Initialize sentence boundary detector
seg = pysbd.Segmenter(language="de", clean=False)

@torch.inference_mode
def translate_sentence(sentence: str):
    # Explicit prompt to ensure only the translation is returned
    prompt = f"Translate the following German sentence to English. Provide only the translation, without any additional explanation or text:\n\nGerman: {sentence}\nEnglish:"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate translation
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # Limit the output length to avoid rambling
        num_beams=1,  # Use greedy decoding for simplicity
        early_stopping=True,  # Stop generation if the model finishes the sentence
        pad_token_id=tokenizer.eos_token_id,  # Use EOS token to stop generation
    )
    
    # Decode the output and remove the prompt
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated_text = translated_text.replace(prompt, "").strip()  # Remove the prompt
    
    return translated_text

def main():
    # Read input sentences from input.txt
    with open("input.txt", "r", encoding="utf-8") as input_file:
        text = input_file.read()

    # Segment the text into sentences
    sentences = seg.segment(text)

    # Translate each sentence
    translated_sentences = []
    for sentence in sentences:
        translated_sentence = translate_sentence(sentence)
        translated_sentences.append(translated_sentence)

    # Write translated sentences to output.txt
    with open("output.txt", "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(translated_sentences))

if __name__ == "__main__":
    main()
