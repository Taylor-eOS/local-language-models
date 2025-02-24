from transformers import pipeline, set_seed
import json

set_seed(42)
pipe = pipeline("text-generation", "malteos/german-r1")

# from gsm8k test set
question = "James beschließt, 3-mal pro Woche 3 Sprints zu laufen.  Er läuft 60 Meter pro Sprint.  Wie viele Meter läuft er insgesamt pro Woche?"
expected_answer = "540"

# xml reasoning and answer format
system_prompt = """
Antworte auf deutsch und in diesem Format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {"role": "user", "content": dataset["question"][3]},
]
response = pipe(messages, max_new_tokens=256)

print(json.dumps(response, indent=4, ensure_ascii=False))

