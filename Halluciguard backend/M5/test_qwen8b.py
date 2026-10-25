
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

MODEL_NAME = "Qwen/Qwen3-8B"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading Qwen3-8B in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model loaded successfully.")

question = input("\nEnter your question:\n> ").strip()

messages = [
    {
        "role": "user",
        "content": question
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=False
)

input_length = inputs["input_ids"].shape[-1]

response = tokenizer.decode(
    outputs[0][input_length:],
    skip_special_tokens=True
)

print("\n================================")
print("QWEN3-8B RESPONSE")
print("================================")
print(response)
