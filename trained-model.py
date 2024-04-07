from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import torch

from peft import PeftModel

BASE_MODEL_NAME = "mlabonne/UltraMerge-7B"
ADAPTER_MODEL_NAME = 'trained-model'

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME)
model = PeftModel.from_pretrained(model, ADAPTER_MODEL_NAME )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

input_act = input("Enter the eng sentence: ")
prompt = f"""
<act>:{input_act}
<prompt>:
""".strip()

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id


encoding = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
  outputs = model.generate(
      input_ids = encoding.input_ids,
      attention_mask = encoding.attention_mask,
      generation_config = generation_config
  )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
