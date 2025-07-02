import transformers
import torch

model_id = "moxin-org/Moxin-7B-Instruct" # or  "moxin-org/Moxin-7B-Reasoning" 
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant!"},
    {"role": "user", "content": "Can you explain the concept of regularization in machine learning?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=1024,
)
print(outputs[0]["generated_text"][-1]["content"])