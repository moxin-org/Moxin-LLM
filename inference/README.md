## Inference

You can use the following code to run inference with the model. 
```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

model_name = 'moxin-org/Moxin-7B-LLM'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Can you explain the concept of regularization in machine learning?"

sequences = pipe(
    prompt,
    do_sample=True,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
)
print(sequences[0]['generated_text'])
```

For the Instruct model and Reasoning model, you can use the following code for inference.
```
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
    {"role": "user", "content": "How are you doing?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=1024,
)
print(outputs[0]["generated_text"][-1])
```

For the inference of our VLM, pleaser refer to [Moxin-VLM](https://github.com/moxin-org/Moxin-VLM) for environment construction and inference code.

### Chat Template

The chat template is formatted as:
```
<|system|>\nYou are a helpful AI assistant!\n<|user|>\nHow are you doing?\n<|assistant|>\nThank you for asking! As an AI, I don't have feelings, but I'm functioning normally and ready to assist you. How can I help you today?<|endoftext|>
```
Or with new lines expanded:
```
<|system|>
You are a helpful AI assistant!
<|user|>
How are you doing?
<|assistant|>
Thank you for asking! As an AI, I don't have feelings, but I'm functioning normally and ready to assist you. How can I help you today?<|endoftext|>
```
