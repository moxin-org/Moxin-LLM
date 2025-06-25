## Inference

You can use the following code to run inference with the model. 

```bash
conda create -n moxin-llm python=3.10 -y
conda activate moxin-llm

pip install -r requirements.txt

python inference-Instruct.py
```

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
