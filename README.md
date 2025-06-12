# Moxin LLM
Moxin is a family of fully open-source and reproducible LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2412.06845-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2412.06845v5)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](https://github.com/moxin-org/Moxin-LLM/blob/main/LICENSE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-moxin--org-yellow.svg?style=for-the-badge&logo=HuggingFace)](https://huggingface.co/moxin-org)

---

## Introduction

Generative AI (GAI) offers unprecedented opportunities for research and innovation, but its commercialization has raised concerns about transparency, reproducibility, and safety. Many open GAI models lack the necessary components for full understanding and reproducibility, and some use restrictive licenses whilst claiming to be “open-source”. To address these concerns, we follow the [Model Openness Framework (MOF)](https://arxiv.org/pdf/2403.13784), a ranked classification system that rates machine learning models based on their completeness and openness, following principles of open science, open source, open data, and open access. 

By promoting transparency and reproducibility, the MOF combats “openwashing” practices and establishes completeness and openness as primary criteria alongside the core tenets of responsible AI. Wide adoption of the MOF will foster a more open AI ecosystem, benefiting research, innovation, and adoption of state-of-the-art models. 

In line with the MOF, we release our datasets used during training, the training scripts, and the trained models. 


## Model
You can download our  [Moxin-7B-Base](https://huggingface.co/moxin-org/moxin-llm-7b),  [Moxin-7B-Chat](https://huggingface.co/moxin-org/moxin-chat-7b), [Moxin-7B-Instruct](https://huggingface.co/moxin-org/moxin-instruct-7b)  and [Moxin-7B-Reasoning](https://huggingface.co/moxin-org/moxin-reasoning-7b) models. 
 

## Evaluation 

### Base Model Evaluation

We test the performance of our base model with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The evaluation results on common datasets are shown below. We test on AI2 Reasoning Challenge (25-shot), HellaSwag (10-shot), MMLU (5-shot), and Winogrande (5-shot).  We release the Moxin-7B-Enhanced  as our base model. We further finetune our base model on Tulu v2 to obtain our chat model. 

|          Models         | ARC-C | Hellaswag |  MMLU | WinoGrade |  Ave  |
|:----------------------:|:-----:|:---------:|:-----:|:---------:|:-----:|
|    Mistral-7B   | 57.59 |   83.25   | 62.42 |   78.77   | 70.51 |
|     LLaMA 3.1-8B     | 54.61 |   81.95   | 65.16 |   77.35   | 69.77 |
|      LLaMA 3-8B      | 55.46 |   82.09   | 65.29 |   77.82   | 70.17 |
|      LLaMA 2-7B      | 49.74 |   78.94   | 45.89 |   74.27   | 62.21 |
|       Qwen 2-7B      | 57.68 |   80.76   | 70.42 |   77.43   | 71.57 |
|       Gemma-7b       | 56.48 |   82.31   | 63.02 |    78.3   | 70.03 |
|    Internlm2.5-7b    | 54.78 |    79.7   | 68.17 |    80.9   | 70.89 |
|     Baichuan2-7B     | 47.87 |   73.89   | 54.13 |    70.8   | 61.67 |
|        Yi-1.5-9B       | 58.36 |   80.36   | 69.54 |   77.53   | 71.48 |
|  Moxin-7B-Original | 53.75 |   75.46   | 59.43 |   70.32   | 64.74 |
| Moxin-7B-Enhanced (Moxin-7B-Base)| 59.47 |   83.08   | 60.97 |   78.69   | 70.55 |


We also test the zero shot performance on AI2 Reasoning Challenge (0-shot), AI2 Reasoning Easy (0-shot), HellaSwag (0-shot), PIQA (0-shot) and Winogrande (0-shot). The results are shown below. 

|      Models       	| HellaSwag 	| WinoGrade 	|  PIQA 	| ARC-E 	| ARC-C 	|  Ave  	|
|:-----------------:	|:---------:	|:---------:	|:-----:	|:-----:	|:-----:	|:-----:	|
| Mistral-7B 	|   80.39   	|    73.4   	| 82.15 	| 78.28 	| 52.22 	| 73.29 	|
|     LLaMA 2-7B    	|   75.99   	|   69.06   	| 79.11 	| 74.54 	| 46.42 	| 69.02 	|
|    LLaMA 2-13B    	|   79.37   	|   72.22   	| 80.52 	|  77.4 	| 49.06 	| 71.71 	|
|    LLaMA 3.1-8B   	|   78.92   	|   74.19   	| 81.12 	| 81.06 	| 53.67 	| 73.79 	|
|      Gemma-7b     	|   80.45   	|   73.72   	|  80.9 	| 79.97 	|  54.1 	| 73.83 	|
|     Qwen v2-7B    	|    78.9   	|   72.38   	| 79.98 	| 74.71 	| 50.09 	| 71.21 	|
|   Internlm2.5-7b  	|   79.14   	|    77.9   	| 80.52 	| 76.16 	| 51.37 	| 73.02 	|
|    Baichuan2-7B   	|   72.25   	|   67.17   	| 77.26 	| 72.98 	| 42.15 	| 66.36 	|
|     Yi-1.5-9B     	|   77.86   	|   73.01   	| 80.74 	| 79.04 	| 55.03 	| 73.14 	|
|    Deepseek-7b    	|   76.13   	|   69.77   	| 79.76 	| 71.04 	|  44.8 	|  68.3 	|
| Moxin-7B-Original 	|   72.06   	|   66.31   	| 78.07 	| 71.47 	| 48.15 	| 67.21 	|
| Moxin-7B-Enhanced (Moxin-7B-Base)  	|   80.03   	|   75.17   	| 82.24 	| 81.12 	| 58.64 	| 75.44 	|



### Instruct Model Evaluation

Our instruct model is trained with [Tulu 3](https://allenai.org/blog/tulu-3-technical). The evaluations are demonstrated below. We evaluate with  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [OLMES](https://github.com/allenai/olmes). 

We test on AI2 Reasoning Challenge (25-shot), HellaSwag (10-shot), MMLU (5-shot), and Winogrande (5-shot).
|Model |ARC-C| Hellaswag| MMLU |WinoGrade| Ave|
|:-----------------:	|:---------:	|:---------:	|:-----:	|:-----:	|:-----:	|
|Mistral 8B Instruct| 62.63 |80.61 |64.16| 79.08| 71.62|
|Llama3.1 8B Instruct| 60.32 |80 |68.18 |77.27| 71.44|
|Qwen2.5 7B Instruct| 66.72 |81.54| 71.3 |74.59| 73.54|
|Moxin-7B-SFT|  60.11 |83.43| 60.56| 77.56| 70.42|
|Moxin-7B-DPO (Moxin-7B-Instruct) | 64.76 |87.19| 58.36| 76.32| 71.66|


We also test the zero shot performance on AI2 Reasoning Challenge (0-shot), AI2 Reasoning Easy (0-shot), HellaSwag (0-shot), PIQA (0-shot) and Winogrande (0-shot). The results are shown below.
|Models | HellaSwag | WinoGrade | PIQA | ARC-E | ARC-C | Ave  |
|:-----------------:	|:---------:	|:---------:	|:-----:	|:-----:	|:-----:	|:-----:	|
|Mistral 8B Instruct | 79.08 | 73.56 | 82.26 | 79.88 | 56.57 | 74.27 | 
| Llama3.1 8B Instruct | 79.21| 74.19 |80.79 |79.71 |55.03 |73.79|
|Qwen2.5 7B Instruct | 80.5 | 71.03 | 80.47 | 81.31 | 55.12 | 73.69 |
|Moxin-7B-SFT  |81.44 |73.09 |81.07 |79.8 |54.67| 74.01|
|Moxin-7B-DPO (Moxin-7B-Instruct) | 85.7 | 73.24 | 81.56 |81.1 |58.02| 75.92|




The evaluation results with OLMES are shown below. 
|Models/Datasets |GSM8K |MATH |Humaneval |Humaneval plus |MMLU |PopQA |BBH |TruthfulQA| Ave|
|:-----------------:	|:---------:	|:---------:	|:-----:	|:-----:	|:-----:	|:-----:	|:-----:	|:-----:	|:-----:	|
|Qwen2.5 7B Instruct |83.8 |14.8 |93.1 |89.7 |76.6 |18.1 |21.7 |63.1| 57.61|
|Gemma2 9B Instruct| 79.7 |29.8 |71.7 |67 |74.6 |28.3 |2.5 |61.4 |51.88|
|Moxin-7B-DPO (Moxin-7B-Instruct) |81.19| 36.42| 82.86| 77.18 |60.85 |23.85 |57.44| 55.27 |59.38|


### Reasoning Model Evaluation

Our reasoning model is trained with [DeepScaleR](https://github.com/agentica-project/rllm). The evaluation on math datasets are demonstrated below. 

|Models/Datasets |MATH 500 |AMC |Minerva Math |OlympiadBench |Ave|
|:-----------------:	|:---------:	|:---------:	|:-----:	|:-----:	|:-----:	|
|Qwen2.5-Math-7B-Base |52.4 |52.5 |12.9 |16.4| 33.55|
|Qwen2.5-Math-7B-Base + 8K MATH SFT |54.6 |22.5| 32.7| 19.6| 32.35|
|Llama-3.1-70B-Instruct| 64.6 |30.1 |35.3| 31.9| 40.48|
|Moxin-7B-RL-DeepScaleR| 68 |57.5 |16.9| 30.4 |43.2|


### VLM Model Evaluation

Our VLM model is trained with [prismatic-vlms](https://github.com/TRI-ML/prismatic-vlms). The evaluation is demonstrated below. 

|                          	|  GQA  	| VizWiz 	| RefCOCO+ 	| OCID-Ref 	|  VSR  	|  POPE 	| TallyQA 	|  Ave. 	|
|--------------------------	|:-----:	|:------:	|:--------:	|:--------:	|:-----:	|:-----:	|:-------:	|:-----:	|
| LLaVa v1.5 7B (Base)     	| 61.58 	|  54.25 	|   49.47  	|   35.07  	| 51.47 	| 86.57 	|  62.06  	| 57.21 	|
| Llama-2 Chat 7B          	| 62.11 	|  56.39 	|   58.5   	|   46.3   	|  61.8 	|  86.8 	|   58.1  	| 61.43 	|
| Mistral v0.1 7B          	|  63.3 	|  55.32 	|   65.1   	|   48.8   	|  58.5 	|  87.1 	|   61.7  	| 62.83 	|
| Mistral Instruct v0.1 7B 	| 62.71 	|  54.35 	|   64.9   	|    48    	|  57.8 	|  87.5 	|   64.5  	| 62.82 	|
| Llama-2 7B               	| 62.44 	|  55.98 	|   59.47  	|   43.89  	| 63.67 	| 86.74 	|  59.22  	| 61.63 	|
| Ours                     	| 64.88 	|  54.08 	|   71.3   	|   48.4   	|  60.8 	|  87.3 	|    66   	| 64.68 	|


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


### Convert to GGUF


Build a typical deep learning environment with pytorch. Then use the script covert_hf_to_gguf.py to convert the hf model to GGUF.
```
python covert_hf_to_gguf.py  path_to_model_directory/
```
Then, you can experiment with this gguf model following [llama.cpp](https://github.com/ggerganov/llama.cpp). 



##  Reinforcement Learning with GRPO

To enhance the CoT capabilities of our model, we adopt RL techniques similar to DeepSeek R1. We first use high quality reasoning data to SFT our instruct model.  The reasoning data mainly includes Openthoughts   and OpenR1-Math-220k. Next, we  adopt  the RL techniques in DeepSeek R1, i.e., GRPO to  finetune our model with RL.  We adopt the [DeepScaleR](https://github.com/agentica-project/rllm)  as our RL training framework.

We first use high quality reasoning data to SFT our instruct (DPO) model.
+ Dataset:  [OpenThoughts](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) and  [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
+ Framework: [open-instruct](https://github.com/allenai/open-instruct)
+ Configuration: [Llama-3.1-Tulu-3-8B-SFT](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)

Refer to 'scripts/finetune/instruct_finetune/sft_finetune.sh' for more details. 

Next, we  adopt GRPO to  finetune our model with RL.
+ Framework, configuration and Dataset: [DeepScaleR](https://github.com/agentica-project/rllm)

Refer to 'scripts/finetune/reason_finetune/train_7b.sh' for more details. 

## Post-Training with Tülu 3

The open-source Tülu 3 dataset and framework are adopted for the model post-training. For our post-training, with our base model, we follow Tülu 3 to perform supervised finetuning (SFT) and then Direct Preference Optimization (DPO).

Specifically, we use the Tülu 3 SFT Mixture dataset from Tülu 3  to train our base model with  the SFT training method for two epochs and obtain our SFT model, following the default training configuration of the Tülu 3 8B SFT model. 
+ Dataset:  [Tülu 3 SFT Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
+ Framework: [open-instruct](https://github.com/allenai/open-instruct)
+ Configuration: [Llama-3.1-Tulu-3-8B-SFT](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)

Refer to 'scripts/finetune/instruct_finetune/sft_finetune.sh' for more details. 

Next, we continue to train our SFT  model  on the Tülu 3 8B Preference Mixture dataset from Tülu 3  with the DPO training method to obtain our DPO model, following the same training configuration of the Tülu 3 8B DPO model. 
+ Dataset:  [Tülu 3 8B Preference Mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture)
+ Framework: [open-instruct](https://github.com/allenai/open-instruct)
+ Configuration: [Llama-3.1-Tulu-3-8B-DPO](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)

Refer to 'scripts/finetune/instruct_finetune/dpo_finetune.sh' for more details. 


## Pre-Training Environment

#### 1. Dataset config
To prepare the dataset, it needs to install the following package,


```
pip install datasets
```

#### 2. Cuda install

We use cuda 11.7. Other cuda versions may also work.
```
get https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run    
```

#### 3. Install pytorch

We use pytorch 2.0.0. 
```
conda create --name llm_train python==3.10
conda activate llm_train
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```

#### 4. Install other packages

To install other packages, follow the requirements.txt
```
pip install -r requirements.txt
```

#### 5. Install flash attention

We use flash-attention 2.2.1.
```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/
git checkout a1576ad                ##  flash-attention 2.2.1
python setup.py  install
cd ./csrc
cd fused_dense_lib  && pip install -v .
cd ../xentropy && pip install -v .
cd ../rotary && pip install -v .
cd ../layer_norm && pip install -v .
```


## Pretrain Datasets


To use the [SlimPajama dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B) for pretraining, you can download the dataset using Hugging Face datasets:
```
import datasets 
ds = datasets.load_dataset("cerebras/SlimPajama-627B")
```
SlimPajama is the largest extensively deduplicated, multi-corpora, open-source dataset for training large language models. SlimPajama was created by cleaning and deduplicating the 1.2T token RedPajama dataset from Together. By filtering out low quality data and duplicates, it  removes 49.6% of bytes, slimming down the RedPajama dataset from 1210B to 627B tokens.   SlimPajama offers the highest quality and most compute efficient data to train on for runs up to 627B tokens. When upsampled, SlimPajama is expected   to perform equal to or better than RedPajama-1T when training at trillion token scale. 


To use the [stack-dedup dataset](https://huggingface.co/datasets/bigcode/the-stack-dedup) for pretraining, you can download the dataset using Hugging Face datasets:
```
from datasets import load_dataset

# full dataset (3TB of data)
ds = load_dataset("bigcode/the-stack-dedup", split="train")

# specific language (e.g. Dockerfiles)
ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/dockerfile", split="train")

# dataset streaming (will only download the data as needed)
ds = load_dataset("bigcode/the-stack-dedup", streaming=True, split="train")
for sample in iter(ds): print(sample["content"])
```
The Stack contains over 6TB of permissively-licensed source code files covering 358 programming languages. The dataset was created as part of the BigCode Project, an open scientific collaboration working on the responsible development of Large Language Models for Code (Code LLMs). The Stack serves as a pre-training dataset for Code LLMs, i.e., code-generating AI systems which enable the synthesis of programs from natural language descriptions as well as other from code snippets. This is the near-deduplicated version with 3TB data.

You can find more details about the DCLM-baseline dataset on the [homepage](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0). 

## Pre-Training

We follow the [ColossalAI](https://github.com/hpcaitech/ColossalAI) framework to train the LLM model. Colossal-AI provides a collection of parallel components for the training. It aims to support   to write the distributed deep learning models just like how you write your model on your laptop. It provides user-friendly tools to kickstart distributed training and inference in a few lines. 

We provide a few examples to show how to run benchmark or pretraining based on Colossal-AI. 

### 1. Training LLM

You can find the shell scripts in 'scripts/train_7B' directory. The main command should be in the format of:
```
colossalai run --nproc_per_node YOUR_GPU_PER_NODE --hostfile YOUR_HOST_FILE \
benchmark.py --OTHER_CONFIGURATIONS
```

#### a. Running on a sinlge node
we provide an example to run the training on a single node as below,
```
colossalai run --nproc_per_node 1 pretrain.py \
        --config 7b \
        --dataset togethercomputer/RedPajama-Data-1T-Sample \
        --batch_size 1 \
        --num_epochs 5 \
        --save_interval 5000 \
        --max_length 2048 \
        --save_dir output-checkpoints \
        --plugin zero2_cpu \
        --lr 2e-5 \
        --expanded_model hpcai-tech/Colossal-LLaMA-2-7b-base
```
In the example, it uses the sample dataset 'togethercomputer/RedPajama-Data-1T-Sample' for training. It trains the 7B model 'hpcai-tech/Colossal-LLaMA-2-7b-base'. You can refer the main file 'run.sh' and 'pretrain.py' for more details. To start the training, run the following, 
```bash
bash run.sh
```

#### b. Running on a sinlge node

we provide an example to run the training on multiple nodes as below,
```
srun colossalai run --num_nodes 8 --nproc_per_node 8 pretrain.py \
        --config 7b \
        --dataset cerebras/SlimPajama-627B \
        --batch_size 1 \
        --num_epochs 10 \
        --save_interval 50000 \
        --max_length 2048 \
        --save_dir output-checkpoints \
        --flash_attention \
        --plugin zero2_cpu \
        --lr 1e-5 \
        --expanded_model hpcai-tech/Colossal-LLaMA-2-7b-base
```
It uses 8 nodes. Put your host file (`hosts.txt`) in this directory with your real host ip or host name.
Here is a sample `hosts.txt`:
```text
hostname1
hostname2
hostname3
...
hostname8
```
You can refer to   the main file 'run-multi-server.sh' and 'pretrain.py' for more details. To start the training, run the following, 

```bash
bash run-multi-server.sh
```

### 2. Benchmark


You can find the shell scripts in 'scripts/benchmark_7B' directory. The benchmark mainly test the throughput of the LLM, without actual model training.  The main command should be in the format of:
```
colossalai run --nproc_per_node YOUR_GPU_PER_NODE --hostfile YOUR_HOST_FILE \
benchmark.py --OTHER_CONFIGURATIONS
```

Here we will show an example of how to run training llama pretraining with 'gemini, batch_size=16, sequence_length=4096, gradient_checkpoint=True, flash_attn=True'.

#### a. Running environment

This experiment was performed on 4 computing nodes with 32 L40S GPUs in total for LLaMA-2 7B. The nodes are connected with RDMA and GPUs within one node are fully connected with NVLink. 

#### b. Running command

```bash
cd scripts/benchmark_7B
```

First, put your host file (`hosts.txt`) in this directory with your real host ip or host name.

Here is a sample `hosts.txt`:
```text
hostname1
hostname2
hostname3
hostname4
```

Then add environment variables to script if needed.

Finally, run the following command to start training:

```bash
bash gemini.sh
```


## Citation

```
@article{zhao2024fully,
  title={Fully Open Source Moxin-7B Technical Report},
  author={Zhao, Pu and Shen, Xuan and Kong, Zhenglun and Shen, Yixin and Chang, Sung-En and Rupprecht, Timothy and Lu, Lei and Nan, Enfu and Yang, Changdi and He, Yumei and others},
  journal={arXiv preprint arXiv:2412.06845},
  year={2024}
}
```




