# Moxin LLM : A Family of Fully Open-Source and Reproducible LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2412.06845-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2412.06845v5)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](https://github.com/moxin-org/Moxin-LLM/blob/main/LICENSE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-moxin--org-yellow.svg?style=for-the-badge&logo=HuggingFace)](https://huggingface.co/moxin-org)

---

## Introduction

Generative AI (GAI) offers unprecedented opportunities for research and innovation, but its commercialization has raised concerns about transparency, reproducibility, and safety. Many open GAI models lack the necessary components for full understanding and reproducibility, and some use restrictive licenses whilst claiming to be “open-source”. To address these concerns, we follow the [Model Openness Framework (MOF)](https://arxiv.org/pdf/2403.13784), a ranked classification system that rates machine learning models based on their completeness and openness, following principles of open science, open source, open data, and open access. 

By promoting transparency and reproducibility, the MOF combats “openwashing” practices and establishes completeness and openness as primary criteria alongside the core tenets of responsible AI. Wide adoption of the MOF will foster a more open AI ecosystem, benefiting research, innovation, and adoption of state-of-the-art models. 

In line with the MOF, we release our datasets used during training, the training scripts, and the trained models. 


### Quick Start
- [Usage Guide](inference) - Inference code with Pytorch.
- [Quantization and Deployment](llamacpp) - Implementation and Inference using Llama.cpp quantized models.

### Documentation
- [Pre-Training Guide](train) - Complete training documentation
- [Post-Training Guide](finetune) - Post-Training with Tülu 3 and Reinforcement Learning with GRPO
- [Evaluation](benchmark) - Benchmarking and evaluation


## Model
You can download our  [Moxin-7B-Base](https://huggingface.co/moxin-org/moxin-llm-7b), [Moxin-7B-Instruct](https://huggingface.co/moxin-org/moxin-instruct-7b), [Moxin-7B-Reasoning](https://huggingface.co/moxin-org/moxin-reasoning-7b) and [Moxin-7B-VLM](https://huggingface.co/moxin-org/Moxin-7B-VLM) models. 
 

## Model Family Overview

### Base Model

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



### Instruct Model

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


### Reasoning Model

Our reasoning model is trained with [DeepScaleR](https://github.com/agentica-project/rllm). The evaluation on math datasets are demonstrated below. 

|Models/Datasets |MATH 500 |AMC |Minerva Math |OlympiadBench |Ave|
|:-----------------:	|:---------:	|:---------:	|:-----:	|:-----:	|:-----:	|
|Qwen2.5-Math-7B-Base |52.4 |52.5 |12.9 |16.4| 33.55|
|Qwen2.5-Math-7B-Base + 8K MATH SFT |54.6 |22.5| 32.7| 19.6| 32.35|
|Llama-3.1-70B-Instruct| 64.6 |30.1 |35.3| 31.9| 40.48|
|Moxin-7B-RL-DeepScaleR| 68 |57.5 |16.9| 30.4 |43.2|


### VLM Model

Our VLM model is trained with [prismatic-vlms](https://github.com/TRI-ML/prismatic-vlms). The evaluation is demonstrated below. 

|                          	|  GQA  	| VizWiz 	| RefCOCO+ 	| OCID-Ref 	|  VSR  	|  POPE 	| TallyQA 	|  Ave. 	|
|--------------------------	|:-----:	|:------:	|:--------:	|:--------:	|:-----:	|:-----:	|:-------:	|:-----:	|
| LLaVa v1.5 7B (Base)     	| 61.58 	|  54.25 	|   49.47  	|   35.07  	| 51.47 	| 86.57 	|  62.06  	| 57.21 	|
| Llama-2 Chat 7B          	| 62.11 	|  56.39 	|   58.5   	|   46.3   	|  61.8 	|  86.8 	|   58.1  	| 61.43 	|
| Mistral v0.1 7B          	|  63.3 	|  55.32 	|   65.1   	|   48.8   	|  58.5 	|  87.1 	|   61.7  	| 62.83 	|
| Mistral Instruct v0.1 7B 	| 62.71 	|  54.35 	|   64.9   	|    48    	|  57.8 	|  87.5 	|   64.5  	| 62.82 	|
| Llama-2 7B               	| 62.44 	|  55.98 	|   59.47  	|   43.89  	| 63.67 	| 86.74 	|  59.22  	| 61.63 	|
| Ours                     	| 64.88 	|  54.08 	|   71.3   	|   48.4   	|  60.8 	|  87.3 	|    66   	| 64.68 	|


## Citation

```
@article{zhao2024fully,
  title={Fully Open Source Moxin-7B Technical Report},
  author={Zhao, Pu and Shen, Xuan and Kong, Zhenglun and Shen, Yixin and Chang, Sung-En and Rupprecht, Timothy and Lu, Lei and Nan, Enfu and Yang, Changdi and He, Yumei and others},
  journal={arXiv preprint arXiv:2412.06845},
  year={2024}
}
```




