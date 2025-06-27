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
