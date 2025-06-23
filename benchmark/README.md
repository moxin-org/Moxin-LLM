## Benchmark

### 1. Installation

Please install the latest ColossalAI from source.

```bash
BUILD_EXT=1 pip install -U git+https://github.com/hpcaitech/ColossalAI
```

Then install other dependencies.

```bash
pip install -r requirements.txt
```

### 4. Shell Script Examples

For your convenience, we provide some shell scripts to run benchmark with various configurations.

You can find them in `scripts/benchmark_7B` and `scripts/benchmark_70B` directory. The main command should be in the format of:
```bash
colossalai run --nproc_per_node YOUR_GPU_PER_NODE --hostfile YOUR_HOST_FILE \
benchmark.py --OTHER_CONFIGURATIONS
```
Here we will show an example of how to run training
llama pretraining with `gemini, batch_size=16, sequence_length=4096, gradient_checkpoint=True, flash_attn=True`.

#### a. Running environment
This experiment was performed on 4 computing nodes with 32 A800/H800 80GB GPUs in total for LLaMA-1 65B or LLaMA-2 70B. The nodes are
connected with RDMA and GPUs within one node are fully connected with NVLink.

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

If you encounter out-of-memory(OOM) error during training with script `gemini.sh`, changing to script `gemini_auto.sh` might be a solution, since gemini_auto will set a upper limit on GPU memory usage through offloading part of the model parameters and optimizer states back to CPU memory. But there's a trade-off: `gemini_auto.sh` will be a bit slower, since more data are transmitted between CPU and GPU.
