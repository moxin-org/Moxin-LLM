## Usage with llama.cpp
Example usage of quantized models with llama.cpp backend (supports GGUF format).

### Getting appropriate quantized version from huggingface
```bash
pip install huggingface_hub hf_transfer  # hf_transfer optional: speeds up downloads

python snapshot_download.py

# Alternative: use huggingface-cli
huggingface-cli download second-state/moxin-instruct-7b-GGUF moxin-instruct-7b-Q6_K.gguf

```

### Building llama.cpp on your local device

```bash
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp

# for linux users
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON  # DLLAMA_CURL=ON/OFF
cmake --build build --config Release -j --clean-first

# for mac & cpu only users
cmake -B build -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j --clean-first

```

### Run with llama-cli or llama-server option
```bash

build/bin/llama-cli -m <PATH_TO_GGUF> -ngl 99 -sys <SYSTEM_PROMPT>  # -ngl 0 if cpu-only

build/bin/llama-server -m <PATH_TO_GGUF> -ngl 99 

```
