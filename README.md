# `bitsandbytes`

This is a ROCM port of the bitsandbytes package. Early version of this port has been attempted but they were poorly maintained and thus, no longer application starting from ROCM 6.0. This repo aims at resolving this pain point and bring the ability to fine tune LLM to AMD GPU, starting with the gfx 1100 family (7900 XTX, 7900 XT, 7900 GRE)

Best effort has been made to to ensure the code runs as well as possible. However, feel free to open an issue or make a PR if such a need arises.

The current repo has been compiled and test run on the following system config

Intel i7-13700K
7900 XTX 24 GB
64GB system memory

## Installation
Clone this repo or download the source code
Create virtual environment
conda create -n mypython39-rocm python=3.9
Install torch-rocm 6.0
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0
Install this packge 
python3 -m pip install ./bitsandbytes.tar.gz 

# Original text from bitsandbytes repo

The `bitsandbytes` library is a lightweight Python wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and 8 & 4-bit quantization functions.

The library includes quantization primitives for 8-bit & 4-bit operations, through `bitsandbytes.nn.Linear8bitLt` and `bitsandbytes.nn.Linear4bit` and 8-bit optimizers through `bitsandbytes.optim` module.

There are ongoing efforts to support further hardware backends, i.e. Intel CPU + GPU, AMD GPU, Apple Silicon. Windows support is quite far along and is on its way as well.

**Please head to the official documentation page:**

**[https://huggingface.co/docs/bitsandbytes/main](https://huggingface.co/docs/bitsandbytes/main)**

## License

The majority of bitsandbytes is licensed under MIT, however small portions of the project are available under separate license terms, as the parts adapted from Pytorch are licensed under the BSD license.

We thank Fabio Cannizzo for his work on [FastBinarySearch](https://github.com/fabiocannizzo/FastBinarySearch) which we use for CPU quantization.
