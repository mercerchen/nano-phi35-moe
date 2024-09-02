# nano-phi35-moe

# Main Objectives

1. Re-write the code to run with only PyTorch. Break up the code into smaller pieces so it is easier for learning.
2. Create a mini-version of the model that can be used for learning purposes.

# Create a new condo environment

```bash
conda create -n torch python=3.11
conda activate torch
pip install -r requirements.txt
```

# Download the model

I prefer using git lfs. If you don't have it, you can run `sudo apt-get install git-lfs` to install it.

```bash
git lfs install
git clone https://huggingface.co/microsoft/Phi-3.5-MoE-instruct
```