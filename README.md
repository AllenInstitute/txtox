### Overview

This repo contains code for training and evaluating models on spatial transcriptomics data.

### Environment

[txtox-torch-gpu]
```
conda create -n txtox-torch-gpu python==3.12
conda activate txtox-torch-gpu
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric
pip install lightning mlflow tensorboard
pip install -e .
pip install -e .[dev] 
```

### Config

Data paths are in `config.toml` at the same location as `pyproject.toml`.

```toml
data_root = "/txtox/data/" # replace with your own path
```
