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
# low level library used by PyG: pyg-lib
pip install https://data.pyg.org/whl/torch-2.2.0%2Bcu121.html
pip install -e .
pip install -e .[dev] 
```

### Config

Data paths are in `config.toml` at the same location as `pyproject.toml`.

```toml
data_root = "/txtox/data/" # replace with your own path
```

### Development

Handy commands for formatting etc.

```bash
ruff format <file-name>
pre-commit run --all-files
```