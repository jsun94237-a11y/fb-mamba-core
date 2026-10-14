# FB-Mamba Core

This repository provides the core implementation of FB-Mamba in PyTorch.

## Contents

- `fb_mamba.py`: SelectiveSSM, forward-backward fusion block, and encoder.
- `smoke_test.py`: synthetic-input forward, backward, optimizer-step, and checkpoint save/load checks.

## Installation and test

Use Python 3.11. A GPU is not required.

```bash
python -m venv .venv
```

Activate the environment:

```bat
:: Windows Command Prompt
.venv\Scripts\activate.bat
```

```bash
# Linux
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python smoke_test.py
```

The three tests should finish with `OK`.

## Usage

```python
import torch
from fb_mamba import FBMambaEncoder

encoder = FBMambaEncoder(input_channels=4)
encoder.eval()
with torch.no_grad():
    features = encoder(torch.randn(2, 32, 4))
print(features.shape)  # torch.Size([2, 256])
```

Input shape: `(batch, sequence_length, input_channels)`.
Output shape: `(batch, d_model)`.
