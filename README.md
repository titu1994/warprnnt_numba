# RNNT loss in Pytorch - Numba JIT compiled (warprnnt_numba) [![Test-CPU](https://github.com/titu1994/warprnnt_numba/actions/workflows/CI-CPU.yml/badge.svg)](https://github.com/titu1994/warprnnt_numba/actions/workflows/CI-CPU.yml)

Warp RNN Transducer Loss for ASR in Pytorch, ported from [HawkAaron/warp-transducer](https://github.com/HawkAaron/warp-transducer) and a replica of the stable version in NVIDIA Neural Module repository ([NVIDIA NeMo](https://github.com/NVIDIA/NeMo)).

NOTE: The code here will have experimental extensions and may be potentially unstable, use the version in NeMo for long term supported loss version of RNNT for PyTorch.

# Supported Features

Currently supports :

1) WarpRNNT loss in pytorch for CPU / CUDA (jit compiled)
2) [FastEmit](https://arxiv.org/abs/2010.11148)
3) Gradient Clipping (from Torch Audio)

# Installation
You will need PyTorch (usually the latest version should be used), plus installation of [Numba](https://numba.pydata.org/) in a Conda environment (pip only environment is untested but may work).

```
# Follow installation instructions to install pytorch from website (with cuda if required)
conda install -c conda-force numba or conda update -c conda-forge numba (to get latest version)

# Then install this library
pip install --upgrade git+https://github.com/titu1994/warprnnt_numba.git
```

# Usage

Import `warprnnt_numba` and use `RNNTLossNumba`. If attempting to use CUDA version of loss, it is advisable to test that your installed CUDA version is compatible with numba version using `numba_utils`.

There is also included a very slow numpy/pytorch explicit-loop based loss implementation for verification of exact correct results.


```python
import torch
import numpy as np
import warprnnt_numba

# Define the loss function
fastemit_lambda = 0.001  # any float >= 0.0
loss_pt = warprnnt_numba.RNNTLossNumba(blank=4, reduction='sum', fastemit_lambda=fastemit_lambda)

# --------------
# Example usage

device = "cuda"
torch.random.manual_seed(0)

# Assume Batchsize=2, Acoustic Timesteps = 8, Label Timesteps = 5 (including BLANK=BOS token),
# and Vocabulary size of 5 tokens (including RNNT BLANK)
acts = torch.randn(2, 8, 5, 5, device=device, requires_grad=True)
sequence_length = torch.tensor([5, 8], dtype=torch.int32,
                               device=device)  # acoustic sequence length. One element must be == acts.shape[1].

# Let 0 be MASK/PAD value, 1-3 be token ids, and 4 represent RNNT BLANK token
# The BLANK token is overloaded for BOS token as well here, but can be different token.
# Let first sample be padded with 0 (actual length = 3). Loss is computed according to supplied `label_lengths`.
# and gradients for the 4th index onwards (0 based indexing).
labels = torch.tensor([[4, 1, 1, 3, 0], [4, 2, 2, 3, 1]], dtype=torch.int32, device=device)
label_lengths = torch.tensor([3, 4], dtype=torch.int32,
                             device=device)  # Lengths here must be WITHOUT the BOS token.

# If on CUDA, log_softmax is computed internally efficiently (preserving memory and speed)
# Compute it explicitly for CPU, this is done automatically for you inside forward() of the loss.
# -1-th vocab index is RNNT blank token here.
loss_func = warprnnt_numba.RNNTLossNumba(blank=4, reduction='none',
                                         fastemit_lambda=0.0, clamp=0.0)
loss = loss_func(acts, labels, sequence_length, label_lengths)
print("Loss :", loss)
loss.sum().backward()

# When parsing the gradients, look at grads[0] -
# Since it was padded in T (sequence_length=5 < T=8), there are gradients only for grads[0, :5, :, :].
# Since it was padded in U (label_lengths=3+1 < U=5), there are gradeints only for grads[0, :5, :3+1, :].
grads = acts.grad
print("Gradients of activations :")
print(grads)
```

# Tests

Tests will perform CPU only checks if there are no GPUs. If GPUs are present, will run all tests once for `cuda:0` as well.

```bash
pytest tests/
```

# Requirements

- pytorch >= 1.10. Older versions might work, not tested.
- numba - Minimum required version is 0.53.0, preferred is 0.54+.
