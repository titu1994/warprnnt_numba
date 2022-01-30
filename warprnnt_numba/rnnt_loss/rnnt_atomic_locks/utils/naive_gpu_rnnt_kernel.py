# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
from numba import cuda

from warprnnt_numba.rnnt_loss.utils import rnnt_helper
from warprnnt_numba.rnnt_loss.utils.cuda_utils.gpu_rnnt_kernel import logp

GPU_RNNT_THREAD_SIZE = 256


@cuda.jit(device=True, inline=True)
def logp(
    denom: torch.Tensor, acts: torch.Tensor, maxT: int, maxU: int, alphabet_size: int, mb: int, t: int, u: int, v: int
):
    """
    Compute the sum of log probability from the activation tensor and its denominator.

    Args:
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the logprobs activation tensor
            across entire vocabulary.
        acts: Tensor of shape [B, T, U, V+1] flattened. Represents the logprobs activation tensor.
        maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        mb: Batch indexer.
        t: Acoustic sequence timestep indexer.
        u: Target sequence timestep indexer.
        v: Vocabulary token indexer.

    Returns:
        The sum of logprobs[mb, t, u, v] + denom[mb, t, u]
    """
    col = (mb * maxT + t) * maxU + u
    return denom[col] + acts[col * alphabet_size + v]


@cuda.jit()
def compute_alphas_kernel_atomic_locks(
        acts: torch.Tensor,
        denom: torch.Tensor,
        alphas: torch.Tensor,
        llForward: torch.Tensor,
        xlen: torch.Tensor,
        ylen: torch.Tensor,
        mlabels: torch.Tensor,  # [B]
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        blank: int,
        lock: torch.Tensor,
):
    """
    Compute alpha (forward variable) probabilities over the transduction step in loop,
    with CUDA atomic locks.

    Baseline reference from SpeechBrain implementation -
    https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/nnet/loss/transducer_loss.py

    Args:
        acts: Tensor of shape [B, T, U, V+1]. Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the logprobs activation tensor
            across entire vocabulary.
        alphas: Zero tensor of shape [B, T, U]. Will be updated inside the kernel with the forward variable
            probabilities.
        llForward: Zero tensor of shape [B]. Represents the log-likelihood of the forward pass.
            Returned as the forward pass loss that is reduced by the optimizer.
        xlen: Vector of length B which contains the actual acoustic sequence lengths in the padded
            activation tensor.
        ylen: Vector of length B which contains the actual target sequence lengths in the padded
            activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token - usually the RNNT blank).
            The matrix contains the padded target transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT blank token in the vocabulary. Generally the first or last token in the vocab.
        lock: Tensor of shape [B, U+1]. Bool values, used for CUDA atomic locking.

    Updates:
        Kernel inplace updates the following inputs:
        -   alphas: forward variable scores.
        -   llForward: log-likelihood of forward variable.
    """
    #  // launch B blocks, each block has U threads
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x

    T = xlen[b]  # select AM length of current sample
    U = ylen[b] + 1  # select target length of current sample, +1 for the blank token

    labels: torch.Tensor = mlabels[b]  # mb label start point, equivalent to mlabels + b * (maxU - 1)
    offset = b * maxT * maxU  # pointer indexing offset

    t = 0

    # Ordinary alpha calculations, broadcast across B=b and U=u
    # Look up forward variable calculation from rnnt_numpy.forward_pass()
    if u < U:
        # for each (B,U) Thread
        # wait the unlock of the previous computation of Alpha[b,U-1,:]
        # Do the computation over the whole Time sequence on alpha[B,U,:]
        # and then unlock the target U+1 for computation
        while t < T:
            if u == 0:
                # for t in range(1, T) step to initialize alphas[b, t, 0]
                if t > 0:
                    alphas[offset + t * maxU] = (
                            alphas[offset +  (t - 1) * maxU]
                            + logp(denom, acts, maxT, maxU, alphabet_size, b, t - 1, 0, blank)
                    )
                cuda.atomic.add(lock, (b, u + 1), -1)
                t += 1
            else:
                if cuda.atomic.add(lock, (b, u), 0) < 0:
                    # for u in range(1, U) step to initialize alphas[b, 0, u]
                    if t == 0:
                        alphas[offset + u] = (
                                alphas[offset + u - 1]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, 0, u - 1, labels[u - 1])
                        )
                    else:
                        # for t in range(1, T) for u in range(1, U) step to compute alphas[b, t, u]
                        emit = (
                                alphas[offset + t * maxU + u - 1]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u - 1, labels[u - 1])
                        )
                        no_emit = (
                                alphas[offset + (t - 1) * maxU + u]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, t - 1, u, blank)
                        )

                        alphas[offset + t * maxU + u] = max(no_emit, emit) + math.log1p(
                            math.exp(-abs(no_emit - emit))
                        )

                    if u < U:
                        cuda.atomic.add(lock, (b, u + 1), -1)

                    cuda.atomic.add(lock, (b, u), 1)
                    t += 1

        # After final sync, alphas[b, T-1, U - 1] + logprobs[b, T-1, U-1, blank] + denom[b, T-1, U-1] gives
        # log-likelihood of forward pass.
        if u == U - 1:
            # for each thread b (utterance)
            llForward[b] = (
                    alphas[offset + (T - 1) * maxU + U - 1]
                    + logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank)
            )


@cuda.jit()
def compute_betas_kernel_atomic_locks(
        acts: torch.Tensor,
        denom: torch.Tensor,
        betas: torch.Tensor,
        llBackward: torch.Tensor,
        xlen: torch.Tensor,
        ylen: torch.Tensor,
        mlabels: torch.Tensor,  # [B, U]
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        blank_: int,
        lock: torch.Tensor,
):
    """
    Compute beta (backward variable) probabilities over the transduction step.

    Args:
        acts: Tensor of shape [B, T, U, V+1] flattened. Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the logprobs activation tensor
            across entire vocabulary.
        betas: Zero tensor of shape [B, T, U]. Will be updated inside the kernel with the backward variable
            probabilities.
        llBackward: Zero tensor of shape [B]. Represents the log-likelihood of the backward pass.
            Returned as the backward pass loss that is reduced by the optimizer.
        xlen: Vector of length B which contains the actual acoustic sequence lengths in the padded
            activation tensor.
        ylen: Vector of length B which contains the actual target sequence lengths in the padded
            activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token - usually the RNNT blank).
            The matrix contains the padded target transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT blank token in the vocabulary. Generally the first or last token in the vocab.
        lock: Tensor of shape [B, U+1]. Bool values, used for CUDA atomic locking.

    Updates:
        Kernel inplace updates the following inputs:
        -   betas: backward variable scores.
        -   llBackward: log-likelihood of backward variable.
    """
    # // launch B blocks, each block has U threads
    b = cuda.blockIdx.x  # // batch id
    u = cuda.threadIdx.x  # label id, u

    T = xlen[b]  # select AM length of current sample
    U = ylen[b] + 1  # select target length of current sample, +1 for the blank token

    offset = b * maxT * maxU  # pointer indexing offset
    labels: torch.Tensor = mlabels[b]  # mb label start point, equivalent to mlabels + b * (maxU - 1)

    # Initilize beta[b, t=T-1, u=U-1] for all b in B with log_probs[b, t=T-1, u=U-1, blank]
    t = T - 1

    # Ordinary beta calculations, broadcast across B=b and U=u
    # Look up backward variable calculation from rnnt_numpy.backward_pass()
    if u < U:
        # Loop in reversed order of T - (reversed(range(T - 1)) steps
        while t >= 0:
            if u == U - 1:
                # Initilize beta[b, t=T-1, u=U-1] for all b in B with log_probs[b, t=T-1, u=U-1, blank]
                if t == T - 1:
                    betas[offset + t * maxU + u] = logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_)

                else:
                    # for t in reversed(range(T - 1)) step to initialize betas[b, t, U-1]
                    betas[offset + t * maxU + u] = (
                            betas[offset + (t + 1) * maxU + u]
                            + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_)
                    )

                cuda.atomic.add(lock, (b, u - 1), -1)
                t -= 1

            else:
                if cuda.atomic.add(lock, (b, u), 0) < 0:
                    if t == T - 1:
                        # Compute betas[b, T - 1, u]
                        betas[offset + t * maxU + u] = (
                                betas[offset + t * maxU + u + 1]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u])
                        )
                    else:
                        # for t in reversed(range(T - 1)) for u in reversed(range(U - 1)) step to compute betas[b, t, u]
                        emit = (
                                betas[offset + t * maxU + u + 1]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u])
                        )
                        no_emit = (betas[offset + (t + 1) * maxU + u]
                                   + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_))

                        betas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)

                    if u > 0:
                        cuda.atomic.add(lock, (b, u - 1), -1)

                    cuda.atomic.add(lock, (b, u), 1)
                    t -= 1

    # After final sync, betas[b, 0, 0] gives
    # log-likelihood of backward pass.
    if u == 0:
        llBackward[b] = betas[offset]
