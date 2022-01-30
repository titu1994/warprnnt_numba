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

import multiprocessing

import torch
from numba import cuda

from warprnnt_numba.rnnt_loss.utils import global_constants, rnnt_helper
from warprnnt_numba.rnnt_loss.rnnt_atomic_locks.utils import naive_gpu_rnnt
from warprnnt_numba.rnnt_loss.rnnt import rnnt_loss_cpu


def rnnt_loss_gpu(
    acts: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    costs: torch.Tensor,
    grads: torch.Tensor,
    blank_label: int,
    fastemit_lambda: float,
    clamp: float,
    num_threads: int,
):
    """
    Wrapper method for accessing GPU RNNT loss.

    CUDA implementation ported from [HawkAaron/warp-transducer](https://github.com/HawkAaron/warp-transducer).

    Args:
        acts: Activation tensor of shape [B, T, U, V+1].
        labels: Ground truth labels of shape [B, U].
        input_lengths: Lengths of the acoustic sequence as a vector of ints [B].
        label_lengths: Lengths of the target sequence as a vector of ints [B].
        costs: Zero vector of length [B] in which costs will be set.
        grads: Zero tensor of shape [B, T, U, V+1] where the gradient will be set.
        blank_label: Index of the blank token in the vocabulary.
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
        clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].
        num_threads: Number of threads for OpenMP.
    """
    minibatch_size = acts.shape[0]
    maxT = acts.shape[1]
    maxU = acts.shape[2]
    alphabet_size = acts.shape[3]

    if hasattr(cuda, 'external_stream'):
        stream = cuda.external_stream(torch.cuda.current_stream(acts.device).cuda_stream)
    else:
        stream = cuda.default_stream()

    if num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    num_threads = max(1, num_threads)  # have to use at least 1 thread

    gpu_size, status = rnnt_helper.get_workspace_size(maxT, maxU, minibatch_size, gpu=True)
    if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
        raise RuntimeError("Invalid parameter passed when calculating working space memory")

    # Select GPU index
    cuda.select_device(acts.device.index)
    gpu_workspace = torch.zeros(gpu_size, device=acts.device, dtype=acts.dtype, requires_grad=False)

    ### VIEW TENSORS AS VECTORS FOR POINTER INDEXING ###
    acts, acts_shape = rnnt_helper.flatten_tensor(acts)

    wrapper = naive_gpu_rnnt.GPURNNTAtomicLocks(
        minibatch=minibatch_size,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=alphabet_size,
        workspace=gpu_workspace,
        blank=blank_label,
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
        num_threads=num_threads,
        stream=stream,
    )

    if grads is None:
        status = wrapper.score_forward(
            acts=acts.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    else:
        ### FLATTEN GRAD TENSOR ###
        grads, grads_shape = rnnt_helper.flatten_tensor(grads)

        status = wrapper.cost_and_grad(
            acts=acts.data,
            grads=grads.data,
            costs=costs.data,
            pad_labels=labels.data,
            label_lengths=label_lengths.data,
            input_lengths=input_lengths.data,
        )

        if status != global_constants.RNNTStatus.RNNT_STATUS_SUCCESS:
            raise RuntimeError("Could not calculate forward scores")

    del gpu_workspace, wrapper
    return True