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

from typing import Optional

import torch

from warprnnt_numba.rnnt_loss.utils import global_constants
from warprnnt_numba.rnnt_loss.utils.cuda_utils import gpu_rnnt_kernel, gpu_rnnt
from warprnnt_numba.rnnt_loss.rnnt_atomic_locks.utils import naive_gpu_rnnt_kernel


class GPURNNTAtomicLocks(gpu_rnnt.GPURNNT):
    def __init__(
        self,
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        workspace,
        blank: int,
        fastemit_lambda: float,
        clamp: float,
        num_threads: int,
        stream,
    ):
        """
        Helper class to launch the CUDA Kernels to compute the Transducer Loss.

        Args:
            minibatch: Int representing the batch size.
            maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
            maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
            alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
            workspace: An allocated chunk of memory that will be sliced off and reshaped into required
                blocks used as working memory.
            blank: Index of the RNNT blank token in the vocabulary. Generally the first or last token in the vocab.
            fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
                FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
            clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].
            num_threads: Number of OMP threads to launch.
            stream: Numba Cuda Stream.
        """
        super().__init__(
            minibatch=minibatch,
            maxT=maxT,
            maxU=maxU,
            alphabet_size=alphabet_size,
            workspace=workspace,
            blank=blank,
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            num_threads=num_threads,
            stream=stream
        )

    def compute_cost_and_score(
        self,
        acts: torch.Tensor,
        grads: Optional[torch.Tensor],
        costs: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> global_constants.RNNTStatus:
        """
        Compute both the loss and the gradients.

        Args:
            acts: A flattened tensor of shape [B, T, U, V+1] representing the activation matrix.
            grad: A flattented zero tensor of same shape as acts.
            costs: A zero vector of length B which will be updated inplace with the log probability costs.
            flat_labels: A flattened matrix of labels of shape [B, U]
            label_lengths: A vector of length B that contains the original lengths of the acoustic sequence.
            input_lengths: A vector of length B that contains the original lengths of the target sequence.

        Updates:
            This will launch kernels that will update inline the following variables:
            -   grads: Gradients of the activation matrix wrt the costs vector.
            -   costs: Negative log likelihood of the forward variable.

        Returns:
            An enum that either represents a successful RNNT operation or failure.
        """
        training = grads is not None

        if training:
            grads *= 0.0  # zero grads

        used_offset, (denom, alphas, betas, llForward, llBackward) = self._prepare_workspace()

        lock = torch.zeros(
            (self.minibatch_, self.maxU_), dtype=torch.int32, device=acts.device
        )

        ######## START EXECUTION ########
        self.log_softmax(acts, denom)

        # Compute alphas
        naive_gpu_rnnt_kernel.compute_alphas_kernel_atomic_locks[self.minibatch_, self.maxU_, self.stream_, 0](
            acts,
            denom,
            alphas,
            llForward,
            input_lengths,
            label_lengths,
            labels,
            self.minibatch_,
            self.maxT_,
            self.maxU_,
            self.alphabet_size_,
            self.blank_,
            lock,
        )

        if training:
            # Compute betas
            lock *= 0

            naive_gpu_rnnt_kernel.compute_betas_kernel_atomic_locks[self.minibatch_, self.maxU_, self.stream_, 0](
                acts,
                denom,
                betas,
                llBackward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
                lock,
            )

            # Compute gradient
            grad_blocks_per_grid = self.minibatch_ * self.maxT_ * self.maxU_
            grad_threads_per_block = naive_gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
            gpu_rnnt_kernel.compute_grad_kernel[grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0](
                grads,
                acts,
                denom,
                alphas,
                betas,
                llForward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
                self.fastemit_lambda_,
                self.clamp_,
            )

        # // cost
        costs.copy_to_device(llForward, stream=self.stream_)
        self.stream_.synchronize()

        # compute negative log likelihood.
        for mb in range(self.minibatch_):
            # Scale llForward by FastEmit lambda
            costs[mb] = -costs[mb]
            costs[mb] = (1.0 + self.fastemit_lambda_) * costs[mb]

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS
