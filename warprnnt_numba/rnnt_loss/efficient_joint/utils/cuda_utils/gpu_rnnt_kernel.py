import math

import torch
from numba import cuda

from warprnnt_numba.rnnt_loss.utils import rnnt_helper

# Index of the log_probs tensor
CACHE_IDX_BLANK = 0
CACHE_IDX_SYMBOL = 1

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
def compute_log_probs(
    denom: torch.Tensor,
    acts: torch.Tensor,
    log_probs: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B]
    row_splits: torch.Tensor,
    row_ids: torch.Tensor,
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
    num_elements: int,
):
    idx01 = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if idx01 >= num_elements:  # Out of boundary
        return

    b = row_ids[idx01]  # batch size

    # // +1 since it is prepended with a blank
    U_p1 = ylen[b] + 1
    offset = row_splits[b]
    idx1 = idx01 - offset

    u = idx1 % U_p1

    p_logits = acts[idx01 * alphabet_size]
    p_denominator = denom[idx01]
    p_targets = mlabels[b]  # * targets_col

    d = p_denominator

    p_log_probs = idx01 * 2
    log_probs[p_log_probs, CACHE_IDX_BLANK] = p_logits[blank_] - d

    if u < U_p1 - 1:
        log_probs[p_log_probs, CACHE_IDX_SYMBOL] = p_logits[p_targets[u]] - d


@cuda.jit()
def compute_alphas_kernel(
    log_probs: torch.Tensor,
    alphas: torch.Tensor,
    llForward: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B]
    row_splits: torch.Tensor,
    row_ids: torch.Tensor,
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
):
    """
    Compute alpha (forward variable) probabilities over the transduction step.

    Args:
        acts: Tensor of shape [B, T, U, V+1] flattened. Represents the logprobs activation tensor.
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

    Updates:
        Kernel inplace updates the following inputs:
        -   alphas: forward variable scores.
        -   llForward: log-likelihood of forward variable.
    """
    # // launch B blocks, each block has U threads
    b = cuda.blockIdx.x  # // batch id
    u = cuda.threadIdx.x  # label id, u
    T = xlen[b]  # select AM length of current sample
    U = ylen[b] + 1  # select target length of current sample, +1 for the blank token

    labels: torch.Tensor = mlabels[b]  # mb label start point, equivalent to mlabels + b * (maxU - 1)
    offset = row_splits[b]  # pointer indexing offset

    p_alpha = offset
    p_log_probs = offset * 2

    # Initilize alpha[b, t=0, u=0] for all b in B
    if u == 0:
        alphas[offset] = 0

    # sync until all alphas are initialized
    cuda.syncthreads()

    # Ordinary alpha calculations, broadcast across B=b and U=u
    # Look up forward variable calculation from rnnt_numpy.forward_pass()
    for n in range(1, T + U - 1):
        t = n - u

        p_alpha_t = p_alpha + t * U
        p_alpha_t_m1 = p_alpha + (t - 1) * U
        p_log_probs_t = p_log_probs + t * U * 2
        p_log_probs_t_m1 = p_log_probs + (t - 1) * U * 2

        if u == 0:
            # for t in range(1, T) step to initialize alphas[b, t, 0]
            if t > 0 and t < T:
                alphas[p_alpha_t] = alphas[p_alpha_t_m1] + log_probs[p_log_probs_t_m1, CACHE_IDX_BLANK]
        elif u < U:
            # for u in range(1, U) step to initialize alphas[b, 0, u]
            if t == 0:
                alphas[p_alpha + u] = alphas[p_alpha + u - 1] + log_probs[p_log_probs + (u - 1) * 2, CACHE_IDX_SYMBOL]

            # for t in range(1, T) for u in range(1, U) step to compute alphas[b, t, u]
            elif t > 0 and t < T:
                no_emit = alphas[p_alpha_t_m1 + u] + log_probs[p_log_probs_t_m1 + u * 2, CACHE_IDX_BLANK]
                emit = alphas[p_alpha_t_m1 + u - 1] + log_probs[p_log_probs_t_m1 + (u - 1) * 2, CACHE_IDX_SYMBOL]

                alphas[p_alpha_t + u] = rnnt_helper.log_sum_exp(emit, no_emit)

        # sync across all B=b and U=u
        cuda.syncthreads()

    # After final sync, alphas[b, T-1, U - 1] + logprobs[b, T-1, U-1, blank] + denom[b, T-1, U-1] gives
    # log-likelihood of forward pass.
    if u == 0:
        loglike = alphas[offset + T * U - 1] + log_probs[p_log_probs + (T * U - 1) * 2, CACHE_IDX_BLANK]
        llForward[b] = loglike


@cuda.jit()
def compute_betas_kernel(
    log_probs: torch.Tensor,
    betas: torch.Tensor,
    llBackward: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B, U]
    row_splits: torch.Tensor,
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
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

    labels: torch.Tensor = mlabels[b]  # mb label start point, equivalent to mlabels + b * (maxU - 1)
    offset = row_splits[b]  # pointer indexing offset

    p_beta = offset
    p_log_probs = offset * 2

    # Initilize beta[b, t=T-1, u=U-1] for all b in B with log_probs[b, t=T-1, u=U-1, blank]
    if u == 0:  # logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_)
        betas[offset + T * U + (U - 1)] = log_probs[p_log_probs + T * U * 2 - 2, CACHE_IDX_BLANK]

    # sync until all betas are initialized
    cuda.syncthreads()

    # Ordinary beta calculations, broadcast across B=b and U=u
    # Look up backward variable calculation from rnnt_numpy.backward_pass()
    for n in range(T + U - 2, -1, -1):
        t = n - u

        p_beta_t = p_beta + t * U
        p_beta_t_p1 = p_beta + (t + 1) * U
        p_log_probs_t = p_log_probs + t * U * 2

        if u == (U - 1):
            # for t in reversed(range(T - 1)) step to initialize betas[b, t, U-1]
            if t >= 0 and t < (T - 1):
                betas[p_beta_t + U - 1] = (
                    betas[p_beta_t_p1 + U - 1] + log_probs[p_log_probs_t + (U - 1) * 2, CACHE_IDX_BLANK]
                )
        elif u < U:
            if t == T - 1:
                # for u in reversed(range(U - 1)) step to initialize betas[b, T-1, u]
                betas[offset + (T - 1) * U + u] = (
                    betas[offset + (T - 1) * U + u + 1]
                    + log_probs[p_log_probs + ((T - 1) * U + u) * 2, CACHE_IDX_SYMBOL]
                )

                # logp(
                #     denom, acts, maxT, maxU, alphabet_size, b, T - 1, u, labels[u]
                # )
            elif (t >= 0) and (t < T - 1):
                # for t in reversed(range(T - 1)) for u in reversed(range(U - 1)) step to compute betas[b, t, u]
                no_emit = betas[p_beta_t_p1 + u] + log_probs[p_log_probs_t + u * 2, CACHE_IDX_BLANK]
                # logp(
                #     denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_
                # )
                emit = betas[p_beta_t_p1 + u + 1] + log_probs[p_log_probs_t + u * 2, CACHE_IDX_SYMBOL]
                # logp(
                #     denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u]
                # )
                betas[offset + t * U + u] = rnnt_helper.log_sum_exp(emit, no_emit)

        # sync across all B=b and U=u
        cuda.syncthreads()

    # After final sync, betas[b, 0, 0] gives
    # log-likelihood of backward pass.
    if u == 0:
        llBackward[b] = betas[offset]


def compute_grad_kernel(
    grads: torch.Tensor,
    acts: torch.Tensor,
    denom: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    logll: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B, U]
    row_splits: torch.Tensor,
    row_ids: torch.Tensor,
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
    num_elements: int,
    fastemit_lambda: float,
    clamp: float,
):
    """
    Compute gradients over the transduction step.

    Args:
        grads: Zero Tensor of shape [B, T, U, V+1]. Is updated by this kernel to contain the gradients
            of this batch of samples.
        acts: Tensor of shape [B, T, U, V+1] flattened. Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the logprobs activation tensor
            across entire vocabulary.
        alphas: Alpha variable, contains forward probabilities. A tensor of shape [B, T, U].
        betas: Beta varoable, contains backward probabilities. A tensor of shape [B, T, U].
        logll: Log-likelihood of the forward variable, represented as a vector of shape [B].
            Represents the log-likelihood of the forward pass.
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
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
        clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].

    Updates:
        Kernel inplace updates the following inputs:
        -   grads: Gradients with respect to the log likelihood (logll).
    """
    # Kernel call:
    # blocks_per_grid = minibatch (b) * maxT (t) * maxU (u)
    # threads_per_block = constant buffer size of parallel threads (v :: Constant)
    # tid = cuda.threadIdx.x  # represents v, taking steps of some constant size
    # idx = tid  # index of v < V+1; in steps of constant buffer size
    # col = cuda.blockIdx.x  # represents a fused index of b * t * u

    idx01 = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if (idx01 >= num_elements):
        return

    # Decompose original indices from fused `col`
    # u = col % maxU  # (b * t * u) % u = u
    # bt = (col - u) // maxU  # (b * t * u - u) // U = b * t
    # t = bt % maxT  # (b * t) % t = t
    # mb = (bt - t) // maxT  # (b * t - t) // T = b

    b = row_ids[idx01]
    T = xlen[b]
    U = ylen[b] + 1
    offset = row_splits[b]

    idx1 = idx01 - offset  # (b * t * u) - offset(b, 0, 0) = (t * u)
    t = idx1 // U
    u = idx1 % U

    # pointer offsets
    p_acts_t_u = idx01 * alphabet_size
    p_denominator = offset
    p_denominator_t = p_denominator + t * U
    labels = mlabels[b]  # labels = mlabels + mb * (maxU - 1);

    p_alpha = offset
    p_alpha_t = p_alpha + t * U

    p_beta = offset
    p_beta_t = p_beta + t * U
    p_beta_t_p1 = p_beta + (t + 1) * U

    p_grad_t_u = idx01 * alphabet_size

    loss = -1 * betas[p_beta]

    if (math.isinf(loss) or math.isnan(loss)):
        for v in range(alphabet_size):
            grads[p_grad_t_u + v] = 0
        return

    c = alphas[p_alpha_t + u] + loss - denom[p_denominator_t + u]

    if u < U - 1:
        target_u = labels[u]
    else:
        target_u = -1  # not used

    for v in range(alphabet_size):
        g = acts[p_acts_t_u + v] + c
        # val = 0

        if (v == blank_ and t == T - 1 and u == U - 1):
            # last blank transition
            val = math.exp(g + betas[p_beta_t + u]) - math.exp(g)

        elif (v == blank_ and t < T - 1):
            val = math.exp(g + betas[p_beta_t + u]) - math.exp(g + betas[p_beta_t_p1 + u])

        elif (v == target_u and u < U - 1):
            val = math.exp(g + betas[p_beta_t + u]) - math.exp(g + betas[p_beta_t + (u + 1)])

        else:
            val = math.exp(g + betas[p_beta_t + u])

        grads[p_grad_t_u + v] = val

    # constants
    # T = xlen[mb]  # select AM length of current sample
    # U = ylen[mb] + 1  # select target length of current sample, +1 for the blank token
    # labels: torch.Tensor = mlabels[mb]  # labels = mlabels + mb * (maxU - 1);

    # # Buffered gradient calculations, broadcast across B=b, T=t and U=u, looped over V with some constant stride.
    # # Look up gradient calculation from rnnt_numpy.compute_gradient()
    # if t < T and u < U:
    #     # For cuda kernels, maximum number of threads per block is limited to some value.
    #     # However, it may be the case that vocabulary size is larger than this limit
    #     # To work around this, an arbitrary thread buffer size is chosen such that,
    #     # 1) each element within the thread pool operates independently of the other
    #     # 2) An inner while loop moves the index of each buffer element by the size of the buffer itself,
    #     #    such that all elements of the vocabulary size are covered in (V + 1 // thread_buffer) number of steps.
    #     # As such, each thread will perform the while loop at least (V + 1 // thread_buffer) number of times
    #     while idx < alphabet_size:
    #         # remember, `col` represents the tri-index [b, t, u]
    #         # therefore; logpk = denom[b, t, u] + acts[b, t, u, v]
    #         logpk = denom[col] + acts[col * alphabet_size + idx]
    #         # initialize the grad of the sample acts[b, t, u, v]
    #         grad = math.exp(alphas[col] + betas[col] + logpk - logll[mb])
    #
    #         # If FastEmit regularization is enabled, calculate the gradeint of probability of predicting the next label
    #         # at the current timestep.
    #         # The formula for this is Equation 9 in https://arxiv.org/abs/2010.11148, multiplied by the log probability
    #         # of the current step (t, u), normalized by the total log likelihood.
    #         # Once the gradient has been calculated, scale it by `fastemit_lambda`, as in Equation 10.
    #         if fastemit_lambda > 0.0 and u < U - 1:
    #             fastemit_grad = fastemit_lambda * math.exp(
    #                 alphas[col]  # alphas(t, u)
    #                 + (denom[col] + acts[col * alphabet_size + labels[u]])  # y_hat(t, u)
    #                 + betas[col + 1]  # betas(t, u+1)
    #                 + logpk  # log Pr(k|t, u)
    #                 - logll[mb]  # total log likelihood for normalization
    #             )
    #         else:
    #             fastemit_grad = 0.0
    #
    #         # Update the gradient of act[b, t, u, v] with the gradient from FastEmit regularization
    #         grad = grad + fastemit_grad
    #
    #         # // grad to last blank transition
    #         # grad[b, T-1, U-1, v=blank] -= exp(alphas[b, t, u) + logpk - logll[b])
    #         if (idx == blank_) and (t == T - 1) and (u == U - 1):
    #             grad -= math.exp(alphas[col] + logpk - logll[mb])
    #
    #         # grad of blank across t < T;
    #         # grad[b, t<T-1, u, v=blank] -= exp(alphas[b, t, u] + logpk - logll[b] betas[b, t + 1, u])
    #         if (idx == blank_) and (t < T - 1):
    #             grad -= math.exp(alphas[col] + logpk - logll[mb] + betas[col + maxU])
    #
    #         # grad of correct token across u < U;
    #         # grad[b, t, u<U-1, v=label[u]] -= exp(alphas[b, t, u] + logpk - logll[b] + betas[b, t, u+1])
    #         # Scale the gradient by (1.0 + FastEmit_lambda) in log space, then exponentiate
    #         if (u < U - 1) and (idx == labels[u]):
    #             # exp(log(1 + fastemit_lambda) + ...) is numerically more stable than
    #             # multiplying (1.0 + fastemit_lambda) with result.
    #             grad -= math.exp(math.log1p(fastemit_lambda) + alphas[col] + logpk - logll[mb] + betas[col + 1])
    #
    #         # update grads[b, t, u, v] = grad
    #         grads[col * alphabet_size + idx] = grad
    #
    #         # clamp gradient (if needed)
    #         if clamp > 0.0:
    #             g = grads[col * alphabet_size + idx]
    #             g = min(g, clamp)
    #             g = max(g, -clamp)
    #             grads[col * alphabet_size + idx] = g
    #
    #         # update internal index through the thread_buffer;
    #         # until idx < V + 1, such that entire vocabulary has been updated.
    #         idx += GPU_RNNT_THREAD_SIZE
