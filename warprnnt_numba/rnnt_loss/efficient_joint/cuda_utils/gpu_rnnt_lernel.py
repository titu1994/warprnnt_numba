import math

import torch
from numba import cuda

from warprnnt_numba.rnnt_loss.utils import rnnt_helper

# Index of the log_probs tensor
CACHE_IDX_BLANK = 0
CACHE_IDX_SYMBOL = 1


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
