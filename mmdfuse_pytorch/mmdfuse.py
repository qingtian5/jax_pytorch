# import jax
# import jax.numpy as jnp
# from jax import random, jit, vmap
# from functools import partial
# from jax.scipy.special import logsumexp
import torch


import torch
import torch.nn.functional as F

def mmdfuse(
    X,
    Y,
    alpha=0.05,
    kernels=("laplace", "gaussian"),
    lambda_multiplier=1,
    number_bandwidths=10,
    number_permutations=2000,
    return_p_val=False,
):
    if Y.shape[0] > X.shape[0]:
        X, Y = Y, X
    m = X.shape[0]
    n = Y.shape[0]
    assert n <= m
    assert n >= 2 and m >= 2
    assert 0 < alpha < 1
    assert lambda_multiplier > 0
    assert number_bandwidths > 1 and isinstance(number_bandwidths, int)
    assert number_permutations > 0 and isinstance(number_permutations, int)
    if isinstance(kernels, str):
        kernels = (kernels,)
    for kernel in kernels:
        assert kernel in (
            "imq",
            "rq",
            "gaussian",
            "matern_0.5_l2",
            "matern_1.5_l2",
            "matern_2.5_l2",
            "matern_3.5_l2",
            "matern_4.5_l2",
            "laplace",
            "matern_0.5_l1",
            "matern_1.5_l1",
            "matern_2.5_l1",
            "matern_3.5_l1",
            "matern_4.5_l1",
        )

    all_kernels_l1 = (
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    )
    all_kernels_l2 = (
        "imq",
        "rq",
        "gaussian",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
    )
    number_kernels = len(kernels)
    kernels_l1 = [k for k in kernels if k in all_kernels_l1]
    kernels_l2 = [k for k in kernels if k in all_kernels_l2]

    # Setup for permutations
    B = number_permutations
    idx = torch.randperm(m + n).repeat(B + 1, 1)
    v11 = torch.cat((torch.ones(m), -torch.ones(n)))  
    V11i = v11.repeat(B + 1, 1)
    V11 = V11i.gather(1, idx)
    V11[-1] = v11
    V11 = V11.transpose(0, 1)

    v10 = torch.cat((torch.ones(m), torch.zeros(n)))
    V10i = v10.repeat(B + 1, 1)
    V10 = V10i.gather(1, idx)
    V10[-1] = v10
    V10 = V10.transpose(0, 1)

    v01 = torch.cat((torch.zeros(m), -torch.ones(n)))
    V01i = v01.repeat(B + 1, 1)
    V01 = V01i.gather(1, idx)
    V01[-1] = v01
    V01 = V01.transpose(0, 1)

    N = number_bandwidths * number_kernels
    M = torch.zeros((N, B + 1))
    kernel_count = -1  

    for r in range(2):
        kernels_l = (kernels_l1, kernels_l2)[r]
        l = ("l1", "l2")[r]
        if len(kernels_l) > 0:
            Z = torch.cat((X, Y))
            # print(Z.shape)
            pairwise_matrix = torch_distances(Z, Z, l, matrix=True)
            # print(f"pairwise_matrix: {pairwise_matrix.shape}")

            def compute_bandwidths(distances, number_bandwidths):
                # print(distances.shape, number_bandwidths)
                median = torch.median(distances)
                distances = distances + (distances == 0).float() * median
                dd = torch.sort(distances).values
                lambda_min = dd[int(torch.floor(torch.tensor(len(dd)) * 0.05).item())] / 2
                lambda_max = dd[int(torch.floor(torch.tensor(len(dd)) * 0.95).item())] * 2

                bandwidths = torch.linspace(lambda_min, lambda_max, number_bandwidths)
                
                return bandwidths

            triu_indices = torch.triu_indices(pairwise_matrix.shape[0], pairwise_matrix.shape[1], offset=0)

            distances = pairwise_matrix[triu_indices[0], triu_indices[1]]
            # print(f"distances: {distances.shape}")

            bandwidths = compute_bandwidths(distances, number_bandwidths)

            for j in range(len(kernels_l)):
                kernel = kernels_l[j]
                kernel_count += 1
                for i in range(number_bandwidths):
                    bandwidth = bandwidths[i]
                    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
                    K.fill_diagonal_(0)
                    unscaled_std = torch.sqrt(torch.sum(K**2))

                    M[kernel_count * number_bandwidths + i] = (
                        torch.sum(V10 * (K @ V10), dim=0) * (n - m + 1) * (n - 1) / (m * (m - 1))
                        + torch.sum(V01 * (K @ V01), dim=0) * (m - n + 1) / m
                        + torch.sum(V11 * (K @ V11), dim=0) * (n - 1) / m
                    ) / unscaled_std * torch.sqrt(torch.tensor(n * (n - 1), dtype=torch.float32))

    all_statistics = torch.logsumexp(lambda_multiplier * M, dim=0) / N
    original_statistic = all_statistics[-1]

    p_val = torch.mean((all_statistics >= original_statistic).float())
    output = p_val <= alpha

    if return_p_val:
        return output.int(), p_val
    else:
        return output.int()



def kernel_matrix(pairwise_matrix, l, kernel, bandwidth, rq_kernel_exponent=0.5):
    """
    Compute kernel matrix for a given kernel and bandwidth.

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel == "gaussian" and l == "l2":
        return torch.exp(-(d**2) / 2)
    elif kernel == "laplace" and l == "l1":
        return torch.exp(-d * torch.sqrt(torch.tensor(2.0)))
    elif kernel == "rq" and l == "l2":
        return (1 + d**2 / (2 * rq_kernel_exponent)) ** (-rq_kernel_exponent)
    elif kernel == "imq" and l == "l2":
        return (1 + d**2) ** (-0.5)
    elif (kernel == "matern_0.5_l1" and l == "l1") or (
        kernel == "matern_0.5_l2" and l == "l2"
    ):
        return torch.exp(-d)
    elif (kernel == "matern_1.5_l1" and l == "l1") or (
        kernel == "matern_1.5_l2" and l == "l2"
    ):
        return (1 + torch.sqrt(torch.tensor(3.0)) * d) * torch.exp(-torch.sqrt(torch.tensor(3.0)) * d)
    elif (kernel == "matern_2.5_l1" and l == "l1") or (
        kernel == "matern_2.5_l2" and l == "l2"
    ):
        return (1 + torch.sqrt(torch.tensor(5.0)) * d + 5 / 3 * d**2) * torch.exp(-torch.sqrt(torch.tensor(5.0)) * d)
    elif (kernel == "matern_3.5_l1" and l == "l1") or (
        kernel == "matern_3.5_l2" and l == "l2"
    ):
        return (
            1 + torch.sqrt(torch.tensor(7.0)) * d + 2 * 7 / 5 * d**2 + 7 * torch.sqrt(torch.tensor(7.0)) / 3 / 5 * d**3
        ) * torch.exp(-torch.sqrt(torch.tensor(7.0)) * d)
    elif (kernel == "matern_4.5_l1" and l == "l1") or (
        kernel == "matern_4.5_l2" and l == "l2"
    ):
        return (
            1
            + 3 * d
            + 3 * (6**2) / 28 * d**2
            + (6**3) / 84 * d**3
            + (6**4) / 1680 * d**4
        ) * torch.exp(-3 * d)
    else:
        raise ValueError('The values of "l" and "kernel" are not valid.')


def torch_distances(X, Y, l, max_samples=None, matrix=False):
    if l == "l1":
        def dist(x, y):
            z = x - y
            return torch.sum(torch.abs(z))

    elif l == "l2":
        def dist(x, y):
            z = x - y
            return torch.sqrt(torch.sum(torch.square(z)))

    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")

    # Apply dist function to each pair of points in X and Y using torch's vectorized operations
    if max_samples:
        X = X[:max_samples]
        Y = Y[:max_samples]

    # Calculate pairwise distances
    pairwise_dist = torch.cdist(X, Y, p=1 if l == "l1" else 2)

    if matrix:
        return pairwise_dist
    else:
        upper_tri_indices = torch.triu_indices(pairwise_dist.size(0), pairwise_dist.size(1))
        return pairwise_dist[upper_tri_indices[0], upper_tri_indices[1]]

