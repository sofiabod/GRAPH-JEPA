import torch
import torch.nn.functional as F


def cosine_sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    # pairwise cosine similarity between corresponding rows
    # z1, z2: [N, D] -> returns [N]
    z1_n = F.normalize(z1, dim=-1)
    z2_n = F.normalize(z2, dim=-1)
    return (z1_n * z2_n).sum(dim=-1)


def effective_rank(z: torch.Tensor) -> float:
    # effective rank = exp(entropy of normalized singular values)
    # z: [N, D]
    # robustness: degenerate inputs (all zeros or numerically vanishing
    # singular values) return 1.0 instead of NaN
    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    s = s[s > 1e-10]
    if s.numel() == 0 or s.sum().item() < 1e-12:
        return 1.0
    s = torch.clamp(s, min=1e-12)
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return entropy.exp().item()


def mean_pairwise_cosine(z: torch.Tensor) -> float:
    # mean cosine similarity over all pairs (excluding self).
    # z: [N, D].
    # if N > 5000, uniformly subsamples 5000 rows with a fixed-seed generator
    # to bound memory of the [N, N] sim matrix.
    if z.shape[0] > 5000:
        gen = torch.Generator(device=z.device).manual_seed(0)
        idx = torch.randperm(z.shape[0], generator=gen, device=z.device)[:5000]
        z = z[idx]
    z_n = F.normalize(z, dim=-1)
    sim_matrix = z_n @ z_n.T  # [N, N]
    n = z_n.shape[0]
    # exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    return sim_matrix[mask].mean().item()
