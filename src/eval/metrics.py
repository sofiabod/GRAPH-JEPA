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
    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return entropy.exp().item()


def mean_pairwise_cosine(z: torch.Tensor) -> float:
    # mean cosine similarity over all pairs (excluding self)
    # z: [N, D]
    z_n = F.normalize(z, dim=-1)
    sim_matrix = z_n @ z_n.T  # [N, N]
    n = z_n.shape[0]
    # exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    return sim_matrix[mask].mean().item()
