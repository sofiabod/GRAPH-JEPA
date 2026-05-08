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


def sparsity_at_threshold(z: torch.Tensor, threshold: float = 0.01) -> float:
    # fraction of near-zero entries in the representation tensor z.
    # connects to rlpjepa (wang et al. 2026) which shows jepa-style training
    # induces sparse heavy-tailed latent distributions; we measure whether
    # bcs-trained graph-jepa latents exhibit the same property.
    # z: [N, D] (or any shape) -> returns scalar in [0, 1]
    return (z.abs() < threshold).float().mean().item()


def mean_velocity_cos(z_seq: torch.Tensor) -> float:
    """mean cosine similarity between consecutive velocity vectors along node trajectories.

    measures how straight node trajectories are in latent space. directly mirrors
    the curvature objective from wang et al. 2026 (temporal straightening for
    latent planning, icml 2026): C = cos(v_t, v_{t+1}) where v_t = z_{t+1} - z_t.

    higher C means straighter trajectories, where euclidean distance better
    approximates geodesic distance, and gradient-based planning is better
    conditioned. wang et al. show that joint encoder-predictor training under
    the jepa objective implicitly produces straighter trajectories; this metric
    quantifies that property in our learned latents.

    z_seq: [T, N, D] sequence of node embeddings over T >= 3 timesteps.
    returns: mean cosine in [-1, 1], or nan if T < 3 (need at least 2 velocities).
    """
    if z_seq.shape[0] < 3:
        return float('nan')
    # velocity at each consecutive pair: [T-1, N, D]
    v = z_seq[1:] - z_seq[:-1]
    # consecutive velocity pairs: [T-2, N, D]
    v_t = v[:-1]
    v_t1 = v[1:]
    # cosine similarity between consecutive velocities, per node, per time: [T-2, N]
    cos = F.cosine_similarity(v_t, v_t1, dim=-1)
    return cos.mean().item()
