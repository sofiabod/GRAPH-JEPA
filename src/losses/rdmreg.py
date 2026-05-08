"""rectified distribution matching regularization (rdmreg) — anti-collapse loss.

ported from kuang et al. 2026, "rectified lpjepa: joint-embedding predictive
architectures with sparse and maximum-entropy representations" (arXiv:2602.01456).
upstream: https://github.com/yilunkuang/rectified-lp-jepa (MIT license).

drop-in alternative to bcs in src/losses/anticollapse.py. pushes representations
to match a (rectified) generalized gaussian distribution via sliced wasserstein
distance.

usage:
    from src.losses.rdmreg import rdmreg_loss, build_projection_vectors, choose_sigma_for_unit_var
    proj = build_projection_vectors(d=256, n_proj=128, device='cuda')
    sigma = choose_sigma_for_unit_var(p=1.0, mu=0.0)  # precompute once
    loss = rdmreg_loss(z1, z2, proj, target_dist='rectified_lp_distribution',
                       lp_norm_parameter=1.0, chosen_sigma=sigma)
"""

import math

import torch
import torch.nn.functional as F
from torch.distributions.laplace import Laplace


def _sample_product_laplace(shape, device, dtype, loc=0.0, scale=1.0 / math.sqrt(2)):
    """sample from a product laplace distribution."""
    loc_t = torch.tensor(loc, device=device, dtype=dtype)
    scale_t = torch.tensor(scale, device=device, dtype=dtype)
    laplace_dist = Laplace(loc=loc_t, scale=scale_t)
    return laplace_dist.sample(shape)


def sample_lp_distribution(shape, p, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """sample from generalized gaussian gn_p(loc, scale).

    p=1.0 is laplace, p=2.0 is gaussian. generic p uses gamma sampling.
    """
    if p == 1.0:
        return _sample_product_laplace(shape, device, dtype, loc=loc, scale=scale)
    elif p == 2.0:
        return loc + scale * torch.randn(shape, device=device, dtype=dtype)
    else:
        sign = torch.empty(shape, device=device, dtype=dtype).bernoulli_(0.5)
        sign = 2 * sign - 1
        gamma = torch.distributions.Gamma(concentration=1.0 / p, rate=1.0)
        g = gamma.sample(shape).to(device=device, dtype=dtype)
        x = sign * (p * g).pow(1.0 / p)
        return loc + scale * x


def determine_sigma_for_lp_dist(p):
    """sigma such that gn_p(0, sigma) has unit variance."""
    return (math.gamma(1 / p) ** (1 / 2)) / ((p ** (1 / p)) * (math.gamma(3 / p) ** (1 / 2)))


def choose_sigma_for_unit_var(p, mu, target_var=1.0, rtol=1e-8, max_iter=200):
    """find sigma > 0 such that var(relu(x)) = target_var when x ~ gn_p(mu, sigma).

    bisection. for the typical case mu=0 the result is approx
    determine_sigma_for_lp_dist(p) / sqrt(2) since relu zeros half the mass.
    upstream uses mpmath for high-precision incomplete gamma. this version uses
    monte-carlo estimation of var(relu(x)) — sufficient accuracy for training.
    """
    n_samples = 100_000

    def var_relu(sig):
        x = sample_lp_distribution((n_samples,), p, loc=mu, scale=sig, device="cpu")
        y = torch.relu(x)
        return y.var().item() - target_var

    lo, hi = 1e-8, 1.0
    flo = var_relu(lo)
    fhi = var_relu(hi)
    k = 0
    while flo * fhi > 0 and k < 200:
        hi *= 2
        fhi = var_relu(hi)
        k += 1
    if flo * fhi > 0:
        raise RuntimeError("failed to bracket a root for sigma; try different range")
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fmid = var_relu(mid)
        if abs(fmid) <= rtol * (1 + target_var):
            return mid
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return (lo + hi) / 2


def build_projection_vectors(d, n_proj, device, dtype=torch.float32, seed=0):
    """random fixed projection matrix [n_proj, d]. shared across views and batches."""
    g = torch.Generator(device=device if device != "cpu" else "cpu")
    g.manual_seed(seed)
    p = torch.randn((n_proj, d), generator=g, device=device, dtype=dtype)
    p = F.normalize(p, dim=1)
    return p


def _swd_one_view(
    features,
    projection_vectors,
    target_dist,
    mean_shift_value: float = 0.0,
    lp_norm_parameter: float = 1.0,
    chosen_sigma: float = 1.0,
):
    """sliced wasserstein distance from features to target distribution.

    features: [B, D]
    projection_vectors: [n_proj, D]
    chosen_sigma: precomputed via choose_sigma_for_unit_var; required (not optional).
    """
    B, D = features.shape
    projected_features = features @ projection_vectors.T  # [B, n_proj]

    if target_dist == "rectified_lp_distribution":
        target_samples = torch.relu(
            sample_lp_distribution(
                shape=(B, D),
                p=lp_norm_parameter,
                loc=mean_shift_value,
                scale=float(chosen_sigma),
                device=features.device,
                dtype=features.dtype,
            )
        )
    elif target_dist == "lp_distribution":
        target_samples = sample_lp_distribution(
            shape=(B, D),
            p=lp_norm_parameter,
            loc=mean_shift_value,
            scale=float(chosen_sigma),
            device=features.device,
            dtype=features.dtype,
        )
    else:
        raise ValueError(f"unsupported target_dist: {target_dist}")

    projected_targets = target_samples @ projection_vectors.T  # [B, n_proj]

    # 1d wasserstein along batch axis: sort, mse
    sorted_features, _ = torch.sort(projected_features, dim=0)
    sorted_targets, _ = torch.sort(projected_targets, dim=0)
    return ((sorted_features - sorted_targets) ** 2).mean()


def rdmreg_loss(
    z1,
    z2,
    projection_vectors,
    target_dist,
    mean_shift_value: float = 0.0,
    lp_norm_parameter: float = 1.0,
    chosen_sigma: float = 1.0,
):
    """rectified distribution matching regularization across two views.

    z1, z2: [B, D] online encoder outputs for context and target views.
    chosen_sigma: precompute once via choose_sigma_for_unit_var.
    returns the swd-to-target averaged across both views.
    """
    swd1 = _swd_one_view(
        z1, projection_vectors, target_dist, mean_shift_value, lp_norm_parameter, chosen_sigma
    )
    swd2 = _swd_one_view(
        z2, projection_vectors, target_dist, mean_shift_value, lp_norm_parameter, chosen_sigma
    )
    return (swd1 + swd2) / 2
