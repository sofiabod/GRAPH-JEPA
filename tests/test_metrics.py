import torch
from src.eval.metrics import cosine_sim, effective_rank, mean_pairwise_cosine


def test_cosine_sim_identical():
    # cosine_sim(z, z) should be all 1.0
    z = torch.randn(10, 256)
    sims = cosine_sim(z, z)
    assert sims.shape == (10,)
    assert torch.allclose(sims, torch.ones(10), atol=1e-5)


def test_cosine_sim_orthogonal():
    # two orthogonal vectors have cosine sim 0
    z1 = torch.tensor([[1.0, 0.0]])
    z2 = torch.tensor([[0.0, 1.0]])
    sims = cosine_sim(z1, z2)
    assert abs(sims[0].item()) < 1e-5


def test_effective_rank_identity():
    # identity matrix has effective rank == D (maximally spread singular values)
    z = torch.eye(64)
    rank = effective_rank(z)
    assert abs(rank - 64.0) < 1.0


def test_effective_rank_rank1():
    # rank-1 matrix has effective rank == 1
    v = torch.randn(64, 1)
    z = v.expand(-1, 64)  # all rows identical -> rank 1
    rank = effective_rank(z)
    assert rank < 1.5


def test_mean_pairwise_cosine_identical():
    # all identical vectors -> mean pairwise cosine == 1
    v = torch.randn(1, 256)
    z = v.expand(20, -1)
    mpc = mean_pairwise_cosine(z)
    assert abs(mpc - 1.0) < 1e-4


def test_mean_pairwise_cosine_range():
    # random unit vectors -> result in [-1, 1]
    z = torch.randn(50, 256)
    mpc = mean_pairwise_cosine(z)
    assert -1.0 <= mpc <= 1.0


def test_effective_rank_all_zero_returns_one():
    # degenerate input: all-zero matrix should return 1.0, not nan
    z = torch.zeros(20, 64)
    rank = effective_rank(z)
    assert rank == 1.0


def test_effective_rank_tiny_signal_returns_finite():
    # numerically vanishing singular values should not propagate as nan
    z = torch.full((10, 16), 1e-15)
    rank = effective_rank(z)
    assert torch.isfinite(torch.tensor(rank)).item()


def test_mean_pairwise_cosine_subsamples_large_input():
    # with N>5000 the function should subsample without erroring out
    z = torch.randn(6000, 32)
    mpc = mean_pairwise_cosine(z)
    assert -1.0 <= mpc <= 1.0


def test_mean_pairwise_cosine_subsample_deterministic():
    # repeated calls on the same input should yield the same value
    z = torch.randn(6000, 32)
    a = mean_pairwise_cosine(z)
    b = mean_pairwise_cosine(z)
    assert abs(a - b) < 1e-6
