from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def all_reduce(x, op):
    # all-reduce for distributed training
    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op)
        return x
    else:
        return x


def _total_batch_size(local_n: int) -> int:
    # return local_n * world_size for epps-pulley scaling
    if dist.is_available() and dist.is_initialized():
        return local_n * dist.get_world_size()
    return local_n


class FullGatherLayer(torch.autograd.Function):
    # gather tensors from all processes with correct backward gradients

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def batch_all_gather(x: torch.Tensor) -> torch.Tensor:
    # gather x across gpus along dim 0; no-op when not distributed
    if dist.is_available() and dist.is_initialized():
        return torch.cat(FullGatherLayer.apply(x), dim=0)
    return x


class HingeStdLoss(torch.nn.Module):
    def __init__(self, std_margin: float = 1.0):
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        x = batch_all_gather(x)
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss


class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def off_diagonal(self, x):
        n, m = x.shape
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        x = batch_all_gather(x)
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        cov_loss = self.off_diagonal(cov).pow(2).mean()
        return cov_loss


class VCLoss(nn.Module):
    # variance-covariance loss attracting means to zero and covariance to identity

    def __init__(self, std_coeff, cov_coeff, proj=None):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.proj = nn.Identity() if proj is None else proj
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, x, actions=None):
        x = x.transpose(0, 1).flatten(1).transpose(0, 1)  # [B*T*H*W, C]
        fx = self.proj(x)  # [B*T*H*W, C']

        std_loss = self.std_loss_fn(fx)
        cov_loss = self.cov_loss_fn(fx)

        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        total_unweighted_loss = std_loss + cov_loss
        loss_dict = {
            "std_loss": std_loss.detach(),
            "cov_loss": cov_loss.detach(),
        }
        return loss, total_unweighted_loss, loss_dict


class VICRegLoss(nn.Module):
    # vicreg loss combining invariance, variance, and covariance terms

    def __init__(self, std_coeff=1.0, cov_coeff=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, z1, z2=None):
        if z2 is not None:
            sim_loss = F.mse_loss(z1, z2)
            var_loss = self.std_loss_fn(z1) + self.std_loss_fn(z2)
            cov_loss = self.cov_loss_fn(z1) + self.cov_loss_fn(z2)
        else:
            centroid = z1.mean(dim=0)  # [B, D]
            sim_loss = (z1 - centroid).square().mean()
            pooled = z1.reshape(-1, z1.shape[-1])  # [V*B, D]
            var_loss = self.std_loss_fn(pooled)
            cov_loss = self.cov_loss_fn(pooled)

        total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


class EppsPulley(nn.Module):
    """epps-pulley test statistic for gaussianity (lejpa-style)."""

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        t = torch.linspace(0, t_max, n_points)
        phi = torch.exp(-0.5 * t**2)
        # trapezoidal weights, doubled for symmetric [-t_max, t_max]
        dt = t_max / (n_points - 1)
        w = torch.full((n_points,), 2.0 * dt)
        w[0] = dt
        w[-1] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("w", w)

    def forward(
        self, x: torch.Tensor, total_batch_size: Optional[int] = None
    ) -> torch.Tensor:
        if total_batch_size is None:
            total_batch_size = x.shape[0]
        x_t = x.unsqueeze(2) * self.t  # [N, M, T]
        cos_ecf = x_t.cos().mean(0)  # [M, T]
        sin_ecf = x_t.sin().mean(0)  # [M, T]
        cos_ecf = all_reduce(cos_ecf, op="AVG")
        sin_ecf = all_reduce(sin_ecf, op="AVG")
        err = self.w * self.phi * ((cos_ecf - self.phi) ** 2 + sin_ecf**2)
        return err.sum(1) * total_batch_size  # [M]


def _sliced_epps_pulley(
    x: torch.Tensor,
    step: int,
    num_slices: int,
    total_batch_size: int,
    epps: EppsPulley,
) -> Tuple[torch.Tensor, int]:
    # random-project x and return (epps_pulley_mean, next_step)
    with torch.no_grad():
        dev = x.device
        g = torch.Generator(device=dev)
        g.manual_seed(step)
        A = torch.randn(x.size(1), num_slices, device=dev, generator=g)
        A /= A.norm(p=2, dim=0)
    projected = x @ A  # [N, num_slices]
    loss = epps(projected, total_batch_size=total_batch_size).mean()
    return loss, step + 1


class BCS(nn.Module):
    # bcs (batched characteristic slicing) loss for sigreg

    def __init__(self, num_slices=1024, lmbd=0.1):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd
        self._total_n = None
        self.epps = EppsPulley()

    def forward(self, z1, z2=None):
        if z2 is not None:
            if self._total_n is None:
                self._total_n = _total_batch_size(z1.shape[0])
            bcs1, _ = _sliced_epps_pulley(
                z1, self.step, self.num_slices, self._total_n, self.epps
            )
            bcs2, self.step = _sliced_epps_pulley(
                z2, self.step, self.num_slices, self._total_n, self.epps
            )
            bcs = (bcs1 + bcs2) / 2
            invariance_loss = F.mse_loss(z1, z2)
        else:
            pooled = z1.reshape(-1, z1.shape[-1])  # [V*B, D]
            if self._total_n is None:
                self._total_n = _total_batch_size(pooled.shape[0])
            bcs, self.step = _sliced_epps_pulley(
                pooled, self.step, self.num_slices, self._total_n, self.epps
            )
            centroid = z1.mean(dim=0)  # [B, D]
            invariance_loss = (z1 - centroid).square().mean()

        total_loss = invariance_loss + self.lmbd * bcs
        return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}
