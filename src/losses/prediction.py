import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.anticollapse import BCS


class TGJEPALoss(nn.Module):
    def __init__(self, lambda_reg=0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.bcs = BCS()

    def forward(self, z_pred, z_target, z_online_all, z_target_all):
        # z_pred: [M, D] predictions for masked nodes
        # z_target: [M, D] targets for masked nodes, stop-gradded
        # z_online_all: [B*N, D] online encoder outputs for full batch
        # z_target_all: [B*N, D] target encoder outputs, stop-gradded
        pred_loss = F.smooth_l1_loss(z_pred, z_target)
        sigreg = self.bcs(z_online_all, z_target_all)
        total = pred_loss + self.lambda_reg * sigreg["loss"]
        return total, pred_loss, sigreg
