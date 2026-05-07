import torch.nn as nn
import torch.nn.functional as F
from src.losses.anticollapse import BCS


class TGJEPALoss(nn.Module):
    def __init__(self, lambda_reg=0.01, bcs_num_slices=1024, bcs_lmbd=0.1):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.bcs = BCS(num_slices=bcs_num_slices, lmbd=bcs_lmbd)

    def forward(self, z_pred, z_target, z_online_all):
        # z_pred: [M, D] predictions for masked nodes (l2-normalized)
        # z_target: [M, D] targets for masked nodes, stop-gradded (already l2-normalized)
        # z_online_all: [B*N, D] online encoder outputs for full batch
        # mse on normalized vectors equals l2 on the unit sphere
        pred_loss = F.mse_loss(z_pred, z_target)
        # bcs is a regularizer over the online encoder distribution alone
        sigreg = self.bcs(z_online_all)
        total = pred_loss + self.lambda_reg * sigreg["loss"]
        return total, pred_loss, sigreg
