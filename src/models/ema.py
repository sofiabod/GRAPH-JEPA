import math
import torch


class EMAUpdater:
    def __init__(self, momentum_start=0.996, momentum_end=1.0, total_steps=1000):
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.total_steps = total_steps

    def get_momentum(self, step):
        # cosine schedule from momentum_start to momentum_end
        m_end = self.momentum_end
        m_start = self.momentum_start
        return m_end - (m_end - m_start) * (1 + math.cos(math.pi * step / self.total_steps)) / 2

    def update(self, online, target, step):
        # ema update: p_target = m * p_target + (1 - m) * p_online
        m = self.get_momentum(step)
        with torch.no_grad():
            for p_o, p_t in zip(online.parameters(), target.encoder.parameters()):
                p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)
