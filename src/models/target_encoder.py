import copy


class TargetEncoder:
    # not nn.Module; exposes parameters() and named_parameters() delegating to inner encoder
    def __init__(self, encoder):
        self.encoder = copy.deepcopy(encoder)
        # always eval mode so dropout and similar stochastic layers stay off
        # ema updates write directly to .data and do not depend on train/eval
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def parameters(self):
        return self.encoder.parameters()

    def named_parameters(self):
        return self.encoder.named_parameters()

    def forward(self, data):
        return self.encoder(data).detach()

    def __call__(self, data):
        return self.forward(data)
