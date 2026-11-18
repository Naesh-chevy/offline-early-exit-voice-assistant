"""DS-CNN keyword spotter + a small hand written QAT (fake quant with straight through grads).

Flow: train fp32 with batchnorm -> fold_bn() -> enable_qat() and fine tune -> export int8 tflite.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round to the int8 grid but let gradients pass as if nothing happened (STE)."""
    q = torch.clamp(torch.round(x / scale), -128, 127) * scale
    return x + (q - x).detach()


class ActQuant(nn.Module):
    """Fake quant for activations. Scale tracks a moving max of |x| while training."""

    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0 / 127))
        self.register_buffer("seen", torch.tensor(0))
        self.momentum = momentum
        self.on = False

    def forward(self, x):
        if not self.on:
            return x
        if self.training:
            s = x.detach().abs().max().clamp(min=1e-5) / 127
            self.scale.copy_(s if self.seen == 0 else self.momentum * self.scale + (1 - self.momentum) * s)
            self.seen.fill_(1)
        return fake_quant(x, self.scale)


def quant_weight(w: torch.Tensor, on: bool) -> torch.Tensor:
    """Per output channel symmetric int8 for weights."""
    if not on:
        return w
    scale = w.detach().abs().flatten(1).max(dim=1).values.clamp(min=1e-8) / 127
    return fake_quant(w, scale.view(-1, *([1] * (w.dim() - 1))))


class QConv(nn.Module):
    """Conv + optional batchnorm + relu with weight/act fake quant hooks. groups=ch means depthwise."""

    def __init__(self, cin, cout, k, stride=1, pad=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act_q = ActQuant()
        self.qat = False
        self.folded = False

    def forward(self, x):
        w = quant_weight(self.conv.weight, self.qat)
        y = F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, groups=self.conv.groups)
        if not self.folded:
            y = self.bn(y)
        return self.act_q(F.relu(y))


class DSCNN(nn.Module):
    """Input (B, 1, n_mels, frames). widths[0] is the first conv, the rest are depthwise separable blocks."""

    def __init__(self, widths=(64, 64, 64, 64), n_classes=2, first_kernel=(10, 4), first_stride=(2, 2)):
        super().__init__()
        self.in_q = ActQuant()
        pad = (first_kernel[0] // 2, first_kernel[1] // 2)
        self.first = QConv(1, widths[0], first_kernel, first_stride, pad)
        blocks = []
        for cin, cout in zip(widths[:-1], widths[1:]):
            blocks.append(QConv(cin, cin, 3, 1, 1, groups=cin))   # depthwise
            blocks.append(QConv(cin, cout, 1))                    # pointwise
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(widths[-1], n_classes)
        self.out_q = ActQuant()
        self.qat = False

    def qconvs(self):
        return [self.first, *self.blocks]

    def forward(self, x):
        x = self.in_q(x)
        for m in self.qconvs():
            x = m(x)
        x = x.mean(dim=(2, 3))
        w = quant_weight(self.fc.weight, self.qat)
        return self.out_q(F.linear(x, w, self.fc.bias))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def fold_bn(model: DSCNN) -> DSCNN:
    """Merge each batchnorm into its conv (w * g/std, new bias). Needed so tflite sees conv+bias only."""
    for m in model.qconvs():
        bn = m.bn
        g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        m.conv.weight.mul_(g.view(-1, 1, 1, 1))
        m.conv.bias = nn.Parameter(bn.bias - bn.running_mean * g)
        m.folded = True
    return model


def enable_qat(model: DSCNN, on: bool = True) -> DSCNN:
    model.qat = on
    model.in_q.on = model.out_q.on = on
    for m in model.qconvs():
        m.qat = on
        m.act_q.on = on
    return model
