"""NAFNet-lite: compact NAFNet (ECCV'22) sized for CPU-friendly inference (~0.9M params)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.dwconv = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw)
        self.act = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.conv2 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv3 = nn.Conv2d(c, ffn, 1)
        self.act2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn // 2, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.act(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta
        y = self.norm2(x)
        y = self.conv3(y)
        y = self.act2(y)
        y = self.conv4(y)
        return x + y * self.gamma


class NAFNetLite(nn.Module):
    def __init__(self, img_ch=3, widths=(16, 32, 64, 128), blocks=(1, 1, 1, 1), middle_blocks=1):
        super().__init__()
        self.intro = nn.Conv2d(img_ch, widths[0], 3, 1, 1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        for i, (w, n) in enumerate(zip(widths, blocks)):
            self.encoders.append(nn.Sequential(*[NAFBlock(w) for _ in range(n)]))
            if i < len(widths) - 1:
                self.downs.append(nn.Conv2d(w, widths[i + 1], 2, 2))
        self.middle = nn.Sequential(*[NAFBlock(widths[-1]) for _ in range(middle_blocks)])
        self.ups, self.atts, self.decoders = nn.ModuleList(), nn.ParameterList(), nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, 2))
            self.atts.append(nn.Parameter(torch.ones(1, widths[i - 1], 1, 1)))
            self.decoders.append(nn.Sequential(*[NAFBlock(widths[i - 1]) for _ in range(blocks[i - 1])]))
        self.ending = nn.Sequential(LayerNorm2d(widths[0]), nn.Conv2d(widths[0], widths[0], 1),
                                    nn.Conv2d(widths[0], img_ch, 3, 1, 1))

    def forward(self, x):
        x0 = x
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders[:-1], self.downs):
            x = enc(x); skips.append(x); x = down(x)
        x = self.encoders[-1](x)
        x = self.middle(x)
        for up, att, dec in zip(self.ups, self.atts, self.decoders):
            x = up(x)
            s = skips.pop()
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, s.shape[-2:])
            x = x + s * att
            x = dec(x)
        return self.ending(x) + x0


MODELS = {
    "nafnet_small":  lambda: NAFNetLite(widths=(16, 32, 64, 128), blocks=(1, 1, 1, 1)),
    "nafnet_medium": lambda: NAFNetLite(widths=(24, 48, 96, 192), blocks=(1, 1, 1, 1)),
    "nafnet_wide":   lambda: NAFNetLite(widths=(32, 64, 128, 256), blocks=(1, 1, 1, 1)),
}


def build_model(name):
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Options: {list(MODELS)}")
    return MODELS[name]()
