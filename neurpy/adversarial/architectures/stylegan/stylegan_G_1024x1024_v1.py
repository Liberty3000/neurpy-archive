import torch.nn as nn
import torch.nn.functional as F
from neurpy.dnn.layers import *

class Mapping(nn.Module):
    def __init__(self, zdim=512, depth=8):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(depth):
            layers.append(EqualizedLinear(zdim, zdim))
            layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Synthesis(nn.Module):
    def __init__(self, zdim):
        super().__init__()

        self.progression = nn.ModuleList([
        EqualizedStyleBlock(512, 512, 3, 1, initial=True),
        EqualizedStyleBlock(512, 512, 3, 1),
        EqualizedStyleBlock(512, 512, 3, 1),
        EqualizedStyleBlock(512, 512, 3, 1),
        EqualizedStyleBlock(512, 256, 3, 1),
        EqualizedStyleBlock(256, 128, 3, 1),
        EqualizedStyleBlock(128, 64,  3, 1),
        EqualizedStyleBlock( 64, 32,  3, 1),
        EqualizedStyleBlock( 32, 16,  3, 1)])
        self.depth = len(self.progression)

        self.to_rgb = nn.ModuleList([
        EqualizedConv2d(512, 3, 1),
        EqualizedConv2d(512, 3, 1),
        EqualizedConv2d(512, 3, 1),
        EqualizedConv2d(512, 3, 1),
        EqualizedConv2d(256, 3, 1),
        EqualizedConv2d(128, 3, 1),
        EqualizedConv2d( 64, 3, 1),
        EqualizedConv2d( 32, 3, 1),
        EqualizedConv2d( 16, 3, 1)])

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        if len(style) < 2:inject_index = [len(self.progression) + 1]
        else:inject_index = random.sample(list(range(step)), len(style) - 1)

        out, crossover = noise[0], 0
        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))
                style_step = style[crossover]
            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]
                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
                out = conv(upsample, style_step, noise[i])
            else:
                out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)
                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](upsample)
                    out = (1 - alpha) * skip_rgb + alpha * out
                break

        return out

class Generator(nn.Module):
    def __init__(self, zdim=512):
        super().__init__()
        self.progress = lambda step:4 * 2**step
        self.synthesis = Synthesis(zdim)
        self.mapping = Mapping(zdim)

    def mean_style(self, x):
        style = self.mapping(x).mean(0, keepdim=True)
        return style

    def forward(self, input, noise=None, step=0, alpha=-1, mean_style=None,
                style_weight=0, mixing_range=(-1, -1)):

        if type(input) not in (list, tuple):
            input = [input]

        styles = [self.mapping(i) for i in input]

        batch = input[0].shape[0]

        if noise is None:
            noise = []
            for i in range(step + 1):
                size = self.progress(i)
                noise.append(torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            styles = [mean_style + style_weight * (style - mean_style) for style in styles]

        return self.synthesis(styles, noise, step, alpha, mixing_range=mixing_range)
