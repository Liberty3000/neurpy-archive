import torch, torch.nn as nn
import torch.nn.functional as F

from neurpy.dnn.layers import EqualizedLinear
from neurpy.dnn.layers import EqualizedConv2d
from neurpy.dnn.layers import EqualizedBlock

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.progression = nn.ModuleList([
        EqualizedBlock( 16,  32, 3, 1),
        EqualizedBlock( 32,  64, 3, 1),
        EqualizedBlock( 64, 128, 3, 1),
        EqualizedBlock(128, 256, 3, 1),
        EqualizedBlock(256, 512, 3, 1),
        EqualizedBlock(512, 512, 3, 1),
        EqualizedBlock(512, 512, 3, 1),
        EqualizedBlock(512, 512, 3, 1),
        EqualizedBlock(513, 512, 3, 1, 4, 0)])
        self.depth = len(self.progression)

        self.from_rgb = nn.ModuleList([
        EqualizedConv2d(3, 16,  1),
        EqualizedConv2d(3, 32,  1),
        EqualizedConv2d(3, 64,  1),
        EqualizedConv2d(3, 128, 1),
        EqualizedConv2d(3, 256, 1),
        EqualizedConv2d(3, 512, 1),
        EqualizedConv2d(3, 512, 1),
        EqualizedConv2d(3, 512, 1),
        EqualizedConv2d(3, 512, 1)])

        self.output_layer = EqualizedLinear(512, 1)

    def forward(self, x, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.depth - i - 1

            if i == step:
                _= self.from_rgb[index](x)

            if i == 0: # minibatch stddev
                stddev = torch.sqrt(_.var(0, unbiased=False) + 1e-8)
                mean = stddev.mean()
                mean = mean.expand(_.size(0), 1, 4, 4)
                _= torch.cat([_, mean], dim=1)

            _= self.progression[index](_)

            if i > 0:
                _= F.interpolate(_, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    skip = self.from_rgb[index + 1](x)
                    skip = F.interpolate(skip, scale_factor=0.5, mode='bilinear', align_corners=False)
                    out = (1 - alpha) * skip + alpha * _

        _= self.output_layer(_.squeeze())
        return _
