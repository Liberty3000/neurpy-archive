import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, xshape, yshape, outputfn=None):
        super().__init__()
        self.zshape = xshape
        self.yshape = yshape
        self.outputfn = outputfn

        self.model = nn.Sequential(
        nn.Conv2d(  3,  64, 4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(.2, True),

        nn.Conv2d( 64, 128, 3, stride=2, padding=1, bias=False),
        nn.GroupNorm(32,128),
        nn.LeakyReLU(.2, True),

        nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
        nn.GroupNorm(32,256),
        nn.LeakyReLU(.2, True),

        nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
        nn.GroupNorm(32,512),
        nn.LeakyReLU(.2, True),

        nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),
        nn.GroupNorm(32,512),
        nn.LeakyReLU(.2, True),

        nn.Conv2d(512,   1, 4, stride=1, padding=0, bias=False))

        self.model.apply(self.initialize)

    def initialize(self, layer):
        if isinstance(layer,nn.Conv2d) :
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if isinstance(layer,nn.BatchNorm2d) :
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
            nn.init.constant_(layer.bias.data, 0.0)

    def forward(self, x):
        _= self.model(x)
        _= self.outputfn(_) if self.outputfn else _
        return _
