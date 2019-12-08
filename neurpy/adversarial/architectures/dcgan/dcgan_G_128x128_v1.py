import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, zshape, xshape, outputfn=None):
        super().__init__()
        self.zshape = zshape
        self.xshape = xshape
        self.outputfn = outputfn

        self.model = nn.Sequential(
        nn.ConvTranspose2d(zshape, 512, 4, stride=1, padding=0, bias=False),
        nn.GroupNorm(128,512),
        nn.LeakyReLU(.2, True),

        nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1, bias=False),
        nn.GroupNorm(64,512),
        nn.LeakyReLU(.2, True),

        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
        nn.GroupNorm(32,256),
        nn.LeakyReLU(.2, True),

        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
        nn.GroupNorm(16,128),
        nn.LeakyReLU(.2, True),

        nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1, bias=False),
        nn.GroupNorm( 8, 64),
        nn.LeakyReLU(.2, True),

        nn.ConvTranspose2d( 64,   3, 4, stride=2, padding=1, bias=False))

        self.model.apply(self.initialize)

    def initialize(self, layer):
        if isinstance(layer,nn.ConvTranspose2d) :
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if isinstance(layer,nn.BatchNorm2d) :
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
            nn.init.constant_(layer.bias.data, 0.0)

    def forward(self, z):
        if len(z.size()[1:]) < 3: z = z.unsqueeze(-1).unsqueeze(-1)

        _= self.model(z)
        _= self.outputfn(_) if self.outputfn else _
        return _
