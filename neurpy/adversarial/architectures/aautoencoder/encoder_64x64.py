import torch as th

class Encoder(th.nn.Module):
    def __init__(self, xdim, zdim):
        super().__init__()
        self.xdim, self.zdim = xdim, zdim

        self.model = th.nn.Sequential(
        th.nn.Conv2d(  3, 64, kernel_size=3, stride=2, padding=1),
        th.nn.GroupNorm(8,64),
        th.nn.LeakyReLU(2e-1,True),

        th.nn.Conv2d( 64, 128, kernel_size=3, stride=2, padding=1),
        th.nn.GroupNorm(16,128),
        th.nn.LeakyReLU(2e-1,True),

        th.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        th.nn.GroupNorm(32,256),
        th.nn.LeakyReLU(2e-1,True),

        th.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        th.nn.GroupNorm(64,512),
        th.nn.LeakyReLU(2e-1,True),

        th.nn.Conv2d(512,1024, kernel_size=3, stride=2, padding=1),
        th.nn.GroupNorm(128,1024),
        th.nn.LeakyReLU(2e-1,True))

        self.bottleneck = th.nn.Conv2d(1024, zdim, 2, stride=1, padding=0)

    def forward(self, x):
        return self.bottleneck(self.model(x))
