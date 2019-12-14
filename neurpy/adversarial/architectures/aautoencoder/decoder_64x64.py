import torch as th

class Decoder(th.nn.Module):
    def __init__(self, zdim, xdim, outputfn):
        super().__init__()
        self.xdim, self.zdim = xdim, zdim
        self.outputfn = outputfn

        self.model = th.nn.Sequential(
        th.nn.ConvTranspose2d(zdim, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
        th.nn.GroupNorm(128,1024),
        th.nn.LeakyReLU(2e-1,True),

        th.nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
        th.nn.GroupNorm(64,512),
        th.nn.LeakyReLU(2e-1,True),

        th.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        th.nn.GroupNorm(32,256),
        th.nn.LeakyReLU(2e-1,True),

        th.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        th.nn.GroupNorm(16,128),
        th.nn.LeakyReLU(2e-1,True),

        th.nn.ConvTranspose2d(128,  64, kernel_size=3, stride=2, padding=1, output_padding=1),
        th.nn.GroupNorm(8,64),
        th.nn.LeakyReLU(2e-1,True),

        th.nn.ConvTranspose2d(64,  3, kernel_size=3, stride=2, padding=1, output_padding=1))

    def forward(self, z):
        if len(z.size()) < 3: z = z.view(*z.size(),1,1)
        return self.outputfn(self.model(z))
