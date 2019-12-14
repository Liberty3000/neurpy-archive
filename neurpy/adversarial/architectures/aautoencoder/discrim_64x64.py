import torch as th

class Discriminator(th.nn.Module):
    def __init__(self, zdim, ydim, outputfn):
        super().__init__()
        self.zdim = zdim
        self.ydim = ydim
        self.outputfn = outputfn

        block = lambda inch=zdim,ch=512,gn=32: th.nn.Sequential(
        th.nn.Linear(inch, ch),
        th.nn.GroupNorm(gn,ch),
        th.nn.LeakyReLU(2e-1))

        layers = [block() for _ in range(6)]
        layers.append(th.nn.Linear(512, ydim))

        self.model = th.nn.Sequential(*layers)

    def forward(self, x):
        if len(x.size()) > 2: x = x.view(x.size(0),-1)
        return self.outputfn(self.model(x))
