import importlib, os, pathlib, threading
import torch as th
from torch.autograd import Variable
import neurpy
from neurpy import util
from neurpy.adversarial.models.model import Model

class DCGAN(Model):
    def __init__(self, config=None):
        super().__init__()
        gargs = [config.zdim, config.xdim, config.goutputfn]
        dargs = [config.xdim, config.ddim, config.doutputfn]
        self.G = neurpy.config.module(config.generator + '.py').Generator(*gargs)
        self.D = neurpy.config.module(config.discriminator + '.py').Discriminator(*dargs)
        self.xdim, self.zdim, self.ddim = config.xdim, config.zdim, config.ddim

    def test(self, config, datagen, generate=True, interp_z=True):
        # G(z)
        if generate:
            neurpy.generate.sample(generator=self.G, n=config.ntest, saver=config.saver,
            sampler=lambda:util.sampler(self.zdim), device=config.device)
        # G(z)
        if interp_z:
            neurpy.interpolate.spherical(sampler=lambda:util.sampler(self.zdim),
            decoder=self.G, interps=config.zinterps,saver=config.saver, device=config.device)

    def forwardbackward(self, config, datagen, callback):
        xreal,_ = next(iter(datagen))
        bsize = xreal.size(0)

        xreal = xreal.to(config.device)
        zfake = Variable(util.sampler(self.zdim, bsize)).to(config.device)

        real,fake = util.labels(self.ddim, bsize,
        rmin=config.rmin, rmax=config.rmax,
        fmin=config.fmin, fmax=config.fmax, device=config.device)

        self.doptim.zero_grad()
        dreal = self.discriminate(xreal)
        Dloss_real = self.dloss(dreal.view(-1,self.ddim), real)
        Dloss_real.backward()

        xfake,dfake = self.forward(zfake)
        Dloss_fake = self.dloss(dfake.view(-1,self.ddim), fake)
        Dloss_fake.backward()
        Dloss = Dloss_real + Dloss_fake
        self.doptim.step()
        Dacry_real = dreal.mean()
        Dacry_fake = dfake.mean()

        self.goptim.zero_grad()
        xfake,dfake = self.forward(zfake)
        Gloss = self.gloss(dfake.view(-1,self.ddim), real)
        Gloss.backward()
        self.goptim.step()

        callback.status['Gloss'].append(Gloss.item())
        callback.status['Dloss'].append(Dloss.item())
        callback.status['Dacry_real'].append(Dacry_real.item())
        callback.status['Dacry_fake'].append(Dacry_fake.item())
