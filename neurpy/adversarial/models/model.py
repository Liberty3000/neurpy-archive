from collections import defaultdict
import copy, importlib, os, random, threading
import imageio, PIL, matplotlib.pyplot as mp
import torch as th, numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from neurpy import util

class Model(th.nn.Module):
    def __init__(self):
        super().__init__()

    def build(self):
        raise NotImplementedError()

    def compile(self):
        raise NotImplementedError()

    def encode(self):
        raise NotImplementedError()

    def decode(self):
        raise NotImplementedError()

    def Gparams(self):
        return list(self.G.parameters())

    def Dparams(self):
        return list(self.D.parameters())

    def generate(self, x):
        return self.G(x)

    def discriminate(self, z):
        return self.D(z)

    def forward(self, z):
        x = self.G(z)
        d = self.D(x)
        return x,d

    def backward(self):
        raise NotImplementedError()

    def forward_backward(self):
        raise NotImplementedError()

    def toggle_parameters(self, G=True, D=False):
        if G:
            for p in self.G.parameters(): p.requires_grad = G
        if D:
            for p in self.D.parameters(): p.requires_grad = D

    def compile(self, config):
        if 'dloss' in config.keys(): self.dloss = config.dloss
        if 'gloss' in config.keys(): self.gloss = config.gloss
        if 'goptim' in config.keys():
            self.goptim = config.goptim(self.G.parameters())
        if 'doptim' in config.keys():
            self.doptim = config.doptim(self.D.parameters())

    def save(self, saver, separately=True):
        th.save(self.G.state_dict(), saver.format('g') + '.sd')
        th.save(self.D.state_dict(), saver.format('d') + '.sd')

    def load(self, loader=None, generator=None, discriminator=None):
        if loader: self.load_state_dict(th.load(loader))
        if generator: self.G.load_state_dict(th.load(generator))
        if discriminator: self.D.load_state_dict(th.load(discriminator))

    def train(self, config, datagen, callback):
        if isinstance(datagen, tuple): traingen, testgen = datagen
        else: traingen, testgen = datagen(config.xdim)

        for epoch in range(1, 1 + config.epochs):
            banner = 'Epoch {:4}/{:4}'.format(epoch, config.epochs)
            print(banner)
            ebar = enumerate(traingen, 1)

            for batch,(xreal,_) in ebar:
                self.forwardbackward(config, traingen, callback)

                args = (batch + (len(traingen) * (epoch - 1)), testgen, config)
                threading.Thread(target=callback.run, args=args).start()
