from collections import defaultdict
import threading, tqdm
import torch as th
from neurpy import util

class Model(th.nn.Module):
    def __init__(self):
        super().__init__()

    def build(self):
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

    def forwardbackward(self):
        raise NotImplementedError()

    def toggle_parameters(self, G=True, D=False):
        if G:
            for p in self.G.parameters(): p.requires_grad = G
        if D:
            for p in self.D.parameters(): p.requires_grad = D

    def compile(self, config):
        self.dloss, self.gloss = config.dloss, config.gloss
        self.goptim = config.goptim(self.G.parameters())
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

        ebar = tqdm.tqdm(range(1, 1 + config.epochs))
        for epoch in ebar:
            banner = 'Epoch {:4}/{:4}'.format(epoch, config.epochs)
            ebar.set_description(banner)
            bbar = tqdm.tqdm(enumerate(traingen, 1),total=len(traingen))
            for batch,(xreal,_) in bbar:
                self.forwardbackward(config, traingen, callback)

                args = (batch + (len(traingen) * (epoch - 1)), testgen, config)
                threading.Thread(target=callback.run, args=args).start()
