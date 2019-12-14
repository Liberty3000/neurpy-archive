import importlib, os, pathlib, threading
import torch as th
from torch.autograd import Variable
import neurpy
from neurpy import util
from neurpy.adversarial.models.model import Model

class AAutoencoder(Model):
    def __init__(self, config):
        super().__init__()
        eargs = [config.xdim, config.zdim]
        gargs = [config.zdim, config.xdim, config.goutputfn]
        dargs = [config.zdim, config.ddim, config.doutputfn]
        self.encoder = neurpy.config.module('{}.py'.format(config.encoder)).Encoder(*eargs)
        self.decoder = neurpy.config.module('{}.py'.format(config.decoder)).Decoder(*gargs)
        self.discrim = neurpy.config.module('{}.py'.format(config.discriminator)).Discriminator(*dargs)
        self.xdim, self.zdim, self.ddim = config.xdim, config.zdim, config.ddim

    def save(self, saver, separately=True):
        th.save(self.encoder.state_dict(),saver.format('encoder') + '.sd')
        th.save(self.decoder.state_dict(),saver.format('decoder') + '.sd')
        th.save(self.discrim.state_dict(),saver.format('discrim') + '.sd')

    def load(self, loader=None, encoder=None, decoder=None, discrim=None):
        if loader:  self.load_state_dict(torch.load(loader))
        if encoder: self.encoder.load_state_dict(torch.load(encoder))
        if decoder: self.decoder.load_state_dict(torch.load(decoder))
        if discrim: self.discrim.load_state_dict(torch.load(discrim))

    def compile(self, config):
        self.gloss, self.dloss  = config.gloss, config.dloss
        self.encoder_optim = config.encoder_optim(self.encoder.parameters())
        self.decoder_optim = config.decoder_optim(self.decoder.parameters())
        self.discrim_optim = config.discrim_optim(self.discrim.parameters())

    def Gparams(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def Dparams(self):
        return list(self.discrim.parameters())

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def generate(self, z):
        return self.decoder(z)

    def discriminate(self, z):
        return self.discrim(z)

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        d = self.discriminate(z)
        return y,z,d

    def test(self, config, datagen, generate=True, interp_z=True, interp_x=True, reconstruct=True):
        # G(z)
        if generate:
            neurpy.generate.sample(generator=self.decoder, n=config.ntest,
                                   saver=config.saver, sampler=lambda:util.sampler(self.zdim),
                                   device=config.device)
        # G(z)
        if interp_z:
            neurpy.interpolate.spherical(sampler=lambda:util.sampler(self.zdim),
                                         decoder=self.decoder,
                                         interps=config.zinterps,
                                         saver=config.saver,
                                         device=config.device)
        # G(E(x))
        if reconstruct:
            neurpy.autoencoder.reconstruct.sample(encoder=self.encoder, decoder=self.decoder,
            datagen=datagen, saver=config.saver, device=config.device)
        # G(E(x))
        if interp_x:
            neurpy.interpolate.spherical(encoder=self.encoder,
                                         decoder=self.decoder,
                                         interps=config.xinterps,
                                         saver=config.saver,
                                         datagen=datagen,
                                         device=config.device)

    def forwardbackward(self, config, datagen, callback):
        xreal,_ = next(iter(datagen))
        bsize = xreal.size(0)

        xreal = xreal.to(config.device)

        yreal,zreal,dreal = self.forward(xreal)
        latent = util.sampler(self.zdim, bsize)
        zfake = Variable(torch.Tensor(latent)).to(config.device)
        real,fake = util.labels(self.ddim, bsize=bsize, device=config.device)

        # reconstruction phase
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        Gloss_ae = self.gloss(yreal, xreal)
        Gloss_ae.backward(retain_graph=True)
        self.decoder_optim.step()
        self.encoder_optim.step()

        # regularization phase
        self.discrim_optim.step()
        dfake = self.discriminate(zfake)
        dreal = dreal.view(-1, self.ddim)
        dfake = dfake.view(-1, self.ddim)
        Dloss_real = self.dloss(dreal, real)
        Dloss_fake = self.dloss(dfake, fake)
        Dloss = Dloss_real + Dloss_fake
        Dloss.backward(retain_graph=True)
        self.discrim_optim.step()

        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        Gloss_gan = self.dloss(dfake, real)
        Gloss_gan.backward()
        self.decoder_optim.step()
        self.encoder_optim.step()

        callback.status['Autoencoder_Loss'].append(Gloss_ae.item())
        callback.status['GAN_Loss'].append(Gloss_gan.item())
        callback.status['Discriminator_Loss'].append(Dloss.item())
