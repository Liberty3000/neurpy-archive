import neurpy, numpy as np
import torch as th, torchvision
import torch.nn.functional as F
import math, time, random, tqdm
from neurpy import util
from neurpy.adversarial.loss import gradient_penalty
from neurpy.adversarial.models.model import Model

class StyleGAN(Model):
    def __init__(self, config):
        super().__init__()
        self.G = neurpy.config.module(config.generator + '.py').Generator()
        self.D = neurpy.config.module(config.discriminator + '.py').Discriminator()
        self.progression = lambda step:4 * (2**step)
        self.xdim, self.zdim, self.ddim = config.xdim, config.zdim, config.ddim

    def generate(self, z, step=0, alpha=1):
        return self.G(z, step=step, alpha=alpha)

    def inference(self, saver, datagen, step=0, alpha=1, style_weight=0.7, itrs=10):
        x,_ = next(iter(datagen))

        zinterps = 64
        mean_style_noise = th.randn(1024, self.zdim)
        zs = th.randn(zinterps, self.zdims)

        mean_style = 0
        for itr in range(itrs):
            mean_style += self.G.mean_style(mean_style_noise)
        mean_style /= (itrs - 1)

        output = self.G(zs, step=step, alpha=1, mean_style=mean_style, style_weight=style_weight)

        save_as = saver.format('sample')
        save_image(output, save_as, padding=0)#, normalize=True, range=(-1,1))

    def accumulate(self, src, dst, decay=0.999):
        src = dict(src.named_parameters())
        dst = dict(dst.named_parameters())

        for key in src.keys():
            dst[key].data.mul_(decay).add_(1 - decay, src[key].data)

    def decay_lrate(self, optimizer, lr=1e-3):
        for group in optimizer.param_groups:
            group['lr'] = lr * group.get('mult', 1)

    def test(self, config, datagen, generate=True, interp_z=True):
        generator = lambda x:self.G(x, step=self.step, alpha=self.alpha)
        sampler = lambda:th.randn(1, self.zdim, device=config.device)

        # G(z)
        with th.no_grad():
            if generate:
                neurpy.generate.sample(generator=generator,
                                       n=config.ntest,
                                       saver=config.saver,
                                       sampler=sampler,
                                       device=config.device)

    def train(self, config, datagen, callback):
        step = int(math.log2(config.xdim_init[-1])) - 2
        progress = lambda x:4*(2**x)
        resolution = progress(step)
        max_step= int(math.log2(config.xdim[-1])) - 2
        bsizes = {4:512, 8:64, 16:32, 32:4, 64:4, 128:2, 256:2, 512:2, 1024:1}

        traingen,testgen = datagen(size=config.xdim_init)
        bar = tqdm.tqdm(range(config.iterations))
        samples = 0
        for i in bar:
            phase = len(traingen)
            alpha = min(1, 1 / phase * (samples + 1))

            if samples > phase * 2:
                step += 1
                if step > max_step: step = max_step
                else: alpha, samples = 0,0,

                resl = progress(step)
                bsize = bsizes[resl]
                traingen,testgen = datagen(size=(3,resl,resl), bsize=bsize)

            self.decay_lrate(self.goptim)
            self.decay_lrate(self.doptim)
            self.step, self.alpha = step, alpha

            loader = iter(traingen)
            try:
                xreal, label = next(loader)
            except (OSError, StopIteration):
                loader = iter(loader)
                xreal, label = next(loader)
            samples += xreal.shape[0]
            bsize = xreal.size(0)
            xreal = xreal.to(config.device)
            label = label.to(config.device)

            if config.mixing_regularization and random.random() < 0.9:
                gen_in11, gen_in12, gen_in21, gen_in22 = th.randn(4, bsize, self.zdim, device='cuda').chunk(4, 0)
                gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
                gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
            else:
                gen_in1, gen_in2 = th.randn(2, bsize, self.zdim, device=config.device).chunk(2,0)
                zfake, gzfake = gen_in1.squeeze(0), gen_in2.squeeze(0)

            xfake = self.G(zfake, step=step, alpha=alpha)
            dfake = self.D(xfake, step=step, alpha=alpha)

            self.D.zero_grad()
            xreal.requires_grad = True
            dreal = self.D(xreal, step=step, alpha=alpha)
            dreal = F.softplus(-dreal).mean()
            dreal.backward(retain_graph=True)

            grad_real = th.autograd.grad(outputs=dreal.sum(), inputs=xreal, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            Gpeny = grad_penalty.item()

            dfake = F.softplus(dfake).mean()
            dfake.backward()
            Dloss = (dreal + dfake).item()
            self.doptim.step()

            if (i + 1) % config.critics == 0:
                util.requires_grad(self.G, True)
                util.requires_grad(self.D, False)

                self.goptim.zero_grad()
                xfake = self.G( gzfake, step=step, alpha=alpha)
                dfake = self.D(xfake, step=step, alpha=alpha)

                Gloss = config.gloss(dfake)

                Gloss.backward()
                self.goptim.step()

                util.requires_grad(self.G, False)
                util.requires_grad(self.D, True)


            callback.status['Gloss'].append(Gloss.item())
            callback.status['Dloss'].append(Dloss)
            callback.status['alpha'].append(alpha)

            bar.set_description(callback.report())

            args = (i, testgen, config)
            callback.run(*args)
