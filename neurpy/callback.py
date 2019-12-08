from collections import defaultdict
import os
import imageio, PIL
import numpy as np, matplotlib.pyplot as mp
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from neurpy import util

def save_to(saver, dirs='constants'):
    path = saver[:saver.rindex('/')] + '/{}'.format(dirs)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, saver.rsplit('/')[-1])
    return path

class Callback(object):
    def __init__(self, model, experiment, saver,
    test_interval=0, save_interval=0):
        self.model = model
        self.experiment = experiment
        self.save_interval = save_interval
        self.test_interval = test_interval
        self.saver = saver if not saver == '' else str(os.getcwd())
        self.xconst, self.zvalid = None, None
        self.status = defaultdict(list)

    def report(self):
        string = str()
        for k,v in self.status.items(): string += '{} {:.4f} '.format(k, v[-1])
        return string

    def log(self, saver):
        save_as = os.path.join(saver, 'stats.csv')
        mode = 'a' if os.path.exists(save_as) else 'w'
        stats = open(save_as, mode)
        line = ''
        if mode == 'w':
            for k in self.status.keys(): line += '{},'.format(k)
            stats.write(line[:-1] + '\n')
            line = ''
        for k,v in self.status.items(): line += '{},'.format(v.pop())
        stats.write(line[:-1] + '\n')
        stats.close()

    def plot(self):
        self.colors = random.sample(util.colors, len(self.status.items()))

        naxs = len(self.status.items())

        fig,axs = mp.subplots(naxs, 1, figsize=(17,7))
        fig.suptitle(self.experiment.rsplit('-'), fontsize=16)
        if naxs < 2: axs = [axs]

        for itr,(ax,(k,v)) in enumerate(zip(axs,self.status.items())):
            ax.plot(range(len(v)), v, label=k, color=self.colors[itr], linewidth=2)
            ax.legend()

            if itr == len(axs) // 2:
                ax.set_ylabel('Loss')

            if itr == len(axs) - 1:
                ax.set_xlabel('Minibatch Iterations')
                ax.set_xticks([int(x) for x in ax.get_xticks()[::2]])
            else:
                ax.set_xticks([])
            ax.set_yticks(ax.get_yticks()[::2])


        fig.savefig(self.saver + '/training_plot.png')
        mp.clf()

    def run(self, batch, datagen, config, conditional=False):
        config.saver = self.saver + '/{}' + '-{}'.format(str(batch).zfill(5))

        # save all model parameters to the weights directory
        if batch % self.save_interval == 0 or batch == 1:
            saver = self.saver + '/weights'
            os.makedirs(saver, exist_ok=True)
            saver = saver + '/{}' + '.{}'.format(str(batch).zfill(5))
            self.model.save(saver)
        if batch % self.test_interval == 0 or batch in [1,16]:
            self.model.test(config, datagen)


    def inference(self, datagen, config):
        model, saver = self.model, config.saver
        sampler = lambda:util.sampler(config.latent_shape)
        grid = int(np.sqrt(config.ntest))

        if self.xconst is None:
            self.xconst,_ = next(iter(datagen))
            save_image(self.xconst, saver[:saver.rindex('/')] + '/validation_data.png', padding=0, nrow=grid)

        with torch.no_grad():
            if hasattr(config, 'label_shape'): # class conditional
                path = save_to(saver, 'generations')
                z = Variable(torch.Tensor(util.sampler(config.latent_shape, config.ntest))).to(config.device)
                classes = np.random.randint(0,config.label_shape, config.ntest)
                y = torch.LongTensor(classes).long().to(config.device)
                x = model.generate(z,torch.eye(config.label_shape)[y].to(config.device))
                save_image(x, path.format('zx') + '.png', padding=0, nrow=grid)

                path = save_to(saver, 'conditionals')
                for label in range(config.label_shape):
                    z = Variable(torch.Tensor(util.sampler(config.latent_shape, config.ntest))).to(config.device)
                    classes = np.random.randint(0,config.label_shape, config.ntest)
                    y = torch.LongTensor(classes).long().to(config.device)
                    x = model.generate(z,torch.eye(config.label_shape)[y].to(config.device))
                    save_image(x, path.format('class.{}'.format(label)) + '.png', padding=0, nrow=grid)
                return

            if config.input_shape != config.latent_shape and not hasattr(config, 'gydim'):
                if self.zvalid is None:
                    self.zvalid = util.sampler(config.latent_shape, config.ntest)
                zvalid = torch.Tensor(self.zvalid).to(config.device)
                const = model.generate(zvalid)
                save_image(const, save_to(saver, 'constants').format('const') + '.png', padding=0, nrow=grid)

                path = save_to(saver, 'interpolations')

                if hasattr(config, 'zinterps'):
                    # interpolate z sampled from prior distribution
                    for itr,n in enumerate(config.zinterps):
                        zs = util.slerp(n=n, sampler=sampler)
                        zs = zs.to(config.device)
                        zs = torch.cat([model.generate(z.unsqueeze(0)) for z in zs],dim=0)
                        ims = [z.detach().cpu().numpy()*255 for z in zs]

                        ims = np.asarray(ims + ims[::-1]).astype(np.uint8)
                        ims = ims.swapaxes(1,-1).swapaxes(1,2)
                        imageio.mimsave(path.format('zspace-{}.z.{}'.format(n, itr)) + '.gif', ims)
                        save_image(zs, path.format('slerp-{}.z.{}'.format(n, itr)) + '.png', padding=0, nrow=int(np.sqrt(n)))

                # interpolate z sampled from encoded data distribution
                if hasattr(config, 'xinterps') and hasattr(model, 'encoder'):
                    x1,y1 = next(iter(datagen))
                    x2,y2 = next(iter(datagen))
                    x1 = x1[0].unsqueeze(0).to(config.device)
                    x2 = x2[0].unsqueeze(0).to(config.device)
                    z1, z2 = model.encode(x1), model.encode(x2)
                    z1, z2 = z1.view(1,-1), z2.view(1,-1)
                    z1, z2 = z1.cpu().numpy(), z2.cpu().numpy()
                    for itr,n in enumerate(config.xinterps):
                        zs = util.slerp(z1=z1, z2=z2, n=n)
                        zs = zs.to(config.device)
                        i = torch.cat([model.decode(z.unsqueeze(0)) for z in zs],dim=0)
                        i = torch.cat((torch.cat((x1,i)),x2),dim=0)
                        i = i[1:-1]

                        ims = [z.detach().cpu().numpy()*255 for z in i]
                        ims = np.asarray(ims + ims[::-1]).astype(np.uint8)
                        ims = ims.swapaxes(1,-1).swapaxes(1,2)
                        save_as = path.format('zspace-{}.x.{}'.format(n, itr)) + '.gif'
                        imageio.mimsave(save_as, ims)
                        save_as = path.format('slerp-{}.x.{}'.format(n, itr)) + '.png'
                        save_image(i, save_as, padding=0, nrow=int(np.sqrt(n)))

                path = save_to(saver, 'reconstructions')
                x,_ = next(iter(datagen))
                x = x.to(config.device)
                z = model.encode(x)
                z = z.to(config.device)
                y = model.decode(z)
                save_image(x, path.format('xinpu') + '.png', padding=0, nrow=grid)
                save_image(y, path.format('ypred') + '.png', padding=0, nrow=grid)

                path = save_to(saver, 'generations')
                xz = util.sampler(config.latent_shape, config.ntest)
                xz = xz.to(config.device)
                zx = torch.cat([model.generate(x.unsqueeze(0)) for x in xz],dim=0)
                save_image(zx, path.format('zx') + '.png', padding=0, nrow=grid)

            if config.input_shape == config.latent_shape:
                path = save_to(saver, 'reconstructions')
                x,z = next(iter(datagen))
                temp = x.clone()
                x = x.to(config.device)
                y = model.generate(x)
                im = torch.cat((z,x.cpu(),y.cpu()),dim=0)
                save_image(im, path.format('output') + '.png', padding=0, nrow=grid)


class Notebook(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notebook = True

    def plot(self):
        super().plot()
        mp.show()

    def run(self, batch, datagen, config, plot=False):
        banner = 'Batch {:4} '.format(batch) + self.report()
        config.saver = self.saver + '/{}' + '-{}'.format(str(batch).zfill(5))

        if batch % self.save_interval == 0 or batch == 1:
            saver = self.saver + '/weights'
            os.makedirs(saver, exist_ok=True)
            self.model.save(os.path.join('{}' + '.{}'.format(str(batch).zfill(8))))

        if batch % self.test_interval == 0 or batch in [1,16]:
            print(banner)
            if plot: self.plot()
            self.inference(datagen, config)

    def inference(self, datagen, config):
        model, saver = self.model, config.saver
        sampler = lambda:util.sampler(config.zshape)
        grid = int(np.sqrt(config.ntest))

        from IPython.display import display, Image

        if self.xconst is None:
            self.xconst,_ = next(iter(datagen))
            save_image(self.xconst, saver[:saver.rindex('/')] + '/validation_data.png', padding=0, nrow=grid)

        with torch.no_grad():
            if hasattr(config, 'label_shape'):
                for y in range(config.label_shape):
                    z = Variable(torch.Tensor(util.sampler(config.latent_shape, config.ntest))).to(config.device)
                    yfake = torch.LongTensor([y]*config.ntest).to(config.device)
                    i = torch.eye(config.label_shape)
                    x = self.G(z, i[yfake].to(config.device))
                    save_image(const, save_to(saver, 'conditionals').format('class-{}'.format(y)) + '.png', padding=0, nrow=grid)
                return

            if config.xshape != config.zshape:
                if self.zvalid is None:
                    self.zvalid = util.sampler(config.zshape, config.ntest)
                zvalid = torch.Tensor(self.zvalid).to(config.device)
                const = model.generate(zvalid)
                save_image(const, save_to(saver, 'constants').format('const') + '.png', padding=0, nrow=grid)

                path = save_to(saver, 'interpolations')
                # interpolate z sampled from prior distribution
                for itr,n in enumerate(config.zinterps):
                    zs = util.slerp(n=n, sampler=sampler)
                    zs = zs.to(config.device)
                    zs = torch.cat([model.generate(z.unsqueeze(0)) for z in zs],dim=0)
                    ims = [z.cpu().numpy()*255 for z in zs]

                    ims = np.asarray(ims + ims[::-1]).astype(np.uint8)
                    ims = ims.swapaxes(1,-1).swapaxes(1,2)
                    imageio.mimsave(path.format('zspace-{}.z.{}'.format(n, itr)) + '.gif', ims)
                    save_image(zs, path.format('slerp-{}.z.{}'.format(n, itr)) + '.png', padding=0, nrow=int(np.sqrt(n)))

                if self.notebook:
                    print('Interpolate G(z~p(z))')
                    display(Image(save_as))

                # interpolate z sampled from encoded data distribution
                x1,y1 = next(iter(datagen))
                x2,y2 = next(iter(datagen))
                x1 = x1[0].unsqueeze(0).to(config.device)
                x2 = x2[0].unsqueeze(0).to(config.device)
                z1, z2 = model.encode(x1), model.encode(x2)
                z1, z2 = z1.view(1,-1), z2.view(1,-1)
                z1, z2 = z1.cpu().numpy(), z2.cpu().numpy()
                for itr,n in enumerate(config.xinterps):
                    zs = util.slerp(z1=z1, z2=z2, n=n)
                    zs = zs.to(config.device)
                    i = torch.cat([model.decode(z.unsqueeze(0)) for z in zs],dim=0)
                    i = torch.cat((torch.cat((x1,i)),x2),dim=0)
                    i = i[1:-1]

                    ims = [z.cpu().numpy()*255 for z in i]
                    ims = np.asarray(ims + ims[::-1]).astype(np.uint8)
                    ims = ims.swapaxes(1,-1).swapaxes(1,2)

                    save_as = path.format('zspace-{}.x.{}'.format(n, itr)) + '.gif'
                    imageio.mimsave(save_as, ims)
                    save_as = path.format('slerp-{}.x.{}'.format(n, itr)) + '.png'
                    save_image(i, save_as, padding=0, nrow=int(np.sqrt(n)))
                    if self.notebook:
                        print('Interpolate G(E(z|x))')
                        display(Image(save_as))

                path = save_to(saver, 'reconstructions')
                x,_ = next(iter(datagen))
                x = x.to(config.device)
                z = model.encode(x)
                z = z.to(config.device)
                y = model.decode(z)
                save_image(x, path.format('xinpu') + '.png', padding=0, nrow=grid)
                save_image(y, path.format('ypred') + '.png', padding=0, nrow=grid)
                if self.notebook:
                    print('Input')
                    display(Image(path.format('xinpu') + '.png'))
                    print('Output')
                    display(Image(path.format('ypred') + '.png'))

                path = save_to(saver, 'generations')
                xz = util.sampler(config.zshape, config.ntest)
                xz = xz.to(config.device)
                zx = torch.cat([model.generate(x.unsqueeze(0)) for x in xz],dim=0)
                save_image(zx, path.format('zx') + '.png', padding=0, nrow=grid)
                if self.notebook:
                    print('Random Sample')
                    display(Image(path.format('zx') + '.png'))
                    print('Validation Sample')
                    display(Image(path.format('const') + '.png'))

            if config.xshape == config.zshape:
                path = save_to(saver, 'reconstructions')
                x,z = next(iter(datagen))
                temp = x.clone()
                x = x.to(config.device)
                y = model.generate(x)
                im = torch.cat((z,x.cpu(),y.cpu()),dim=0)
                save_image(im, path.format('output') + '.png', padding=0, nrow=grid)
                if self.notebook:
                    print('Input')
                    display(Image(path.format('source') + '.png'))
                    print('Target')
                    display(Image(path.format('target') + '.png'))
                    print('Output')
                    display(Image(path.format('output') + '.png'))
