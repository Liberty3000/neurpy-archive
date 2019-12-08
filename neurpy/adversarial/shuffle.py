import click, json, os, random, tqdm
import imageio, torchvision
from torchvision.utils import save_image
import numpy as np, torch as th
from neurpy.config import initialize
from neurpy import util

# takes a batch of samples from the training or test set, where n % 2 == 0,
# where half the images are from the dataset, and half are synthesized from
# a generative model
def sample(generator, datagen, n, saver, sampler, device=th.device('cpu')):
    path = os.path.join(saver, 'shuffle')

    idxs = list(range(n))
    random.shuffle(idxs)
    reals, fakes = idxs[:n//2], idxs[n//2:]

    with th.no_grad():
        ims = []
        for itr in tqdm.tqdm(range(n)):
            if itr in reals:
                z = sampler().to(device)
                z = z.view(z.size(0),-1,1,1)
                x = generator(z).squeeze(0)
            elif itr in fakes:
                x,_ = next(iter(datagen))
                x = x[0]

            ims.append(x.cpu())

        save_image(ims, path.format('mix') + '.png', padding=0, nrow=int(np.sqrt(n)))

@click.command()
@click.argument('config', default='config/stylegan_1024x1024_v1.json')
def run(config):
    with open(config, 'r') as f: args = json.load(f)
    model, datagen, cfg = initialize(args)
    print(cfg.experiment)

    if hasattr(model,'decoder'): generator = model.decoder
    if hasattr(model,'generator'): generator = model.generator
    if hasattr(model,'G'): generator = model.G

    _,testgen = datagen(cfg.xdim)

    sample(generator, testgen, cfg.ntest, cfg.saver,
    sampler=lambda:util.sampler(cfg.zdim), device=cfg.device)


if __name__ == '__main__':
    run()
