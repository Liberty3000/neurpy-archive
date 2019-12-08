import click, glob, json, os, sys, tqdm
import imageio, torchvision
import numpy as np, torch

from neurpy.config import initialize
from neurpy import util

def sample(generator, n, saver, sampler, device=torch.device('cpu')):
    path = util.save_to(saver, 'generations')
    with torch.no_grad():
        ims = []
        for _ in range(n):
            z = sampler().to(device)
            x = generator(z).squeeze(0)
            ims.append(x)
        torchvision.utils.save_image(ims, path.format('zx') + '.png', padding=0, nrow=int(np.sqrt(n)))

@click.command()
@click.argument('config', default='config/stylegan_1024x1024_v1.json')
def run(config):
    with open(config, 'r') as f: args = json.load(f)
    model, datagen, cfg = initialize(args)
    print(cfg.experiment)

    if hasattr(model,'decoder'): generator = model.decoder
    if hasattr(model,'generator'): generator = model.generator
    if hasattr(model,'G'): generator = model.G

    sample(generator, cfg.ntest, cfg.saver,
    sampler=lambda:util.sampler(cfg.zdim), device=cfg.device)


if __name__ == '__main__':
    run()
