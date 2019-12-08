import click, glob, json, os, sys, tqdm
import imageio, torchvision
import numpy as np, torch

from neurpy.config import initialize
from neurpy import util

@click.command()
@click.argument('config', default='config/stylegan_1024x1024_v1.json')
def run(config):
    with open(config, 'r') as f: args = json.load(f)
    model, datagen, cfg = initialize(args)
    print(cfg.experiment)

    saver = os.path.join(cfg.saver, 'generations.png')
    ims = []
    for _ in range(128):
        z = util.sampler(cfg.zshape).to(cfg.device)
        x = model.generate(z).squeeze(0)
        ims.append(x)
    torchvision.utils.save_image(ims, saver, padding=0)

if __name__ == '__main__':
    run()
