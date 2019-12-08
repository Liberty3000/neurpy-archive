import click, glob, json, os, sys, tqdm
import imageio, torchvision
import numpy as np

from neurpy.config import initialize
from neurpy import util

@click.command()
@click.argument('config', default='config/stylegan_1024x1024_v1.json')
def run(config):
    with open(config, 'r') as f: args = json.load(f)
    model, datagen, cfg = initialize(args)
    print(cfg.experiment)

    steps,N,dur = cfg.swerve

    saver = os.path.join(cfg.saver, 'swerve')
    os.makedirs(saver)
    itr = 0
    z1 = util.sampler(cfg.zshape)
    for step in tqdm.tqdm(range(steps)):
        z2 = util.sampler(cfg.zshape)
        zs = util.slerp(N, z1=z1.numpy(), z2=z2.numpy())
        for n,z in enumerate(zs):
            save_as = os.path.join(saver, str(itr).zfill(4))
            z = z.to(cfg.device).unsqueeze(0)
            x = model.generate(z)
            torchvision.utils.save_image(x, os.path.join(saver, save_as) + '.png', padding=0)
            itr += 1
        z1 = z2

    imfiles = sorted(list(glob.glob(os.path.join(saver, '*.png'))))
    imfiles = imfiles + imfiles[::-1]
    save_as = os.path.join(saver, str(itr).zfill(4)) + '.gif'
    with imageio.get_writer(save_as, mode='I', duration=dur) as writer:
        for imfile in tqdm.tqdm(imfiles): writer.append_data(imageio.imread(imfile))

if __name__ == '__main__':
    run()
