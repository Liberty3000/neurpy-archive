import click, json, os, tqdm
import imageio, torchvision
import numpy as np
import torch
from torchvision.utils import save_image
from neurpy.config import initialize
from neurpy import util

def decode(decoder, latent, save_as, device=torch.device('cpu'), reverse=True):
    with imageio.get_writer(save_as, mode='I') as writer:
        xs = []
        for n,z in enumerate(latent):
            z = z.to(device).unsqueeze(0)
            xs += [decoder(z)]
            writer.append_data(x.numpy())
        if reverse:
            for x in xs[::-1]: xs += [x]
    return xs

def spherical(decoder, interps, saver,
              sampler=None, encoder=None, datagen=None, device=torch.device('cpu')):
    path = util.save_to(saver, 'interpolations')
    with torch.no_grad():
        if datagen and encoder:
            x1,y1 = next(iter(datagen))
            x2,y2 = next(iter(datagen))
            x1 = x1[0].unsqueeze(0).to(device)
            x2 = x2[0].unsqueeze(0).to(device)
            z1, z2 = encoder(x1), encoder(x2)
            z1, z2 = z1.view(1,-1), z2.view(1,-1)
            z1, z2 = z1.cpu().numpy(), z2.cpu().numpy()

            for itr,n in enumerate(interps):
                zs = util.slerp(z1=z1, z2=z2, n=n)
                zs = zs.to(device).view(zs.size(0),-1,1,1)
                i = torch.cat([decoder(z.unsqueeze(0)) for z in zs],dim=0)
                i = torch.cat((torch.cat((x1,i)),x2),dim=0)
                i = i[1:-1]

                # deprocess
                ims = [z.detach().cpu().numpy()*255 for z in i]
                ims = np.asarray(ims + ims[::-1]).astype(np.uint8)
                ims = ims.swapaxes(1,-1).swapaxes(1,2)

                save_gif_as = path.format('zspace-{}.x.{}'.format(n, itr)) + '.gif'
                imageio.mimsave(save_gif_as, ims)
                save_grid_as = path.format('slerp-{}.x.{}'.format(n, itr)) + '.png'
                save_image(i, save_grid_as, padding=0, nrow=int(np.sqrt(n)))
        else:
            for itr,n in enumerate(interps):
                zs = util.slerp(n=n, sampler=sampler)
                zs = zs.to(device).view(zs.size(0),-1,1,1)
                zs = torch.cat([decoder(z.unsqueeze(0)) for z in zs],dim=0)
                ims = [z.detach().cpu().numpy()*255 for z in zs]

                ims = np.asarray(ims + ims[::-1]).astype(np.uint8)
                ims = ims.swapaxes(1,-1).swapaxes(1,2)

                save_gif_as = path.format('zspace-{}.z.{}'.format(n, itr)) + '.gif'
                imageio.mimsave(save_gif_as, ims)
                save_grid_as = path.format('slerp-{}.z.{}'.format(n, itr)) + '.png'
                save_image(zs, save_grid_as, padding=0, nrow=int(np.sqrt(n)))


def linear(decoder, N, *args):
    zs = util.lerp(N, sampler=lambda:util.sampler(decoder.zdim))
    saver = os.path.join(saver, 'lerp.{}'.format(itr))
    os.makedirs(saver)
    return decode(decoder, zs, saver, *args)


@click.command()
@click.argument('config', default='config/stylegan_1024x1024_v1.json')
def run(config):
    with open(config, 'r') as f: args = json.load(f)
    model, datagen, cfg = initialize(args)
    print(cfg.experiment)

    for itr,N in enumerate(cfg.zinterps):
        itr = str(itr).zfill(4)
        saver = os.path.join(cfg.saver, 'interpolations')
        os.makedirs(saver, exist_ok=True)

        saver = os.path.join(saver, 'slerp.{}'.format(itr))
        os.makedirs(saver)

        zs = util.slerp(N, sampler=lambda:util.sampler(cfg.zshape))
        for n,z in enumerate(zs):
            save_as = os.path.join(saver, str(n).zfill(4))
            z = z.to(cfg.device).unsqueeze(0)
            x = model.generate(z)
            torchvision.utils.save_image(x, save_as + '.png', padding=0)

        imfiles = sorted(list(glob.glob(os.path.join(saver, '*.png'))))
        imfiles = imfiles + imfiles[::-1]
        save_as = os.path.join(saver, itr) + '.gif'
        with imageio.get_writer(save_as, mode='I') as writer:
            for imfile in imfiles:
                writer.append_data(imageio.imread(imfile))

if __name__ == '__main__':
    run()
