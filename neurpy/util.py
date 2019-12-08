import imageio, os, random, sklearn
import matplotlib.pyplot as mp, numpy as np, torch as th
from torchvision.utils import save_image

def save_to(saver, dirs='outputs'):
    path = saver[:saver.rindex('/')] + '/{}'.format(dirs)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, saver.rsplit('/')[-1])
    return path

def gif(saver, tensor=None, imfiles=None, reverse=True):
    if imfiles:
        imfiles = sorted(list(glob.glob(os.path.join(saver, '*.png'))))
        if reverse: imfiles = imfiles + imfiles[::-1]
        save_as = os.path.join(saver, str(itr).zfill(4)) + '.gif'
        with imageio.get_writer(save_as, mode='I', duration=dur) as writer:
            for imfile in tqdm.tqdm(imfiles):
                writer.append_data(imageio.imread(imfile))

def summary(model):
    print(model)
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    print('params| {:,}'.format(sum(params)))

def labels(yshape, bsize, rmin=1, rmax=1, fmin=0, fmax=0, device=th.device('cpu')):
    if yshape == 1:
        real = th.FloatTensor([np.random.uniform(rmin, rmax)
        for _ in range(bsize)]).unsqueeze(-1)
        fake = th.FloatTensor([np.random.uniform(fmin, fmax)
        for _ in range(bsize)]).unsqueeze(-1)
    if yshape == 2:
        real = [[np.random.uniform(rmin,rmax),np.random.uniform(fmin,fmax)] for _ in range(bsize)]
        fake = [[np.random.uniform(fmin,fmax),np.random.uniform(rmin,rmax)] for _ in range(bsize)]
        real = th.FloatTensor(real)
        fake = th.FloatTensor(fake)
    return real.to(device), fake.to(device)

def hypersphere(z, radius=1):
    return z * radius / z.norm(p=2, dim=1, keepdim=True)

def sampler(zdim, bsize=1, expand=False, device=th.device('cpu')):
    tensor = th.randn((bsize, zdim))
    if expand: tensor = tensor.unsqueeze(-1).unsqueeze(-1)
    return tensor.to(device)

def requires_grad(model, flag=True):
    for p in model.parameters(): p.requires_grad = flag

def onehot(labels, ydim=2, device=th.device('cpu')):
    return th.eye(ydim)[labels].to(device)

def categorical(ydim=0, bsize=1, label=None, device=th.device('cpu')):
    if label:
        ys = np.zeros(ydim)
        ys[label] = 1
        ys = [ys]
    else: ys = np.eye(ydim)
    return th.Tensor([random.choice(ys) for _ in range(bsize)]).to(device)

def walk(decoder, sampler, n=32, steps=128):
    z1 = sampler()
    for _ in range(steps):
        z2 = sampler()
        zs = util.slerp(n=n, z1=z1, z2=z2, sampler=sampler)
        zs = [decoder(z.unsqueeze(0)) for z in zs]
        yield zs
        z1 = z2

def lerp(n, sampler=None, z1=None, z2=None):
    z1 = th.FloatTensor(sampler()) if z1 is None else z1
    z2 = th.FloatTensor(sampler()) if z2 is None else z2
    zs = [(z1*x) + z2 * (1 - x) for x in np.linspace(0,1,int(n))]
    return th.stack(zs)

def slerp(n, sampler=None, z1=None, z2=None, epsl=1e-10):
    batch = []
    for b in range(n):
        zs = []
        z1 = sampler().cpu().numpy() if z1 is None else z1
        z2 = sampler().cpu().numpy() if z2 is None else z2
        for x in np.linspace(0,1,n):
            lo,hi = (z1 / np.linalg.norm(z1)), (z2 / np.linalg.norm(z2))
            omega = np.arccos(np.dot(lo,hi.T))
            left = np.sin((1.0 - x) * omega) / (np.sin(omega) * lo)
            rght = np.sin(x * omega) / (np.sin(omega) * hi)
            zs.append(th.FloatTensor(left) + th.FloatTensor(rght))
        batch.append(zs)
    batch = th.cat(tuple(zs))
    return batch
