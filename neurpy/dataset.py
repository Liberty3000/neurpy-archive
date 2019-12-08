from collections import defaultdict
import glob, os
from PIL import Image
import numpy as np, torch as th
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.datasets import ImageFolder
from torchvision import transforms

# image-to-label classification or regression;
# supervision where the label is of smaller
# dimensionality than the image; 3D RBG to 2D vector
class ImageRegression(Dataset):
    def __init__(self, home, data, transform):
        super().__init__()
        self.home = home
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        imfile = list(self.data.keys())[index]
        labels = list(self.data.values())[index]
        imfile = os.path.join(self.home, imfile)

        im = Image.open(imfile).convert('RGB')
        im = self.transform(im)
        lb = th.from_numpy(np.array(labels).astype(np.float32))
        return im,lb

    def __len__(self):
        return len(list(self.data.keys()))

# image-to-image translation;
# supervision where the label is also an image
class ImageTranslation(Dataset):
    def __init__(self, home, data, source_domain, target_domain, transform):
        super().__init__()
        self.home = home
        self.data = data
        self.transform = transform
        self.source = source_domain
        self.target = target_domain

    def __getitem__(self, index):
        imfile = self.data[index]
        a = self.source.format(imfile)
        b = self.target.format(imfile)
        im = self.transform(Image.open(a).convert('RGB'))
        lb = self.transform(Image.open(b).convert('RGB'))
        return im,lb

    def __len__(self):
        return len(self.data)


# image-to-image translation; supervision
# where the label is masked image
class ImageSegmentation(Dataset):
    def __init__(self, home, data, source_domain, target_domain, transform):
        super().__init__()
        self.home = home
        self.data = data
        self.transform = transform
        self.source = source_domain
        self.target = target_domain

    def __getitem__(self, index):
        imfile = self.data[index]
        a = self.source.format(imfile)
        b = self.target.format(imfile)
        im = self.transform(Image.open(a).convert('RGB'))
        lb = self.transform(Image.open(b).convert('RGB'))
        return im,lb

    def __len__(self):
        return len(self.data)

def image(imfile, dims=None, batch=True, flips=True):
    if shape is not None: trans += [transforms.Resize(*dims)]
    if flips: trans += [transforms.RandomHorizontalFlip(p=0.5)]
    trans += [transforms.ToTensor()]
    im = trans(Image.open(a).convert('RGB'))
    if batch: im.unsqueeze_(0)
    return im

def images(dataset, shape=(3,64,64), bsize=1, ntest=1, home=os.path.expanduser('~/Developer/data')):
    trans = []
    if shape is not None: trans += [transforms.Resize((shape[-2],shape[-1]))]
    trans += [transforms.RandomHorizontalFlip(p=0.5)]
    trans += [transforms.ToTensor()]
    transform = transforms.Compose(transforms=trans)
    imgs = ImageFolder(os.path.join(home, dataset), transform=transform)
    trdata = DataLoader(dataset=imgs, batch_size=bsize, shuffle=True)
    tsdata = DataLoader(dataset=imgs, batch_size=ntest, shuffle=True)
    return (trdata, tsdata)

def imagenet(shape=(3,256,256), bsize=1, ntest=1, normalize=True, home=os.path.expanduser('~/Developer/data/imagenet')):
    trainfold = os.path.join(home, 'ILSVRC/Data/CLS-LOC/train')
    testfold = os.path.join(home, 'ILSVRC/Data/CLS-LOC/test')

    if shape is not None: traintrans = [transforms.Resize((shape[-2],shape[-1]))]
    traintrans += [transforms.RandomHorizontalFlip(p=0.5)]
    traintrans += [transforms.ToTensor()]
    testtrans = traintrans
    if normalize: traintrans += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    trainset = ImageFolder(os.path.join(home, trainfold), transform=transforms.Compose(transforms=traintrans))
    testset = ImageFolder(os.path.join(home, trainfold), transform=transforms.Compose(transforms=testtrans))

    trdata = DataLoader(dataset=trainset, batch_size=bsize, shuffle=True)
    tsdata = DataLoader(dataset=testset, batch_size=ntest, shuffle=True)
    return (trdata, tsdata)

def celeba(shape=(3,128,128), bsize=1, ntest=1, home=os.path.expanduser('~/Developer/data/celeba')):
    imdir  = os.path.join(home, 'img_align_celeba')
    splitf = os.path.join(home, 'list_eval_partition.txt')
    attrbf = os.path.join(home, 'list_attr_celeba.txt')

    splits = list(open(splitf).readlines())
    attrbs = list(open(attrbf).readlines())
    samples = int(attrbs[0])
    classes = attrbs[1].rstrip().split(' ')

    classes = dict(zip(range(len(classes)),classes))

    trainset = defaultdict(list)
    testset  = defaultdict(list)

    for a,s in zip(attrbs[2:],splits):
        fname,split = s.split(' ')[0], s.split(' ')[1]
        split = int(s.split(' ')[-1])
        attrs = [int(ch) for ch in a.split(' ')[1:] if ch]
        attrs = [ch if ch is 1 in attrs else 0 for ch in attrs]

        if split == 0: trainset[fname] = attrs
        else: testset[fname] = attrs

    trans = []
    if shape is not None: trans += [transforms.Resize((shape[-2],shape[-1]))]
    trans += [transforms.RandomHorizontalFlip(p=0.5)]
    trans += [transforms.ToTensor()]
    traindata = ImageRegression(home=imdir, data=trainset, transform=transforms.Compose(trans))

    testdata = ImageRegression(home=imdir, data=testset, transform=transforms.Compose(trans))

    traingen = DataLoader(traindata, batch_size=bsize, shuffle=True)
    testgen  = DataLoader(testdata,  batch_size=bsize, shuffle=True)

    return (traingen, testgen)

def cifar10(shape=(3,32,32), bsize=1, ntest=1, home=os.path.expanduser('~/Developer/data/cifar10')):
    traintrf = transforms.Compose([
    transforms.Resize((shape[-2],shape[-1])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()])
    trainset = CIFAR10(root=home, train=True, transform=traintrf, download=True)
    traingen = DataLoader(trainset, batch_size=bsize, shuffle=True)

    testtrf = transforms.Compose([transforms.Resize((shape[-2],shape[-1])),transforms.ToTensor()])
    testset = CIFAR10(root=home, train=False, transform=testtrf, download=True)
    testgen = DataLoader(testset, batch_size=ntest, shuffle=True)

    return (traingen, testgen)

def mnist(shape=(1,28,28), bsize=1, home=os.path.expanduser('~/Developer/data/mnist')):
    traintrf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()])
    trainset = MNIST(root=home, train=True, transform=traintrf, download=True)
    traingen = DataLoader(trainset, batch_size=bsize, shuffle=True)

    testtrf = transforms.Compose([transforms.ToTensor()])
    testset = MNIST(root=home, train=False, transform=testtrf, download=True)
    testgen = DataLoader(testset, batch_size=64, shuffle=True)

    return (traingen, testgen)

def facades(shape, bsize=16, ntest=9, a2b=True, home=os.path.expanduser('~/Developer/data/facades')):
    source_domain = '{}.png' if a2b else '{}.jpg'
    target_domain = '{}.jpg' if a2b else '{}.png'

    trdir = os.path.join(home, 'base')
    tsdir = os.path.join(home, 'extended')
    trainset= [f[:-4] for f in glob.glob(os.path.join(trdir, source_domain.format('*')))]
    testset = [f[:-4] for f in glob.glob(os.path.join(tsdir, source_domain.format('*')))]

    trans = []
    if shape is not None: trans += [transforms.Resize((shape[-2],shape[-1]))]
    trans += [transforms.RandomHorizontalFlip(p=0.5)]
    trans += [transforms.ToTensor()]
    transform = transforms.Compose(transforms=trans)

    traindata= ImageTranslation(traindir, trainset,
                source=source_domain, target=target_domain, transform=transform)
    testdata = ImageTranslation(testdir, testset,
               source=source_domain, target=target_domain, transform=transform)

    traingen = DataLoader(traindata, batch_size=bsize, shuffle=True)
    testgen  = DataLoader(testdata,  batch_size=ntest, shuffle=True)

    return (traingen, testgen)

def pascalvoc(shape, bsize=16, ntest=9, home=os.path.expanduser('~/Developer/data/pascalvoc')):
    impath = os.path.join(home, 'JPEGImages/{}.jpg')
    lbpath = os.path.join(home, 'SegmentationClass/{}.jpg')

    path = os.path.join(home, 'ImageSets/Segmentation/{}.txt')
    with open(os.path.join(path.format('train'))) as f:
        trainset = [line.rstrip('\n') for line in f]
    with open(os.path.join(path.format('trainbval'))) as f:
        validset = [line.rstrip('\n') for line in f]
    with open(os.path.join(path.format('test'))) as f:
        testset = [line.rstrip('\n') for line in f]

    class SemanticSegmentation(Dataset):
        def __init__(self, split):
            super().__init__()
            assert 'train' in split or 'test' in split
            self.split = split

        def __getitem__(self, index):

            a = self.source.format(imfile)
            b = self.target.format(imfile)
            im = self.transform(Image.open(a).convert('RGB'))
            lb = self.transform(Image.open(b).convert('RGB'))
            return im,lb

        def __len__(self):
            return len(self.data)


def load(config):
    xdim = config.xdim if hasattr(config, 'xdim') else config.input_shape

    bsize = config.bsize if hasattr(config, 'bsize') else 1
    ntest = config.ntest if hasattr(config, 'ntest') else bsize

    if 'imagenet' in config.dataset:
        datagen = lambda size=xdim,bsize=bsize:celeba(size, bsize, ntest)
    elif 'celeba' in config.dataset:
        datagen = lambda size=xdim,bsize=bsize:celeba(size, bsize, ntest)
    elif 'cifar10' in config.dataset:
        datagen = lambda size=xdim,bsize=bsize:cifar10(size, bsize, ntest)
    elif 'mnist' in config.dataset:
        datagen = lambda size=xdim,bsize=bsize:mnist(size, bsize, ntest)
    elif 'facades' in config.dataset:
        datagen = lambda size=xdim,bsize=bsize:facades(size, bsize=bsize, ntest=ntest)
    else:
        datagen = lambda size=xdim,bsize=bsize:images(config.dataset,size,bsize=bsize,ntest=ntest)

    return datagen
