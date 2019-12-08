import glob, importlib, json, os, pathlib, time
import logging as logx
import numpy as np, torch as th
from neurpy import dataset

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

functions = {
'linear':lambda x:x,
'identity':lambda x:x,
# activation functions
'relu':th.nn.ReLU(True),
'lelu':th.nn.LeakyReLU(2e-1,True),
'elu':th.nn.ELU(True),
'celu':th.nn.CELU(True),
'selu':th.nn.SELU(True),
'log_softmax':lambda x:th.nn.LogSoftmax(dim=1)(x),
'sigmoid':lambda x:th.sigmoid(x),
'softmax':lambda x:th.softmax(x,dim=1),
'tanh'   :lambda x:th.tanh(x),
# loss functions
'bce':th.nn.BCELoss(),
'bce_with_logits':th.nn.BCEWithLogitsLoss(),
'cce':th.nn.CrossEntropyLoss(),
'nll':th.nn.NLLLoss(),
'mse':th.nn.MSELoss(),
'l1' :th.nn.L1Loss(),
'smooth_l1':th.nn.SmoothL1Loss(),
'huber':th.nn.SmoothL1Loss(),
'r1' :lambda pred:th.nn.functional.softplus(-pred).mean(),
'kld':lambda mu,logvar:-0.5 * th.mean(1 + logvar - mu.pow(2) - logvar.exp())
}

def distribution(spec):
    if isinstance(spec, dict):
        distr_id = list(spec.keys())[0]
        distr_params = spec[distr_id]
        mini,maxi = tuple(distr_params.values())
    else: distr_id = spec

    if 'uniform' == distr_id:
        sampler = lambda bsize=1,mini=mini,maxi=maxi:th.from_numpy(np.random.uniform(mini,maxi,size=bsize)).float()
    elif 'normal' == distr_id:
        sampler = lambda zdim,bsize=1:th.from_numpy(np.random.normal(size=(bsize,zdim))).float()
    elif 'categorical' in distr_id:
        sampler = lambda ydim,bsize=1:th.from_numpy(np.eye(ydim)[np.random.randint(0,ydim,bsize)]).float()
    else: raise NotImplementedError()

    return sampler

def optimizer(spec, params=None):
    optim_id = list(spec.keys())[0]
    hparams = spec[optim_id]
    keys = list(hparams.keys())

    if 'lr' in keys: lr = float(hparams['lr'])
    else: lr = 1e-3
    if 'weight_decay' in keys: weight_decay = float(hparams['weight_decay'])
    if 'momentum' in keys: momentum = float(hparams['momentum'])
    if 'betas' in keys: betas = list(hparams['betas'])
    else: betas = (0.9, 0.999)

    if 'sgd' == optim_id:
        optim = lambda params:th.optim.SGD(params, lr=lr, momentum=momentum,
        weight_decay=weight_decay)
    if 'adam' == optim_id:
        optim = lambda params:th.optim.Adam(params, lr=lr, betas=betas,
        weight_decay=weight_decay)
    if 'amsgrad' == optim_id:
        optim = lambda params:th.optim.Adam(params, lr=lr, betas=betas,
        weight_decay=weight_decay, amsgrad=True)

    return optim(params) if params else optim

def log(stats, saver='.', save_as='stats.csv'):
    save_as = os.path.join(saver, save_as)
    mode = 'a' if os.path.exists(save_as) else 'w'
    lfile = open(save_as, mode)
    line = ''
    if mode == 'w':
        for k in stats.keys(): line += '{},'.format(k)
        lfile.write(line[:-1] + '\n')
        line = ''
    for k,v in stats.items(): line += '{},'.format(v.pop())
    lfile.write(line[:-1] + '\n')
    lfile.close()
    return stats

def mount(model, device='cpu'):
    if 'device' in args.keys() and args['device'] == 'cuda':
        logx.info('checking for gpu accelerators.')
        if th.cuda.is_available():
            banner = 'success. found {} cuda-enabled devices.'
            logx.info(banner.format(th.cuda.device_count()))
            device = 'cuda'
        else:
            logx.warning('the gpu was requested but no devices were found.')

    logx.info('mounting {} to device.'.format(model_id))
    model.to(th.device(device))
    logx.info('success. the model was mounted to the device.')

def module(path):
    for root,dirs,fname in os.walk(os.getcwd()):
        if path in fname:
            path = os.path.join(root,path)
            break

    mfile = os.path.join(pathlib.Path(__file__).parents[1], path)
    mspec = importlib.util.spec_from_file_location(mfile, mfile)

    mmodule = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mmodule)
    return mmodule

def initialize(args):
    modules = pathlib.Path(__file__).parents[0]
    package = pathlib.Path(__file__).parents[1]

    # unique identifier
    now = time.strftime('%B.%Y.%d.%M.%S')
    model_id, data_id = args['model'], args['dataset']
    if not 'experiment' in args.keys():
        args.update({'experiment':'{}-{}_{}'.format(model_id, data_id, now)})

    # designate where the experiment is saved
    if not 'saver' in args.keys():
        if not 'weights' in args.keys(): folder = os.path.join(package,'experiments')
        else: folder = os.path.join(package, 'archive')
        saver = os.path.join(folder, args['experiment'])
        os.makedirs(saver, exist_ok=True)
        args['saver'] = saver

    # keep track of the model/device loading
    logf = '{}/{}.log'.format(args['saver'], now)
    logx.basicConfig(filename=logf, level=logx.INFO)
    logx.info('organizing experiment under {}'.format(args['experiment']))
    configf = '{}/config.json'.format(args['saver'])
    if not os.path.isfile(configf):
        with open(configf, 'w') as f: json.dump(args, f, indent=2)
        logx.info('saving run parameters to {}'.format(configf))

    # set the torch interface config
    for k,v in args.items():
        if v in list(functions.keys()): args[k] = functions[v]
        if 'nonlin'  in k: args[k] = functions[v]
        if   'loss'  in k: args[k] = functions[v]
        if  'optim'  in k: args[k] = optimizer(v)
        if 'sampler' in k: args[k] = distribution(v)
    config = Config(args)

    # dynamically load the model module
    mfile = '{}.py'.format(model_id.lower())
    mmodule = module(mfile)

    logx.info('loading model from {}'.format(mfile))
    Model = getattr(mmodule, model_id)
    logx.info('model initialized as {}'.format(mmodule))

    # instantiate the model
    model = Model(config)

    # load pretrained weights
    if 'weights' in args.keys():
        banner = 'loading weights from {} for {}.'
        logx.info(banner.format(args['weights'], args['experiment']))
        model.load(**args['weights'])
        logx.info('successfully loaded pre-trained weights.')

    # set target hardware
    device = 'cpu'
    if 'device' in args.keys() and args['device'] == 'cuda':
        logx.info('checking for gpu accelerators.')
        if th.cuda.is_available():
            banner = 'success. found {} cuda-enabled device(s).'
            logx.info(banner.format(th.cuda.device_count()))
            device = 'cuda'
        else:
            logx.warning('the gpu was requested but no devices were found.')
    logx.info('device set to {}.'.format(device))
    config.device = th.device(device)
    logx.info('mounting {} to device.'.format(model_id))
    model.to(config.device)
    logx.info('success. the model was mounted to the device.')

    return model, dataset.load(config), config
