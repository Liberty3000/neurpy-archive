import click, collections, os, time

import neurpy
from neurpy.dnn import classifier

def plot(stats):
    fig,(ax1,ax2) = mp.subplots(1,2,figsize=(16,6))
    fig.suptitle('Training')

    for id,curve in zip(classifiers.keys(),np.asarray(stats['train_loss'])):
        ax1.plot(curve, alpha=0.7, label=id)

    for id,curve in zip(classifiers.keys(),np.asarray(stats['top1_train_acry'])):
        ax2.plot(curve, alpha=0.7, label=id)

    ax1.set_title('Objective Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_title('Classification Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Top-1 Accuracy')
    ax1.legend()

    fig,(ax1,ax2) = mp.subplots(1,2,figsize=(16,6))
    fig.suptitle('Testing')

    for id,curve in zip(classifiers.keys(),np.asarray(stats['test_loss'])):
        ax1.plot(curve, alpha=0.7, label=id)

    for id,curve in zip(classifiers.keys(),np.asarray(stats['top1_test_acry'])):
        ax2.plot(curve, alpha=0.7, label=id)

    ax1.set_title('Objective Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_title('Classification Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Top-1 Accuracy')
    ax1.legend()

    fig.savefig('plot.png')

@click.command()
@click.argument('saver', default='experiments/')
@click.argument('device', default='cuda')
@click.argument('architectures', default=['resnet18','darknet19'])

@click.argument('dataset', default='imagenet')
@click.argument('ydim', default=1000)
@click.argument('xdim', default=[3,128,128])
@click.argument('loss', default='nll')
@click.argument('outputfn', default='logsoftmax')

@click.argument('dataset', default='celeba')
@click.argument('ydim', default=40)
@click.argument('loss', default='bce')
@click.argument('outputfn', default='sigmoid')

@click.argument('bsize', default=128)
@click.argument('epochs', default=32)

@click.argument('optimizer', type=dict, default={"amsgrad":{"lr":3e-4,"weight_decay":1e-4}})

@click.argument('ntest', default=1)
@click.argument('topk', default=False)
def run(**args):
    config = neurpy.config.Config(args)

    config.saver = os.path.join(config.saver, '{}_trainer'.format(config.dataset))
    os.makedirs(config.saver, exist_ok=True)

    datagen = neurpy.dataset.load(config)

    ensemble = collections.defaultdict(list)
    models = {k: neurpy.dnn.classifiers[k] for k in config.architectures}

    for id,model in models.items():
        print('training {}...'.format(id))
        beg = time.time()
        config.id = id

        stats = neurpy.dnn.classifier.train(model(config.xdim, config.ydim), datagen, config)

        m, s = divmod(time.time() - beg, 60)
        h, m = divmod(m, 60)
        print('done. {} h {} m {} s\n'.format(h,m,s))

        ensemble['train_loss'].append(stats['train_loss'])
        ensemble['top1_train_acry'].append(stats['top1_train_acry'])
        ensemble['test_loss'].append(stats['test_loss'])
        ensemble['top1_test_acry'].append(stats['top1_test_acry'])

    # analyze and compare...
    plot(ensemble)

if __name__ == '__main__':
    run()
