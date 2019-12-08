import click, glob, json, os, time, tqdm
from collections import defaultdict
import torch as th

import neurpy
from neurpy import util
from neurpy.config import initialize, log
from neurpy.callback import Callback

class Classifier(th.nn.Module):
    def __init__(self, model, optim, loss, outputfn, topk=[1,3,5]):
        super().__init__()
        self.model = model
        self.optim = optim
        self.loss  = loss
        self.outputfn = outputfn
        self.stats = defaultdict(list)
        self.topk = topk

    def forward(self, x):
        logit = self.model(x)
        ypred = self.outputfn(logit)
        return ypred

    def forwardbackward(self, x, ytrue, metrics=True):
        ypred = self.forward(x)
        if ytrue.size() != ypred.size(): ytrue = th.max(ytrue,dim=1)[1].long()
        loss = self.loss(ypred, ytrue)
        self.backward(loss)
        self.stats['train_loss'].append(loss.item())
        if self.topk:
            top1,top3,top5 = neurpy.dnn.metrics.accuracy(ypred,ytrue,topk=self.topk)
            self.stats['top1_train_acry'].append(top1.item())
            self.stats['top3_train_acry'].append(top3.item())
            self.stats['top5_train_acry'].append(top5.item())

    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def validate(self, x, ytrue, metrics=True):
        ypred = self.forward(x)
        if ytrue.size() != ypred.size(): ytrue = th.max(ytrue,dim=1)[1].long()
        loss = self.loss(ypred, ytrue)

        self.stats['test_loss'].append(loss.item())
        if self.topk:
            top1,top3,top5 = neurpy.dnn.metrics.accuracy(ypred,ytrue,topk=self.topk)
            self.stats['top1_test_acry'].append(top1.item())
            self.stats['top3_test_acry'].append(top3.item())
            self.stats['top5_test_acry'].append(top5.item())

    def train(self, datagen, config, interval=1000):
        traingen, testgen = datagen(config.xdim)

        for epoch in range(1,1 + config.epochs):
            banner = '\tEpoch {:3}/{:3}| Batch Size {}'

            print(banner.format(epoch, config.epochs, config.bsize))
            bar = tqdm.tqdm(enumerate(traingen), desc=banner, total=len(traingen))

            for batch,(xinpu,ytrue) in bar:
                xinpu = xinpu.to(th.device(config.device))
                ytrue = ytrue.to(th.device(config.device))

                self.forwardbackward(xinpu, ytrue)

                with th.no_grad():
                    xtest,ytest = next(iter(testgen))
                    xtest = xtest.to(th.device(config.device))
                    ytest = ytest.to(th.device(config.device))
                    clf.validate(xtest, ytest)

                bar.set_description('Loss {:.3f}'.format(clf.stats['train_loss'][-1]))
                bar.refresh()
                save_as = os.path.join(config.saver, '{}-{}'.format(config.id, now))

                clf.states = neurpy.config.log(clf.stats, save_as=save_as + '.csv')

                itr = batch + (len(traingen) * (epoch - 1))
                if itr == 100 or (itr + 1) % interval == 0:
                    th.save(self.model.state_dict(), save_as + '.th')

def plot(stats, show=False):
    fig,((ax0,ax1),(ax2,ax3)) = mp.subplots(2, 2, figsize=(17,17))

    losses = np.array(stats['train_loss'])
    ax0.scatter(np.argmin(losses), np.min(losses))
    ax0.plot(losses, label='Train Loss', color='#4286f4')

    train_acries = np.asarray([stats['top1_train_acry'],stats['top3_train_acry'],stats['top5_train_acry']]).T
    ax1.plot(train_acries[:,0], color='#41a3f4', label='Top 1 Train Acc.', alpha=0.5)
    ax1.plot(train_acries[:,1], color='#0dd894', label='Top 3 Train Acc.', alpha=0.5)
    ax1.plot(train_acries[:,2], color='#0ec2ef', label='Top 5 Train Acc.', alpha=0.5)
    ax1.scatter(np.argmax(train_acries[:,0]), np.max(train_acries[:,0]))

    ax1.plot([50]*len(train_acries),linestyle='--',color='#ff0000')
    ax1.plot([60]*len(train_acries),linestyle='--',color='#ce1212')
    ax1.plot([70]*len(train_acries),linestyle='--',color='#963939')
    ax1.plot([80]*len(train_acries),linestyle='--',color='#cc8a8a')

    ax0.set_title('Loss')
    ax0.set_ylabel('Loss'), ax0.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    ax1.set_ylabel('Top-K Accuracy'), ax1.set_xlabel('Epoch')
    ax1.set_ylim(0,100), ax1.set_ylabel('Accuracy')

    losses = np.array(stats['test_loss'])
    ax2.scatter(np.argmin(losses), np.min(losses))
    ax2.plot(losses, label='Test Loss', color='#ed7710')

    test_acries = np.asarray([stats['top1_test_acry'],stats['top3_test_acry'],stats['top5_test_acry']]).T
    ax3.plot(test_acries[:,0], color='#a57aa5', label='Top 1 Test Acc.', alpha=0.5)
    ax3.plot(test_acries[:,1], color='#643c84', label='Top 3 Test Acc.', alpha=0.5)
    ax3.plot(test_acries[:,2], color='#a81c4b', label='Top 5 Test Acc.', alpha=0.5)
    ax3.scatter(np.argmax(test_acries[:,0]), np.max(test_acries[:,0]))

    ax3.plot([50]*len(train_acries),linestyle='--',color='#ff0000')
    ax3.plot([60]*len(train_acries),linestyle='--',color='#ce1212')
    ax3.plot([70]*len(train_acries),linestyle='--',color='#963939')
    ax3.plot([80]*len(train_acries),linestyle='--',color='#cc8a8a')

    ax2.set_title('Loss')
    ax2.set_ylabel('Loss'), ax2.set_xlabel('Epoch')
    ax3.set_title('Accuracy')
    ax3.set_ylabel('Top-K Accuracy'), ax3.set_xlabel('Epoch')
    ax3.set_ylim(0,100), ax3.set_ylabel('Accuracy')

    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    if show: mp.show()

def train(model, datagen, config, interval=1000):
    now = time.strftime('%B.%Y.%d.%M.%S')

    traingen, testgen = datagen(config.xdim)

    model.to(th.device(config.device))

    loss = neurpy.config.functions[config.loss]
    optim = neurpy.config.optimizer(config.optimizer, params=model.parameters())
    outputfn = neurpy.config.functions[config.outputfn]

    clf = Classifier(model, optim, loss, outputfn, topk=config.topk)

    for epoch in range(1,1 + config.epochs):
        banner = '\tEpoch {:3}/{:3}| Batch Size {}'

        print(banner.format(epoch, config.epochs, config.bsize))
        bar = tqdm.tqdm(enumerate(traingen), desc=banner, total=len(traingen))

        for batch,(xinpu,ytrue) in bar:
            xinpu = xinpu.to(th.device(config.device))
            ytrue = ytrue.to(th.device(config.device))

            clf.forwardbackward(xinpu, ytrue)

            with th.no_grad():
                xtest,ytest = next(iter(testgen))
                xtest = xtest.to(th.device(config.device))
                ytest = ytest.to(th.device(config.device))
                clf.validate(xtest, ytest)

            bar.set_description('Loss {:.3f}'.format(clf.stats['train_loss'][-1]))
            bar.refresh()
            save_as = os.path.join(config.saver, '{}-{}'.format(config.id, now))

            clf.states = neurpy.config.log(clf.stats, save_as=save_as + '.csv')

            itr = batch + (len(traingen) * (epoch - 1))
            if itr == 100 or (itr + 1) % interval == 0:
                th.save(model.state_dict(), save_as + '.th')
