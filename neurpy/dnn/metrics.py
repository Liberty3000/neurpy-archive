import math
import matplotlib.pyplot as mp
import sklearn
import torch

def psnr(ytrue, ypred):
    mse = torch.nn.functional.mse_loss(ypred, ytrue)
    psnr = 10 * math.log10(1 / mse.item())
    return psnr

def accuracy(ypred, ytrue, topk=[1,3,5]):
    with torch.no_grad():
        _,pred = ypred.topk(max(topk), 1, True, True)
        pred = pred.t()
        correct = pred.eq(ytrue.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correctk = correct[:k].view(-1).float().sum(0, keepdim=True)
            result.append(correctk.mul_(100.0 / ytrue.size(0)))

        return result

def confusion_matrix(ytrue, ypred, classes, title='Confusion Matrix',
    cmap=mp.cm.plasma, notebook=False):

    mat = sklearn.metrics.confusion_matrix(ytrue,ypred)

    fig, ax = mp.subplots()
    im = ax.imshow(mat, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    fig.tight_layout()

    ax.set(xticks=np.arange(mat.shape[1]),
           yticks=np.arange(mat.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label', xlabel='Pred. Label')

    thresh = mat.max() / 2
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j,i, format(mat[i,j], 'd'), ha='center', va='center',
                    color='white' if mat[i,j] < thresh else 'black')

    return mat
