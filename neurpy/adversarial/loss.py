import torch

def GDPPLoss(fake, real, backward=True):
    def compute_diversity():
        phi = F.normalize(phi, p=2, dim=1)
        ab = torch.mm(phi, phi.t())
        eigvals, eigvecs = torch.symeig(sb, eigenvectors=True)
        return eigvals, eigvecs
    def normalize_minmax(eigvals):
        vmin, vmax = torch.min(eigvals), torch.max(eigvals)
        return (eigvals - vmin) / (vmax - vmin)

def exponential_moving_average(Gs, G, alpha=0.999, global_step=999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def moving_average(Gs, G, alpha=0.999, global_step=999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def clip_weights(model, clip=0.01):
    for p in model.parameters(): p.data.clamp_(-clip,clip)

def gradient_penalty(xreal, dreal, coeff=5):
    greal = torch.autograd.grad(outputs=dreal.sum(), inputs=xreal, create_graph=True)[0]
    gpeny = (greal.view(greal.size(0), -1).norm(2, dim=1) ** 2).mean()
    gpeny = coeff * gpeny
    gpeny.backward()
    return gpeny.item()

def wasserstein_gradient_penalty(D, real, fake, coeff=1, device=torch.device('cpu')):
    alpha = torch.rand(real.size(0), 1, 1, 1).expand_as(real).to(device)
    interp = alpha * real.data + (1 - alpha) * fake.data
    x = torch.autograd.Variable(interp, requires_grad=True).to(device)

    dpred = D(x)

    signal = torch.ones(dpred.size()).to(device)
    grads = torch.autograd.grad(outputs=dpred, inputs=x, grad_outputs=signal,
           retain_graph=True, create_graph=True, only_inputs=True)[0]

    grad = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) **2).mean()
    return gp * coeff
