import math, torch, torch.nn as nn

class WeightScaling(nn.Module):
    def __init__(self, x, gain=2):
        super().__init__()
        self.gain = gain
        self.scale = (self.gain / x.weight[0].numel()) ** 0.5

    def forward(self, x):
        return x * self.scale

    def __repr__(self):
        return '{}(gain={})'.format(self.__class__.__name__, self.gain)

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equalize_lrate(module, name='weight'):
    EqualLR.apply(module, name)
    return module

class EqualizedConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equalize_lrate(conv)

    def forward(self, input):
        return self.conv(input)

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equalize_lrate(linear)

    def forward(self, input):
        return self.linear(input)

class EqualizedBlock(nn.Module):
    def __init__(self, inch, ouch, kernel_size, padding,
        kernel_size2=None, padding2=None, pixel_norm=True, spectral_norm=False):
        super().__init__()

        pad1, pad2 = padding, padding
        if padding2 is not None: pad2 = padding2

        kernel1, kernel2 = kernel_size, kernel_size
        if kernel_size2 is not None: kernel2 = kernel_size2

        self.conv = nn.Sequential(
        EqualizedConv2d(inch, ouch, kernel1, padding=pad1),
        nn.LeakyReLU(0.2, inplace=True),
        EqualizedConv2d(ouch, ouch, kernel2, padding=pad2),
        nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        out = self.conv(input)

        return out

class AdaIN(nn.Module):
    def __init__(self, inch, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(inch)
        self.style = EqualizedLinear(style_dim, inch * 2)

        self.style.linear.bias.data[:inch] = 1
        self.style.linear.bias.data[inch:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta
        return out

class Noise(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise

class Constant(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out

class EqualizedStyleBlock(nn.Module):
    def __init__(self,inch,ouch,kernel_size=3,padding=1,style_dim=512,initial=False):
        super().__init__()

        if initial:
            self.conv1 = Constant(inch)
        else:
            self.conv1 = EqualizedConv2d(inch, ouch, kernel_size, padding=padding)

        self.noise1 = equalize_lrate(Noise(ouch))
        self.adain1 = AdaIN(ouch, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = EqualizedConv2d(ouch, ouch, kernel_size, padding=padding)
        self.noise2 = equalize_lrate(Noise(ouch))
        self.adain2 = AdaIN(ouch, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.adain1(out, style)
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.adain2(out, style)
        out = self.lrelu2(out)

        return out

class SelfAttention(nn.Module):
    def __init__(self, inch, activation, temperature=1):
        super().__init__()
        self.inch = inch
        self.activation = activation

        self.key = nn.Conv2d(inch, inch//8, kernel_size=1)
        self.query = nn.Conv2d(inch, inch//8, kernel_size=1)
        self.value = nn.Conv2d(inch, inch, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N,C,W,H = x.size()
        query = self.query(x).view(B, -1, W*H).permute(0,2,1)
        key = self.key(x).view(B, -1, W*H)

        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value(x).view(B, -1, W*H)

        y = torch.bmm(value, attention.permute(0,2,1)).view(B,C,W,H)
        return self.gamma * y + x, attention

class Blur(nn.Module):
    def __init__(self, kernel=[[1, 2, 1], [2, 4, 2], [1, 2, 1]]):
        super().__init__()
        weight = torch.tensor(kernel, dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        self.register_buffer('weight', weight)

    def forward(self, x):
        return F.conv2d(x, self.weight.repeat(x.shape[1], 1, 1, 1), padding=1, groups=input.shape[1])

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)+ 1e-8)

class MinibatchDiscrimination(nn.Module):
    def __init__(self, inch, ouch, kernel_dims, mean=False):
        super().__init__()
        self.inch, self.ouch = inch, ouch
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = th.nn.Parameter(th.Tensor(inch, ouch, kernel_dims))
        th.nn.init.normal(self.T, 0, 1)

    def forward(self, x):
        matrices = x.mm(self.T.view(self.inhch, -1))
        matrices = matrices.view(-1, self.ouch, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        exp_norm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)  # NxB

        if self.mean: o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        stdv = torch.sqrt(x.var(0, unbiased=False) + 1e-8)
        mean = stdv.mean()
        mean = mean.expand(_.size(0), 1, x.size(-2), x.size(-1))
        _= torch.cat([_, mean], dim=1)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0),-1)

def conv3x3bn(inch, ouch, stride=1, bias=False, **kwargs):
    return nn.Sequential(
    nn.Conv2d(inch,ouch, kernel_size=3, bias=bias, stride=stride, **kwargs),
    nn.BatchNorm2d(ouch))

def conv1x1bn(inch, ouch, bias=False, **kwargs):
    return nn.Sequential(
    nn.Conv2d(inch,ouch, kernel_size=1, bias=bias, **kwargs),
    nn.BatchNorm2d(ouch))

def bnreluconv2x(inch, stride=1, bias=False, **kwargs):
    return nn.Sequential(
    nn.BatchNorm2d(inch),
    nn.Conv2d(inch,inch, kernel_size=3, bias=bias, stride=stride, **kwargs),
    nn.ReLU(True),
    nn.BatchNorm2d(inch),
    nn.Conv2d(inch,inch, kernel_size=3, bias=bias, stride=stride, **kwargs),
    nn.ReLU(True))
