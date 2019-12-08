import torch as th, torch.nn as nn

# Table 1. - https://pjreddie.com/media/files/papers/YOLOv3.pdf
class Block(th.nn.Module):
    def __init__(self, ch=64, cardinality=1, groups=32, bias=False,
    preact=True, nonlin=nn.LeakyReLU(2e-1)):
        super().__init__()
        self.nonlin = nonlin
        self.preact = preact
        self.conv1 = th.nn.Conv2d(ch,ch//2, kernel_size=1, padding=0, bias=bias, groups=cardinality)
        self.norm1 = th.nn.GroupNorm(groups,ch//2)
        self.conv2 = th.nn.Conv2d(ch//2,ch, kernel_size=3, padding=1, bias=bias, groups=cardinality)
        self.norm2 = th.nn.GroupNorm(groups,ch)

    def forward(self, x):
        _= self.nonlin(self.conv1(x))
        return x + self.nonlin(self.conv2(_)) if self.preact \
        else self.nonlin(x + self.conv2(_))

class Transition(th.nn.Module):
    def __init__(self, inch, ouch, stride=2, cardinality=1, groups=32, bias=False, nonlin=nn.LeakyReLU(2e-1)):
        super().__init__()
        self.nonlin = nonlin
        self.conv = th.nn.Conv2d(inch, ouch, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = th.nn.GroupNorm(groups, ouch)
    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))

class Darknet(nn.Module):
    def __init__(self, depths, channels,
                 input_shape=(3,224,224),
                 output_shape=1000,
                 nonlin=nn.LeakyReLU(2e-1)):
        super().__init__()
        self.nonlin = nonlin

        layers = [Transition(input_shape[0],channels[0], stride=1)]
        for itr,(ch,depth) in enumerate(zip(channels[:-1],depths),1):
            layers += [Transition(ch, ch*2)]
            layers += [Block(ch*2)] * depth
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], output_shape)

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, features=False):
        _= self.layers(x)
        if features: return _
        _= self.pool(_)
        return self.classifier(_.squeeze())

def darknet53(xdim=(3,224,224), ydim=1000, nonlin=nn.LeakyReLU(2e-1)):
    return Darknet(depths=[1,2,8,8,4], channels=[2**_ for _ in range(5,11)],
                   input_shape=xdim, output_shape=ydim, nonlin=nonlin)
