import torch, torch.nn as nn
from neurpy.dnn.layers import conv1x1bn, conv3x3bn

class Block(nn.Module):
    def __init__(self, inch, ouch, stride=1, reduce=None, nonlin=nn.ReLU(True)):
        super().__init__()
        self.inch = inch
        self.ouch = ouch
        self.stride = stride
        self.reduce = reduce
        self.nonlin = nonlin
        self.conv1bn = conv3x3bn(inch,ouch, stride=stride)
        self.conv2bn = conv3x3bn(ouch,ouch)

    def forward(self, x):
        i = self.reduce(x) if self.reduce else x
        _= nonlin(self.conv1bn(x))
        _= nonlin(i + self.conv2bn(_))
        return _

class Bottleneck(nn.Module):
    def __init__(self, inch, ouch, stride=1, reduce=None, expand=4, nonlin=nn.ReLU(True)):
        self.inch = inch
        self.stride = stride
        self.reduce = reduce
        self.nonlin = nonlin
        self.expand = expand
        self.conv1bn = conv3x3bn(inch,ouch, stride=stride)
        self.conv2bn = conv1x1bn(ouch,ouch)
        self.conv3bn = conv3x3bn(ouch,ouch * expand)
        self.ouch = ouch * expand

    def forward(self, x):
        i = self.reduce(x) if self.reduce else x
        _= nonlin(self.conv1bn(x))
        _= nonlin(self.conv2bn(_))
        _= nonlin(i + self.conv3bn(_))
        return _

def Stem_v1(inch):
    return nn.Sequential(
    nn.Conv2d(3, inch, 7, stride=2, padding=3, bias=False),
    nn.BatchNorm2d(inch),
    nn.MaxPool2d(3, stride=2, padding=1))

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 depths,
                 xshape=(3,224,224),
                 inch=2**6,
                 nch=None,
                 ydim=1000,
                 nonlin=nn.ReLU(inplace=True),
                 fc=False):
        super().__init__()
        self.block = block
        self.inch = inch
        self.nonlin = nonlin
        self.ydim = int(ydim)

        layers = []
        nch = self.inch if not nch else nch
        self.stem = Stem_v1(self.inch)

        blocks = zip(range(nch,len(depths)),depths)
        for itr,(inch,depth) in enumerate(blocks,1):
            stride = 2 if itr > 1 else itr
            layers += [self.residual(inch, depth, stride=stride)]
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        feats = self.output_features(xshape)

        if self.ydim: self.clf = nn.Conv2d(feats, ydim, 1) if fc else nn.Linear(feats, ydim)
        else: self.clf = lambda x:x

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def residual(self, chans, blocks, stride=1):
        layers,reduce = [],None

        if stride != 1 or self.inch != 2**chans * self.expand:
            reduce = conv1x1bn(self.inch, chans * self.expand, stride=stride)

        layers += [self.block(self.inch, ouch)]
        self.inch = chans * self.expansion

        layers += [self.block(self.inch, chans) for _ in range(1,blocks)]
        return nn.Sequential(*layers)

    def output_features(self, shape):
        _= torch.rand(1, *shape)
        _= self.nonlin(self.stem(_))
        _= self.layers(_)
        _= self.avgpool(_)
        return _.data.view(1,-1).size(1)

    def forward(self, x, features=False):
        _= self.nonlin(self.stem(x))
        _= self.layers(_)
        _= self.avgpool(_)
        if features: return _
        _= _.view(_.size(0),-1)
        return self.clf(_)

def resnet18(xshape=(3,224,224), ydim=1000, nonlin=nn.ReLU(True)):
    return ResNet(Block, [2, 2, 2, 2],
    xshape=xshape, ydim=ydim, nonlin=nonlin)

def resnet34(xshape=(3,224,224), ydim=1000, nonlin=nn.ReLU(True)):
    return ResNet(Block, [3, 4, 6, 3],
    xshape=xshape, ydim=ydim, nonlin=nonlin)

def resnet50(xshape=(3,224,224), ydim=1000, nonlin=nn.ReLU(True)):
    return ResNet(Bottleneck, [3, 4, 6, 3],
    xshape=xshape, ydim=ydim, nonlin=nonlin)

def resnet101(xshape=(3,224,224), ydim=1000, nonlin=nn.ReLU(True)):
    return ResNet(Bottleneck, [3, 4, 23, 3],
    xshape=xshape, ydim=ydim, nonlin=nonlin)

def resnet152(xshape=(3,224,224), ydim=1000, nonlin=nn.ReLU(True)):
    return ResNet(Bottleneck, [3, 8, 36, 3],
    xshape=xshape, ydim=ydim, nonlin=nonlin)

def resnet1001(xshape=(3,224,224), ydim=1000, nonlin=nn.ReLU(True)):
    return ResNet(Bottleneck, [111, 111, 111], inch=16, nch=64,
    xshape=xshape, ydim=ydim, nonlin=nonlin)
