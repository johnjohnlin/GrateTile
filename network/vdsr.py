import torch
import torch.nn as nn
from math import sqrt
from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

global feature_maps
feature_maps = list()

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        feature_maps.append(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        global feature_maps
        residual = x
        out = self.relu(self.input(x))
        feature_maps.append(out)
        out = self.residual_layer(out)
        fmap = feature_maps.copy()
        feature_maps = list()
        # out = self.output(out)
        # out = torch.add(out,residual)
        return fmap


def vdsr(path, pretrained=False):
    model = Net()
    if pretrained:
        model.load_state_dict(torch.load(path))

    return model


if __name__ == '__main__':
    net = vdsr(path='./vdsr.pth', pretrained=True)
