import torch.nn as nn
from models.vgg import vgg16
from torchsummary import summary
from mxnet.gluon.data import DataLoader
import torch.nn.functional as F

class VGGModel(nn.Module):

    def __init__(self):
        super(VGGModel, self).__init__()
        self.VGG = vgg16(pretrained=True)  
        self.VGG.cuda()

    def forward(self, input_data):

        output, feature = self.VGG(input_data)
        return output, feature


class classfyNet(nn.Module):
    def __init__(self):
        super(classfyNet, self).__init__()
        self.linear = nn.Sequential(  # input shape (32, 3, 4)
            nn.Linear(2048, 512),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(512, 11)
        )
 
    def forward(self, input_data):
        out = self.linear(input_data)
        return out