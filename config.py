from network.alexnet import alexnet
from network.vgg import *
from torchvision import transforms

BATCH_SIZE = 1

alexnet_config = dict(
    kernel_stride_padding = [(5,1,2), (3,1,1), (3,1,1), (3,1,1)],
    hwc_list = [(27,27,64), (13,13,192), (13,13,384), (13,13,256)],
    net = alexnet(pretrained=True)
)

vgg11_config = dict(
    kernel_stride_padding = [(3,1,1)]*7,
    hwc_list = [(112, 112, 64),
                (56, 56, 128),
                (56, 56, 256),
                (28, 28, 256),
                (28, 28, 512),
                (14, 14, 512),
                (14, 14, 512)],
    net = vgg11(pretrained=True)
)

vgg13_config = dict(
    kernel_stride_padding = [(3,1,1)]*9,
    hwc_list = [(224, 224, 64),
                (112, 112, 64),
                (112, 112, 128),
                (56, 56, 128),
                (56, 56, 256),
                (28, 28, 256),
                (28, 28, 512),
                (14, 14, 512),
                (14, 14, 512)],
    net = vgg13(pretrained=True)
)

vgg16_config = dict(
    kernel_stride_padding = [(3,1,1)]*12,
    hwc_list = [(224, 224, 64),
                (112, 112, 64),
                (112, 112, 128),
                (56, 56, 128),
                (56, 56, 256),
                (56, 56, 256),
                (28, 28, 256),
                (28, 28, 512),
                (28, 28, 512),
                (14, 14, 512),
                (14, 14, 512),
                (14, 14, 512)],
    net = vgg16(pretrained=True)
)

vgg19_config = dict(
    kernel_stride_padding = [(3,1,1)]*15,
    hwc_list = [(224, 224, 64),
                (112, 112, 64),
                (112, 112, 128),
                (56, 56, 128),
                (56, 56, 256),
                (56, 56, 256),
                (56, 56, 256),
                (28, 28, 256),
                (28, 28, 512),
                (28, 28, 512),
                (28, 28, 512),
                (14, 14, 512),
                (14, 14, 512),
                (14, 14, 512),
                (14, 14, 512)],
    net = vgg19(pretrained=True)
)

def NetConfig(net='alexnet'):
    if net == 'alexnet':
        return alexnet_config
    elif net == 'vgg11':
        return vgg11_config
    elif net == 'vgg13':
        return vgg13_config
    elif net == 'vgg16':
        return vgg16_config
    elif net == 'vgg19':
        return vgg19_config