from network.alexnet import alexnet
from network.vgg import *
from network.resnet import *
from torchvision import transforms
import pickle

BATCH_SIZE = 1
res_size = pickle.load(open('config/res_size.pkl', 'rb'))

def blk(type,first,i):
    if type=='BasicBlock':
        if first and i != 0:
            return [(3,2,1),(3,1,1)]
        else:
            return [(3,1,1),(3,1,1)]
    else:
        if first and i != 0:
            return [(1,1,0),(3,2,1),(1,1,0)]
        else:
            return [(1,1,0),(3,1,1),(1,1,0)]

def resnet_ksp(block, layers):
    ksp = list()
    for i,num in enumerate(layers):
        ksp += blk(block,True,i)
        bk = blk(block,False,i)
        ksp += bk*(num-1)
    return ksp

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

resnet18_config = dict(
    kernel_stride_padding = resnet_ksp('BasicBlock', [2, 2, 2, 2]),
    hwc_list = res_size['18'],
    net = resnet18(pretrained=True)
)

resnet34_config = dict(
    kernel_stride_padding = resnet_ksp('BasicBlock', [3, 4, 6, 3]),
    hwc_list = res_size['34'],
    net = resnet34(pretrained=True)
)

resnet50_config = dict(
    kernel_stride_padding = resnet_ksp('Bottleneck', [3, 4, 6, 3]),
    hwc_list = res_size['50'],
    net = resnet50(pretrained=True)
)

resnet101_config = dict(
    kernel_stride_padding = resnet_ksp('Bottleneck', [3, 4, 23, 3]),
    hwc_list = res_size['101'],
    net = resnet101(pretrained=True)
)

resnet152_config = dict(
    kernel_stride_padding = resnet_ksp('Bottleneck', [3, 8, 36, 3]),
    hwc_list = res_size['152'],
    net = resnet152(pretrained=True)
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
    elif net == 'resnet18':
        return resnet18_config
    elif net == 'resnet34':
        return resnet34_config
    elif net == 'resnet50':
        return resnet50_config
    elif net == 'resnet101':
        return resnet101_config
    elif net == 'resnet152':
        return resnet152_config
