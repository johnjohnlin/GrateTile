#!/usr/bin/env python3
from network.alexnet import alexnet
import numpy as np
import torchvision
from torchvision import transforms
import torch
import cv2
import os
from dataloader import myDataset
import torch.utils.data as Data
from tqdm import tqdm
import math
import argparse
import matplotlib.pyplot as plt
from model.GrateTile import *
## hyper parameter
cuda = True
BATCH_SIZE = 5
parser = argparse.ArgumentParser(description='grate tiling')
parser.add_argument('--grate_tile', action='store_true', help='Grate tiling / Naive tiling (T/F)')
parser.add_argument('--simulate_cache', action='store_true', help='True to perform cache simulation')
args = parser.parse_args()

def AddressPattern(indicators, bit_maps, i):
    clc = CacheLineCalculator(indicators, bit_maps)
    cache_lines = 0
    ks_list = [(5,1,2), (3,1,1), (3,1,1), (3,1,1)] # kernel_size, stride, padding
    hwc_list = [(27,27,64), (13,13,192), (13,13,384), (13,13,256)] # feature size and channel
    hwc = hwc_list[i]
    a, b, w_n, h_n, p = get_para(ks_list[i], hwc[0], hwc[1])
    fc = FetchCalculator((b,a), (b,a), hwc[2])
    idc = 0
    while idc+8 <= hwc[2]:
        idy = 0
        while idy+a+b*2 <= w_n:
            idx  = 0
            while idx+a+b*2 <= h_n:
                xyc, bmask = fc.Fetch((idx,idy,idc), (idx+a+b*2, idy+a+b*2, idc+8))
                cache_line = clc.Fetch(xyc, bmask)
                cache_lines += cache_line
                idx += a+b
            idy += a+b
        idc += a+b
    return cache_lines

def ExtractTileParameter(k, s, p, h, w):
    f = (8-1)*s+k
    b = k-s
    a = f-2*b
    if args.grate_tile:
        w_pad = f + 8 * ((w+2*p-f-1)//8 + 1)
        h_pad = f + 8 * ((h+2*p-f-1)//8 + 1)
    else:
        w_pad = 8 * ((w+2*p-1)//8 + 1)
        h_pad = 8 * ((h+2*p-1)//8 + 1)

    return a, b, w_n, h_n, p

def SplitFeature(feature, xysplit, csplit):
    c, h, w, _ = feature.shape # torch.Size([8*batch size, 36, 36, 8]) or torch.Size([8*batch size, 32, 32, 8])
    nblock_x = w // sum(xysplit)
    nblock_y = h // sum(xysplit)
    nblock_c = c // sum(csplit)
    split_idx_x = np.cumsum(np.repeat(xysplit, nblock_x))[:-1]
    split_idx_y = np.cumsum(np.repeat(xysplit, nblock_y))[:-1]
    split_idx_c = np.cumsum(np.repeat(  cplit, nblock_c))[:-1]
    split_list = []
    for channel_group in np.split(feature, split_idx_c, axis=0):
        column_group = np.split(feat, split_idx_y, axis=1) for feat in channel_group
        # split and flatten
        splits = [item for feat in split_list for item in np.split(feat, split_idx_x, axis=2)]
        split_list.append(splits)
    # list of list of array
    return split_list

def Compress(block):
    cache = []
    bit_map = [] # 2d list size([9,9]) or ([4,4]) , each item for torch.Size([4,4,8]) or ([8,8,8])
    indicator = [] # 2d list size([9,9]) or ([4,4]) , each item for an int

    for i in range(len(block)): # h direction block number
        temp_list_b = []  # bit map element
        temp_list_i = []  # indicator element
        for j in range(len(block[i])): # w direction block number
            temp_list_b.append(block[i][j]>0)
            block_t =  block[i][j].reshape(-1) # reshape to 4*4*8=128 or 8*8*8=512
            block[i][j] = block_t[block_t.nonzero()].reshape(-1)  # block_t.nonzero(): index of nonzero
            ######### align to cache line ##### to do
            block[i][j] = torch.cat((block[i][j] ,torch.zeros((8-block[i][j].shape[0]%8))))
            if block[i][j].shape[0]%8 != 0:
                raise ValueError('please align to cache line')

            temp_list_i.append(block[i][j].nonzero().reshape(-1).shape[0])
            cache.append(block[i][j])
        indicator.append(temp_list_i)
        bit_map.append(temp_list_b)
    cache = torch.cat(cache) #1d torch torch.Size([81]) or ([16]) each item for a float tensor

    return bit_map, cache, indicator

def Spasity(bit_map, cache, indicator):
    spasitys = []
    sum_id = 0 # indicator summation
    bit_bm = 0 # bit map bit summation
    bit_data = 0
    bit_idx = 0
    h, w = len(bit_map), len(bit_map[0])
    for i in range(h):
        for j in range(w):
            spasitys.append(( bit_map[i][j].reshape(-1).shape[0]-indicator[i][j]) / float(bit_map[i][j].reshape(-1).shape[0]))
            sum_id += indicator[i][j]
            bit_bm += math.ceil(bit_map[i][j].reshape(-1).shape[0]/16/8)*8*16
            bit_idx += 16
    bit_data = sum_id*16
    return spasitys, bit_data, bit_idx, bit_bm

def Grate(feature, layer_ksp):
    # TODO: currently we only handle layer 2 (batch, 64, 27, 27)
    batch, c, h, w = feature.shape
    k, s, p = layer_ksp
    kCSPLIT = 8

    # convert NCHW format to (NC)HWC' format
    # shape=(batch, 64, 27, 27) --> (8*batch, 8, 27, 27) --> (8*batch, 27, 27, 8)
    feature_reshape = feature.view(-1,kCSPLIT,h,w).permute(0,2,3,1)
    a, b, w_pad, h_pad = ExtractTileParameter(k, s, p, h, w)
    # padding; after that, shape=(8*batch, 36, 36, 8) or (8*batch, 32, 32, 8)
    feature_pad = torch.zeros((feature_reshape.shape[0], h_pad, w_pad, kCSPLIT))
    feature_pad[:, p:p+h, p:p+w, :] = feature
    # split the feature map
    xysplit = (a, b) if args.tiling_mode else (8,)
    split_list = SplitFeature(feature_pad, xysplit, kCSPLIT)

    if not args.simulate_cache:
        spasrity_list = []
        bits_data = 0
        bits_idx = 0
        bits_bm = 0

        for block in split_list:
            bit_map, cache, indicator = Compress(block)

            spasritys, bit_data, bit_idx, bit_bm = Spasrity(bit_map, cache, indicator)

            spasrity_list += spasritys
            bits_data += bit_data
            bits_idx += bit_idx
            bits_bm += bit_bm
        return bits_data, bits_idx, bits_bm, spasrity_list
    else:
        indicators = []
        bit_maps = []
        for i in range(len(split_list)): # len(split_list) = batch size * 8
            bit_map, cache, indicator = Compress(split_list[i])
            indicators.append(indicator)
            bit_maps.append(bit_map)

        return bit_maps, cache, indicators

def main():
    dataset = myDataset(img_dir='/home/mediagti2/Dataset/Imagenet',
                        train=False,
                        transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),]))
    dataloader = Data.DataLoader(
            dataset, batch_size=5,
            shuffle=False,
            num_workers=30,)
    net = alexnet(pretrained=True)

    pbar = tqdm(total=len(dataloader), ncols=120)
    bits_data_all = 0
    bits_idx_all = 0
    bits_bm_all = 0
    spasrity_all = []
    cache_lines_all = 0
    for step, (img, label) in enumerate(dataloader):
        if cuda:
            net = net.cuda()
        img = img.cuda()
        net.eval()
        # ksp: kernel stride padding
        output, feature1, feature2, feature3, feature4, feature5, ksp_list = net(img)
        if not args.simulate_cache:
            bits_data, bits_idx, bits_bm, spasrity_list = Grate(feature2, ksp_list[1])
            bits_data_all += bits_data
            bits_idx_all += bits_idx
            bits_bm_all += bits_bm
            spasrity_all += spasrity_list
        else:
            bit_maps, cache, indicators = Grate(feature4, ksp_list[3])
            cache_lines_all += AddressPattern(indicators, bit_maps, 3)
        pbar.update()

    if not args.simulate_cache:
        avg_data_bit = bits_data_all // len(dataset)
        avg_idx_bit = bits_data_all // len(dataset)
        avg_bits_bm = bits_bm_all // len(dataset)
        print('Average data bits : {}\nAverage idx bits : {}\nAverage bmap bits : {}\n'.format(
                avg_data_bit, avg_idx_bit, avg_bits_bm))
    else:
        print('Cache line: {}'.format(cache_lines_all // len(dataset)))

    pbar.close()

if __name__ == '__main__':
    main()
