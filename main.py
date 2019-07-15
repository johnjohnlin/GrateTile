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
parser.add_argument('--tiling_mode', action='store_true', help='type foe True for tiling')
parser.add_argument('--calcu_sparsity', action='store_true', help='type foe True for calculate sparsity')
args = parser.parse_args()
def address_patten(indicators, bit_maps, i):
    clc = CacheLineCalculator(indicators, bit_maps)
    cache_lines = 0
    ks_list=[(5,1,2),(3,1,1),(3,1,1),(3,1,1)]   # kernel_size, stride, padding
    hwc_list = [(27,27,64),(13,13,192),(13,13,384),(13,13,256)] # feature size and channel
    hwc = hwc_list[i]
    a, b, w_n, h_n, p = get_para(ks_list,i,hwc[0],hwc[1])
    #print(a, b, w_n,h_n)
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
    
def get_para(ks_list,idx,h,w):
    k, s, p = ks_list[idx]
    f = (8-1)*s+k
    b = k-s
    a = f-2*b
    if args.tiling_mode == True:
        w_n = f + 8*math.ceil((w+2*p-f)/8.)
        h_n = f + 8*math.ceil((h+2*p-f)/8.)
    else:
        w_n = 8*math.ceil((w+2*p)/8.)
        h_n = 8*math.ceil((h+2*p)/8.)

    return a, b, w_n, h_n, p 
    
def split_tiling_ft(ft_expand,a,b):
    split_list = []
    c, h, w, _ = ft_expand.shape # torch.Size([8*batch size, 36, 36, 8]) or torch.Size([8*batch size, 32, 32, 8])
    # print(c, h, w)
    idx, idy = 0, 0
    for i in range(c):
        temp_y = []
        flag_h = True  ## +b
        flag_t = b
        while idy < h:
            temp_x = []
            flag_w = True  ## +b
            while idx < w:

                if flag_w:
                    temp_x.append(ft_expand[i,idy:idy+flag_t,idx:idx+b,:])
                    idx += b
                else:
                    temp_x.append(ft_expand[i,idy:idy+flag_t,idx:idx+a,:])
                    idx += a
                flag_w = not flag_w
            temp_y.append(temp_x)

            if flag_h:
                idy += b
                flag_t = a
            else:
                idy += a
                flag_t = b
            flag_h = not flag_h
            idx = 0
        split_list.append(temp_y)
        idy = 0
    # print(len(split_list),len(split_list[0]),len(split_list[0][0]))
    # print(split_list[0][0][0].shape,split_list[0][0][1].shape,split_list[0][1][0].shape,split_list[0][1][1].shape)
    return split_list  
def split_notiling_ft(ft_expand):
    split_list = []
    c, h, w, _ = ft_expand.shape
    idx, idy = 0, 0
    for i in range(c):
        temp_y = []
        while idy < h:
            temp_x = []
            while idx < w:
                temp_x.append(ft_expand[i,idy:idy+8,idx:idx+8,:])
                idx += 8
            temp_y.append(temp_x)
            idy += 8
            idx = 0
        split_list.append(temp_y)
        idy = 0
    #print(len(split_list),len(split_list[0]),len(split_list[0][0]))
    #print(split_list[0][0][0].shape,split_list[0][1][1].shape,split_list[0][0][1].shape,split_list[0][1][0].shape)
    return split_list 

def compress(block):
    cache = [] 
    bit_map = [] # 2d list size([9,9]) or ([4,4]) , each item for torch.Size([4,4,8]) or ([8,8,8])
    indicator = [] # 2d list size([9,9]) or ([4,4]) , each item for an int 
    #print('block[0][0]:',block[0][0])
    
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
def spasity(bit_map, cache, indicator):
    spasitys = []
    sum_id = 0 # indicator summation
    bit_bm = 0 # bit map bit summation
    bit_data = 0
    bit_idx = 0
    h, w = len(bit_map), len(bit_map[0])
    # print(h,w)
    for i in range(h):
        for j in range(w):
            spasitys.append(( bit_map[i][j].reshape(-1).shape[0]-indicator[i][j]) / float(bit_map[i][j].reshape(-1).shape[0]))
            sum_id += indicator[i][j]
            bit_bm += math.ceil(bit_map[i][j].reshape(-1).shape[0]/16/8)*8*16
            # print(bit_map[i][j].reshape(-1).shape[0])
            # print(math.ceil(bit_map[i][j].reshape(-1).shape[0]/16/8)*8*16)
            bit_idx += 16
    bit_data = sum_id*16
    # print(spasity,bit_data)
    print(bit_bm)
    # raise ValueError('Lets look')
    return spasitys, bit_data, bit_idx, bit_bm

def grate(feature, ks_list, idx):
    batch, c, h, w = feature.shape
    
    feature = feature.view(-1,8,h,w) # torch.Size([8*batch size, 8, 27, 27])
    feature = feature.permute(0,2,3,1) # torch.Size([8*batch size, 27, 27, 8])
    #print(feature.shape)
    a, b, w_n, h_n, p = get_para(ks_list,idx,h,w)
    ft_expand = torch.zeros((feature.shape[0],h_n,w_n,8))  # torch.Size([8*batch size, 36, 36, 8]) or torch.Size([8*batch size, 32, 32, 8])
    ft_expand[:, p:p+h, p:p+w, :] = feature
    #print(ft_expand.shape)
    if args.tiling_mode == True:
        split_list = split_tiling_ft(ft_expand,a,b)
    else:
        split_list = split_notiling_ft(ft_expand)
    #print(len(split_list[0][0]))
    
    if args.calcu_sparsity:
        spasity_list = []
        bits_data = 0
        bits_idx = 0
        bits_bm = 0
        
        for i in range(len(split_list)):  # len(split_list) = batch size * 8
            bit_map, cache, indicator = compress(split_list[i])
            
            spasitys, bit_data, bit_idx, bit_bm = spasity(bit_map, cache, indicator)
            
            spasity_list += spasitys
            bits_data += bit_data
            bits_idx += bit_idx
            bits_bm += bit_bm
            # print(bit_bm)
        # print(bits_bm)
        # raise ValueError('Lets look')
        return bits_data, bits_idx, bits_bm, spasity_list
    else:
        indicators = []
        bit_maps = []
        for i in range(len(split_list)):  # len(split_list) = batch size * 8
            bit_map, cache, indicator = compress(split_list[i])
            indicators.append(indicator)
            bit_maps.append(bit_map)

        return bit_maps, cache, indicators

def main():
    # ks_list = [(5,1,2),(3,1,1),(3,1,1),(3,1,1)]
    # h_list = [27,13,13,13]
    # for idx in range(4):
    #     print(get_para(ks_list,idx,h_list[idx],h_list[idx]))





    dataset = myDataset(img_dir='/home/mediagti2/Dataset/Imagenet',
                        train=False,
                        transform=transforms.Compose([
                        #transforms.ToPILImage(),
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                           ]))
    dataloader = Data.DataLoader(dataset, batch_size=5,
                                         shuffle=False,
                                         num_workers= 30,
                                         )
    net = alexnet(pretrained=True)                                   

    pbar = tqdm(total=len(dataloader),ncols=120)
    bits_data_all = 0.0
    bits_idx_all = 0.0
    bits_bm_all = 0.0
    spasity_all = [] 
    cache_lines_all = 0
    for step, (img, label) in enumerate(dataloader):
        if cuda:
            net = net.cuda()
        img = img.cuda()
        net.eval()
        output, feature1, feature2, feature3, feature4, feature5, ks_list = net(img)  # ks_list include kernel size, stride and padding
        if args.calcu_sparsity:
            bits_data, bits_idx, bits_bm, spasity_list = grate(feature2, ks_list, 1)
            bits_data_all += bits_data
            bits_idx_all += bits_idx
            bits_bm_all += bits_bm    
            spasity_all += spasity_list           
        else:
            bit_maps, cache, indicators = grate(feature4, ks_list, 3)
            cache_lines_all += address_patten(indicators,bit_maps, 3)

        
        pbar.update()
    if args.calcu_sparsity:
        print('Average data bits :', int(bits_data_all/len(dataset)),'Average idx bits :', int(bits_idx_all/len(dataset)),'Average bmap bits :', int(bits_bm_all/len(dataset))  )
    else:
        print('Cache_line: ', cache_lines_all/len(dataset))
    pbar.close()
    # hist = np.histogram([i for i in spasity_all if i < 1], bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # plt.xticks(np.arange(10),( '<0.1', '<0.2', '<0.3', '<0.4', '<0.5', '<0.6', '<0.7', '<0.8', '<0.9', '<1'))
    # plt.bar(np.arange(10), hist[0])  # arguments are passed to np.histogram
    # plt.savefig("sparsity_ft3_%s.jpg"%args.tiling_mode)

if __name__ == '__main__':
    main()
