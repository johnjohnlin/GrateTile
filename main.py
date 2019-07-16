
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
BATCH_SIZE = 1
parser = argparse.ArgumentParser(description='Integrate tiling')
parser.add_argument('--tiling_mode', action='store_true', help='type foe True for tiling')
parser.add_argument('--calcu_sparsity', action='store_true', help='type foe True for calculate sparsity')
parser.add_argument('--layer', default=0, type=int, help='the layer')
args = parser.parse_args()
def AddressPatten2CacheLineNum(indicators, bit_maps, i):
    clc = CacheLineCalculator(indicators, bit_maps)
    fmap_cache_lines = 0
    bmap_cache_lines = 0
    kernel_stride_padding=[(5,1,2),(3,1,1),(3,1,1),(3,1,1)]   # kernel_size, stride, padding
    hwc_list = [(27,27,64),(13,13,192),(13,13,384),(13,13,256)] # feature size and channel
    hwc = hwc_list[i]
    a, b, w_new, h_new, padding = ParameterGet(kernel_stride_padding,i,hwc[0],hwc[1])
    #print(a, b, w_new,h_new)
    fc = FetchCalculator((b,a), (b,a), hwc[2])
    idc = 0
    while idc+8 <= hwc[2]:
        idy = 0
        while idy+a+b <= h_new:
            idx  = 0
            while idx+a+b <= w_new:
                xyc, bmask = fc.Fetch((idx,idy,idc), (min(idx+a+b*2,w_new), min(idy+a+b*2,h_new), idc+8))
                cache_line, cache_line_bm = clc.Fetch(xyc, bmask)
                fmap_cache_lines += cache_line
                bmap_cache_lines += cache_line_bm
                # print(cache_line_bm)
                idx += a+b
            idy += a+b
        idc += a+b
    
    return fmap_cache_lines, bmap_cache_lines
    
def ParameterGet(kernel_stride_padding,idx,h,w):
    kernel, stride, padding = kernel_stride_padding[idx]
    base_size = (8-1)*stride+kernel
    b = kernel-stride
    a = base_size-2*b
    # if args.tiling_mode == True:
    #     w_new = base_size + 8*math.ceil((w+2*padding-base_size)/8.)
    #     h_new = base_size + 8*math.ceil((h+2*padding-base_size)/8.)
    # else:
    w_new = 8*((w+2*padding)//8+1)
    h_new = 8*((h+2*padding)//8+1)

    return a, b, w_new, h_new, padding
    
def SplitTilingFeature(feature_expand,a,b):
    split_list = []
    c, h, w, _ = feature_expand.shape # torch.Size([8*batch size, 36, 36, 8]) or torch.Size([8*batch size, 32, 32, 8])
    #print(c, h, w)
    idx, idy = 0, 0
    for i in range(c):
        y_temp = []
        flag_h = True  ## +b
        h_size = b
        while idy < h:
            x_temp = []
            flag_w = True  ## +b
            while idx < w:

                if flag_w:
                    x_temp.append(feature_expand[i,idy:idy+h_size,idx:idx+b,:])
                    idx += b
                else:
                    x_temp.append(feature_expand[i,idy:idy+h_size,idx:idx+a,:])
                    idx += a
                flag_w = not flag_w
            y_temp.append(x_temp)

            if flag_h:
                idy += b
                h_size = a
            else:
                idy += a
                h_size = b
            flag_h = not flag_h
            idx = 0
        split_list.append(y_temp)
        idy = 0
    # print(len(split_list),len(split_list[0]),len(split_list[0][0]))
    # print(split_list[0][0][0].shape,split_list[0][0][1].shape,split_list[0][1][0].shape,split_list[0][1][1].shape)

    return split_list  
def SplitNonTilingFeature(feature_expand):
    split_list = []
    c, h, w, _ = feature_expand.shape
    idx, idy = 0, 0
    for i in range(c):
        y_temp = []
        while idy < h:
            x_temp = []
            while idx < w:
                x_temp.append(feature_expand[i,idy:idy+8,idx:idx+8,:])
                idx += 8
            y_temp.append(x_temp)
            idy += 8
            idx = 0
        split_list.append(y_temp)
        idy = 0
    #print(len(split_list),len(split_list[0]),len(split_list[0][0]))
    #print(split_list[0][0][0].shape,split_list[0][1][1].shape,split_list[0][0][1].shape,split_list[0][1][0].shape)
    return split_list 

def Compress(block):
    cache = [] 
    bit_map = [] # 2d list size([9,9]) or ([4,4]) , each item for torch.Size([4,4,8]) or ([8,8,8])
    indicator = [] # 2d list size([9,9]) or ([4,4]) , each item for an int 
    #print('block[0][0]:',block[0][0])
    
    for i in range(len(block)): # h direction block number
        list_bmap_temp = []  # bit map element
        list_indicator_temp = []  # indicator element
        for j in range(len(block[i])): # w direction block number
            list_bmap_temp.append(block[i][j]>0)
            block_temp =  block[i][j].reshape(-1) # reshape to 4*4*8=128 or 8*8*8=512
            block[i][j] = block_temp[block_temp.nonzero()].reshape(-1)  # block_temp.nonzero(): index of nonzero
            ######### align to cache line ##### to do
            block[i][j] = torch.cat((block[i][j] ,torch.zeros((8-block[i][j].shape[0]%8)))) 
            if block[i][j].shape[0]%8 != 0:
                raise ValueError('please align to cache line')
            
            list_indicator_temp.append(block[i][j].nonzero().reshape(-1).shape[0])
            cache.append(block[i][j])
        indicator.append(list_indicator_temp)
        bit_map.append(list_bmap_temp)
    cache = torch.cat(cache) #1d torch torch.Size([81]) or ([16]) each item for a float tensor

    return bit_map, cache, indicator
def SpasityCalculator(bit_map, cache, indicator):
    spasitys = []
    num_bitmap_bits = 0 # bit map bit summation
    num_fmap_bits = 0
    num_indicator_bits = 0
    h, w = len(bit_map), len(bit_map[0])
    # print(h,w)
    for i in range(h):
        for j in range(w):
            spasitys.append(( bit_map[i][j].reshape(-1).shape[0]-indicator[i][j]) / float(bit_map[i][j].reshape(-1).shape[0]))
            num_fmap_bits += indicator[i][j]*16 # one of nonzero feature map is 16 bits
            num_bitmap_bits += bit_map[i][j].reshape(-1).shape[0]
            # print(bit_map[i][j].reshape(-1).shape[0])
            # print(math.ceil(bit_map[i][j].reshape(-1).shape[0]/16/8)*8*16)
            if args.tiling_mode:
                num_indicator_bits += 16/4 # 16+4+4 indicator + Compress or not + if zero of not ::: 4 tile for 24bit
            else:
                num_indicator_bits += 8 #  6+1+1 indicator + Compress or not + if zero of not 
    #num_fmap_bits = sum_id*16
    # print(SpasityCalculator,num_fmap_bits)
    # print(num_bitmap_bits)
    # raise ValueError('Lets look')
    return spasitys, num_fmap_bits, num_indicator_bits, num_bitmap_bits

def Integrate(feature, kernel_stride_padding, idx):
    batch_size, c, h, w = feature.shape
 
    feature = feature.view(-1,8,h,w) # torch.Size([8*batch size, 8, 27, 27])
    feature = feature.permute(0,2,3,1) # torch.Size([8*batch size, 27, 27, 8])
    #print(feature.shape)
    a, b, w_new, h_new, padding = ParameterGet(kernel_stride_padding,idx,h,w)
    feature_expand = torch.zeros((feature.shape[0],h_new,w_new,8))  # torch.Size([8*batch size, 36, 36, 8]) or torch.Size([8*batch size, 32, 32, 8])
    feature_expand[:, padding:padding+h, padding:padding+w, :] = feature
    #print(feature_expand.shape)
    if args.tiling_mode == True:
        split_list = SplitTilingFeature(feature_expand,a,b)
    else:
        split_list = SplitNonTilingFeature(feature_expand)
    #print(len(split_list[0][0]))
    
    if args.calcu_sparsity:
        spasity_list = []
        all_fmap_bits = 0
        all_indicator_bits = 0
        all_bitmap_bits = 0
        for i in range(len(split_list)):  # len(split_list) = batch size * 8
            bit_map, cache, indicator = Compress(split_list[i])
            
            spasitys, num_fmap_bits, num_indicator_bits, num_bitmap_bits = SpasityCalculator(bit_map, cache, indicator)
            
            spasity_list += spasitys
            all_fmap_bits += num_fmap_bits
            all_indicator_bits += num_indicator_bits
            all_bitmap_bits += num_bitmap_bits
            # print(num_bitmap_bits)
        # print(all_bitmap_bits)
        # raise ValueError('Lets look')
        return all_fmap_bits, all_indicator_bits, all_bitmap_bits, spasity_list
    else:
        indicators = []
        bit_maps = []
        for i in range(len(split_list)):  # len(split_list) = batch size * 8
            bit_map, cache, indicator = Compress(split_list[i])
            indicators.append(indicator)
            bit_maps.append(bit_map)
        return bit_maps, cache, indicators

		
def main():
    # kernel_stride_padding = [(5,1,2),(3,1,1),(3,1,1),(3,1,1)]
    # h_list = [27,13,13,13]
    # for idx in range(4):
    #     print(ParameterGet(kernel_stride_padding,idx,h_list[idx],h_list[idx]))
    print('===================================')
    print('Tiling = ', args.tiling_mode)
    print('Sparsity = ', args.calcu_sparsity)
    print('Layer = ', args.layer)
    print('===================================')



    dataset = myDataset(img_dir='/home/mediagti2/Dataset/Imagenet',
                        train=False,
                        transform=transforms.Compose([
                        #transforms.ToPILImage(),
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                           ]))
    dataloader = Data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         num_workers= 30,
                                         )
    net = alexnet(pretrained=True)                                   

    pbar = tqdm(total=len(dataloader),ncols=120)
    sum_fmap_bits = 0.0
    sum_indicator_bits = 0.0
    sum_bitmap_bits = 0.0
    sum_spasity = [] 
    cache_lines_data = 0
    bmap_cache_lines = 0
    write = {}
    img_total = 0
    for step, (img, label) in enumerate(dataloader):
        if cuda:
            net = net.cuda()
        img = img.cuda()
        net.eval()
        output, feature1, feature2, feature3, feature4, feature5, kernel_stride_padding = net(img)  # kernel_stride_padding include kernel size, stride and padding

        if args.layer == 0:
            feature = feature1
        elif args.layer == 1:
            feature = feature2
        elif args.layer == 2:
            feature = feature3
        else:
            feature = feature4

        if args.calcu_sparsity:
            all_fmap_bits, all_indicator_bits, all_bitmap_bits, spasity_list = Integrate(feature, kernel_stride_padding, args.layer)
            sum_fmap_bits += all_fmap_bits
            sum_indicator_bits += all_indicator_bits
            sum_bitmap_bits += all_bitmap_bits    
            sum_spasity += spasity_list           
        else:
            bit_maps, cache, indicators = Integrate(feature, kernel_stride_padding, args.layer)
            fmap_temp, bmap_temp = AddressPatten2CacheLineNum(indicators,bit_maps, args.layer)
            # print(bmap_temp)
            cache_lines_data += fmap_temp
            bmap_cache_lines += bmap_temp
            img_total += 1
            write['Cache_lines_data'] = '{:.2f}'.format(cache_lines_data/img_total)
            write['Cache_line_bm'] = bmap_cache_lines/img_total
            pbar.set_postfix(write)
      
        pbar.update()
    pbar.close()
    if args.calcu_sparsity:
        print('Average data bits :', int(sum_fmap_bits/len(dataset)),'Average idx bits :', int(sum_indicator_bits/len(dataset)),'Average bmap bits :', int(sum_bitmap_bits/len(dataset))  )
        hist = np.histogram([i for i in sum_spasity if i < 1], bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.xticks(np.arange(10),( '<0.1', '<0.2', '<0.3', '<0.4', '<0.5', '<0.6', '<0.7', '<0.8', '<0.9', '<1'))
        plt.bar(np.arange(10), hist[0])  # arguments are passed to np.histogram
        plt.savefig("sparsity_ft%d_%s.jpg"%(args.layer,args.tiling_mode))
    else:
        print('Cache_line_data: ', cache_lines_data/len(dataset), 'Cache_line_bm: ', bmap_cache_lines/len(dataset))
    

if __name__ == '__main__':
    main()