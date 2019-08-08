import numpy as np
from torchvision import transforms
import torch
from dataloader import myDataset
import torch.utils.data as Data
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from model.GrateTile import *
import config
import math
import os

################################## Warning ###########################################
# kernel size (5*5) with stride 2 is NOT support
######################################################################################

################################## Note ##############################################
# raw block size = 1*1*8
# uniform block size = 8*8*8
######################################################################################

parser = argparse.ArgumentParser(description='Integrate tiling')
parser.add_argument('--mode', default='uniform', type=str, help='Which mode')
parser.add_argument('--output_size', nargs='+', type=int, default=[8,8,8], help='output size')
parser.add_argument('--dense',    action='store_true')
parser.add_argument('--simulate_memory',   action='store_true', help='True to perform memory simulation')
parser.add_argument('--layer', default=0, type=int, help='the layer')
parser.add_argument('--model', default='alexnet', help='pre-train model')

args = parser.parse_args()

## hyper parameter
BATCH_SIZE = config.BATCH_SIZE
Config = config.NetConfig(args.model)
kernel_stride_padding = Config['kernel_stride_padding']   # kernel_size, stride, padding
output_size = tuple(args.output_size[:-1]) #(w, h)
csplit = (args.output_size[-1],) if args.mode == 'non_uniform' else (8,)
args.output_size[-1] = csplit[0]
uniform_blk_size = 8

transform_list = [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]

if args.model == 'vdsr':
    data_type = 'YCbCr'
    del transform_list[-1]
else :
    data_type = 'RGB'


def AddressPatten2CacheLineNum(indicators, hwc, ksp):
    cache_lines = 0
    _, c, h, w = hwc

    wsplit, w_pad, fetch_size_w = ExtractTileParameter(ksp, w, output_size[0])
    hsplit, h_pad, fetch_size_h = ExtractTileParameter(ksp, h, output_size[1])

    fc = FetchCalculator(wsplit, hsplit, csplit)
    clc = CacheLineCalculator(indicators, wsplit, hsplit, csplit)

    fetch_stride_w = output_size[0] * ksp[1]
    fetch_stride_h = output_size[1] * ksp[1]
    address_pattern_grid = np.mgrid[0:c:csplit[0], 0:h_pad:fetch_stride_h, 0:w_pad:fetch_stride_w] # csplit CANNOT split to (a,b)
    address_pattern = np.column_stack([b.flat for b in address_pattern_grid])[:,::-1].astype('i4') # w, h, c
    for address in address_pattern:
        head = (address[0], address[1], address[2])
        tile_size = (min(head[0]+fetch_size_w,w_pad)-head[0], min(head[1]+fetch_size_h,h_pad)-head[1], csplit[0])

        block_id, block_mask = fc.Fetch(head, tile_size)
        cache_line = clc.Fetch(block_id, block_mask)
        cache_lines += cache_line

    return cache_lines

def Indicator2CacheLine(indicator):
    c, h, w = len(indicator), len(indicator[0]), len(indicator[0][0])
    indicator_temp = 0
    for i in range(c):
        for j in range(h):
            for k in range(w):

                indicator_temp += indicator[i][j][k]
                indicator[i][j][k] = (indicator_temp-1)//8+1
                indicator_temp = indicator_temp%8

    return indicator

def ExtractTileParameter(ksp, fmap_size, output_size):
    kernel, stride, padding = ksp
    fetch_size = (output_size-1)*stride+kernel
    b = kernel-stride
    a = fetch_size-2*b
    a = a%output_size if stride == 2 else a

    if args.mode == 'raw':
        split = (1,)
    elif args.mode == 'non_uniform': # if b=0, split should be (a,)
        split = (b, a) if b > 0 else (a,)
    elif args.mode == 'uniform':
        split = (uniform_blk_size,)

    unit_blk_size = sum(split) # block size for each case
    fmap_size_pad = unit_blk_size*((fmap_size+padding-1)//unit_blk_size+1)

    # fetch_size = fetch_size if b>0 else output_size # for ksp=1,2,0

    return split, fmap_size_pad, fetch_size

def SplitFeature(feature, wsplit, hsplit, csplit):

    c, h, w  = feature.shape # torch.Size([batch*channel, 32, 32]) assume channel%8=0
    nblock_w = w // sum(wsplit)
    nblock_h = h // sum(hsplit)
    nblock_c = c // sum(csplit)

    split_idx_w = np.cumsum(np.tile(wsplit, nblock_w))[:-1]
    split_idx_h = np.cumsum(np.tile(hsplit, nblock_h))[:-1]
    split_idx_c = np.cumsum(np.tile(csplit, nblock_c))[:-1]

    split_list = []
    for channel_group in np.split(feature.cpu().detach(), split_idx_c, axis=0):
        column_group = np.split(channel_group, split_idx_h, axis=1)
        # split and flatten
        splits = [np.split(feat, split_idx_w, axis=2) for feat in column_group]
        split_list.append(splits)
    #print(len(split_list),len(split_list[0]),len(split_list[0][0])) #channel/8, col:8, row:8
    return split_list

def Compress(block):
    cache = []
    bit_map = [] # 2d list size([9,9]) or ([4,4]) , each item for torch.Size([4,4,8]) or ([8,8,8])
    indicator = [] # 2d list size([9,9]) or ([4,4]) , each item for an int

    for i in range(len(block)): # h direction block number
        list_indicator_temp = []  # indicator element
        list_cache_temp = []
        list_bit_map = []
        for j in range(len(block[i])): # w direction block number
            orginal = block[i][j]
            temp = block[i][j].reshape(-1)
            compressed = temp[temp.nonzero()].reshape(-1)  # temp.nonzero(): index of nonzero
            all_zero = compressed.nonzero().reshape(-1).shape[0] == 0

            bit_map_tmp = (orginal>0).reshape(-1).float()
            if bit_map_tmp.shape[0]%16 != 0:
                bit_map_tmp =  torch.cat((bit_map_tmp ,torch.zeros((16-bit_map_tmp.shape[0]%16))))

            binary_mul = torch.tensor([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]).reshape(-1,1).float()
            bit_map_tmp = torch.matmul(bit_map_tmp.reshape(-1,16), binary_mul).reshape(-1)
            compressed = torch.cat((bit_map_tmp ,compressed))

            if not args.dense and compressed.shape[0]%8 != 0:
                compressed = torch.cat((compressed ,torch.zeros((8-compressed.shape[0]%8))))

            if all_zero:
                list_indicator_temp.append(0)
                list_cache_temp.append(torch.tensor([]))
                list_bit_map.append(0)
            elif orginal.reshape(-1).shape[0] <= compressed.shape[0]:
                #uncompressed
                list_indicator_temp.append(orginal.reshape(-1).shape[0])
                list_cache_temp.append(orginal.reshape(-1))
                list_bit_map.append(0)
            else:
                #compressed
                list_indicator_temp.append(compressed.shape[0])
                list_cache_temp.append(compressed)
                list_bit_map.append(bit_map_tmp.shape[0])


        indicator.append(list_indicator_temp)
        cache.append(list_cache_temp)
        bit_map.append(list_bit_map)

    return cache, indicator, bit_map

def MemoryCalculator(cache, bit_map, block):
    num_dram_bits = 0
    num_bmap_bits = 0
    num_sram_bits = 0
    h, w = len(cache), len(cache[0])

    for i in range(h):
        for j in range(w):
            num_dram_bits += cache[i][j].shape[0]*16
            num_bmap_bits += bit_map[i][j]*16
            if args.dense:
                num_sram_bits += 32 # 32 bits pointer
            else:
                #######################################
                ##### assume uniform block size = 8 ###
                #######################################
                num_sram_bits_temp = (block[i][j].reshape(-1).shape[0]-1)//8+1
                num_sram_bits += 8 if args.mode == "uniform"  else (math.log(num_sram_bits_temp, 2)-1)//1+1 # indicator bits
            # if args.mode == 'non_uniform':
            #     num_sram_bits += 16/4 # 16 indicator ::: 4 tile for 16bit
            # else:
            #     num_sram_bits += 8 #  6 indicator
    return num_dram_bits, num_bmap_bits, num_sram_bits

def Integrate(feature, ksp, idx):
    batch_size, c, h, w = feature.shape
    padding = ksp[2]
    feature = feature.view(-1, h, w) # torch.Size([8*batch size, 8, 27, 27])
    #feature = feature.permute(0,2,3,1) # torch.Size([8*batch size, 27, 27, 8])
    wsplit, w_pad, _ = ExtractTileParameter(ksp, w, output_size[0])
    hsplit, h_pad, _ = ExtractTileParameter(ksp, h, output_size[1])
    feature_padded = torch.zeros((feature.shape[0], h_pad, w_pad))  # torch.Size([8*batch size, 36, 36, 8]) or torch.Size([8*batch size, 32, 32, 8])
    feature_padded[:, padding:padding+h, padding:padding+w] = feature

    split_list = SplitFeature(feature_padded, wsplit, hsplit, csplit)

    if args.simulate_memory:
        all_dram_bits = 0
        all_bmap_bits = 0
        all_sram_bits = 0
        for block_2D in split_list:  # len(split_list) = batch size * 8
            cache, indicator, bit_map = Compress(block_2D)
            num_dram_bits, num_bmap_bits, num_sram_bits = MemoryCalculator(cache, bit_map, block_2D)

            all_dram_bits += num_dram_bits
            all_bmap_bits += num_bmap_bits
            all_sram_bits += num_sram_bits
        return all_dram_bits, all_bmap_bits, all_sram_bits
    else:
        indicators = []
        caches = []
        for block_2D in split_list:  # len(split_list) = batch size * 8
            cache, indicator, bit_map = Compress(block_2D)
            indicators.append(indicator)
            caches.append(cache)
        return cache, indicators

def main():
    print('===================================')
    print('Mode = ', args.mode)
    print('Memory = ', args.simulate_memory)
    print('Dense = ', args.dense)
    print('Network = ', args.model)
    print('Layer = ', args.layer)
    print('Output size = ',args.output_size)
    print('===================================')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    histogram_save_path = os.path.join('profile',args.model)
    if args.simulate_memory and not os.path.isdir(histogram_save_path):
        os.system('mkdir -p %s' % histogram_save_path)

    dataset = myDataset(data_type=data_type,
                        img_dir='/home/mediarti2/Dataset/Imagenet',
                        transform=transforms.Compose(transform_list))
    dataloader = Data.DataLoader(dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers= 30,
                                )
    net = Config['net'].cuda()
    net.eval()

    pbar = tqdm(total=1000,ncols=120)

    #build cache line dict
    cache_lines = {}
    for ksp in kernel_stride_padding[args.layer]:
        cache_lines[str(ksp)] = 0

    #build memory dict
    sum_dram_bits = {}
    sum_bmap_bits = {}
    sum_sram_bits = {}
    for ksp in kernel_stride_padding[args.layer]:
        sum_dram_bits[str(ksp)] = 0
        sum_bmap_bits[str(ksp)] = 0
        sum_sram_bits[str(ksp)] = 0

    write = {}
    img_total = 0
    for img, _ in dataloader:
        img = img.cuda()

        if args.model == "vdsr" :
            img = img[:,0,:,:].unsqueeze(1).cuda()

        img_total += 1
        features = net(img)  # kernel_stride_padding include kernel size, stride and padding
        feature = features[args.layer]

        for ksp in kernel_stride_padding[args.layer]:
            feature_scale = torch.nn.functional.interpolate(feature,scale_factor=(0.5,0.5)) if ksp == (1,2,0) else feature
            ksp_t = (1, 1, 0) if ksp == (1,2,0) else ksp

            if args.simulate_memory:
                all_dram_bits, all_bmap_bits, all_sram_bits = Integrate(feature_scale, ksp_t, args.layer)
                sum_dram_bits[str(ksp)] += all_dram_bits
                sum_bmap_bits[str(ksp)] += all_bmap_bits
                sum_sram_bits[str(ksp)] += all_sram_bits
            else:
                cache, indicators = Integrate(feature_scale, ksp_t, args.layer)
                indicators = Indicator2CacheLine(indicators)
                cache_line = AddressPatten2CacheLineNum(indicators, feature_scale.shape, ksp_t)
                cache_lines[str(ksp)] += cache_line
                write['cache_lines '+str(ksp)] = '{:.2f}'.format(cache_lines[str(ksp)]/img_total)

        pbar.set_postfix(write)
        pbar.update()

        if img_total == 1000:
            break

    pbar.close()

    if args.simulate_memory:
        for ksp in kernel_stride_padding[args.layer]:
            print('kernel size: %d, stride: %d, padding: %d'%(ksp[0], ksp[1], ksp[2]))
            print('Dram_bits: {:.2f}'.format(sum_dram_bits[str(ksp)]/img_total))
            print('Sram_bits: {:.2f}'.format(sum_sram_bits[str(ksp)]/img_total))
            print('Bmap bits: {:.2f}'.format(sum_bmap_bits[str(ksp)]/img_total))  #compressed data bits = dram bits - bmap bits
    else:
        for ksp in kernel_stride_padding[args.layer]:
            print('kernel size: %d, stride: %d, padding: %d'%(ksp[0], ksp[1], ksp[2]))
            print('Cache_lines: {:.2f}'.format(cache_lines[str(ksp)]/img_total))


if __name__ == '__main__':
    main()