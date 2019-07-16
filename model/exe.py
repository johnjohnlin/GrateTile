#!/usr/bin/env python3
import argparse
import math
parser = argparse.ArgumentParser(description='grate tiling')
parser.add_argument('--tiling_mode', action='store_true', help='type foe True for tiling')

args = parser.parse_args()
def get_para(ks_list,idx,h,w):
    k, s, p = ks_list[idx]
    f = (8-1)*s+k
    b = k-s
    a = f-2*b
    # if args.tiling_mode == True:
    #     w_n = f + 8*math.ceil((w+2*p-f)/8.)
    #     h_n = f + 8*math.ceil((h+2*p-f)/8.)
    # else:
    w_n = 8*math.ceil((w+2*p)/8.)
    h_n = 8*math.ceil((h+2*p)/8.)

    return a, b, w_n, h_n, p 


if __name__ == '__main__':
    from GrateTile import *
    import numpy as np
    #fc = FetchCalculator((3,2), (2,1), 8)
    #xyc, bmask = fc.Fetch((0,0,0), (6,5,8))
    #xyc, bmask = fc.Fetch((19,16,0), (31,28,7))
    ks_list=[(5,1,2),(3,1,1),(3,1,1),(3,1,1)]   # kernel_size, stride, padding
    hwc_list = [(27,27,64),(13,13,192),(13,13,384),(13,13,256)] # feature size and channel
    for i,hwc in enumerate(hwc_list):
        a, b, w_n, h_n, p = get_para(ks_list,i,hwc[0],hwc[1])
        #print(a, b, w_n,h_n)
        fc = FetchCalculator((b,a), (b,a), 8)
        amount = 0
        idc = 0
        while idc+8 <= hwc[2]:
            idy = 0
            while idy+a+b <= h_n:
                idx  = 0
                while idx+a+b <= w_n:
                    xyc, bmask = fc.Fetch((idx,idy,idc), (min(idx+a+b*2,w_n), min(idy+a+b*2,h_n), idc+8))
                    print('channel',(idx,idy,idc), (min(idx+a+b*2,w_n), min(idy+a+b*2,h_n), idc+8))
                    print(xyc)
                    print(bmask)
                    amount += 1
                    idx += a+b
                idy += a+b
            idc += a+b
        print('-------------------------------------------------------------------',amount,'-------------------------------------------------------------------')
        
    
            