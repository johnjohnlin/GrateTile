import numpy as np
import math 

class FetchCalculator(object):
    def __init__(self, xsplit, ysplit, csplit):
        # For example, xsplit=(3,2), ysplit=(2,1)
        #
        # aaaxx
        # aaaxx
        # yyybb
        self.xsplit_ = xsplit
        self.ysplit_ = ysplit
        self.xblk_size_ = sum(xsplit)
        self.yblk_size_ = sum(ysplit)
        self.cblk_size_ = csplit

    def ClampBlock(self, block_id, head, tile_size):
        l_side = block_id[0] * self.xblk_size_
        u_side = block_id[1] * self.yblk_size_
        r_side = l_side + self.xblk_size_
        d_side = u_side + self.yblk_size_

        l_side = max(l_side, head[0])
        u_side = max(u_side, head[1])
        r_side = min(r_side, head[0] + tile_size[0])
        d_side = min(d_side, head[1] + tile_size[1])

        return l_side, u_side, r_side, d_side, x, y

    def CalculateMask(self, l_side, u_side, r_side, d_side, x, y):
        mask = np.ones((len(self.ysplit_), len(self.xsplit_)), dtype='i4')
        mask1 = [1,1,1,1]
        mask2 = [1,1,1,1]
        if l_side >= x+self.xsplit[0] and r_side >= x+self.xsplit[0]:
            mask1 = [0,1,0,1]
        elif l_side <= x+self.xsplit[0] and r_side <= x+self.xsplit[0]:
            mask1 = [1,0,1,0]
        
        if u_side >= y+self.ysplit[0] and d_side >= y+self.ysplit[0]:
            mask2 = [0,0,1,1]
        elif u_side <= y+self.ysplit[0] and d_side <= y+self.ysplit[0]:
            mask2 = [1,1,0,0]

        mask = [mask1[i] and mask2[i] for i in range(4)]

        return mask
    
    def Fetch(self, head, tile_size):
        # For example, when xsplit=(3,2), ysplit=(2,1)
        # head = (0,0,0), tile_size=(6,5,8)
        # The upper case pixels are fetched
        # AAAXX|Aaaxx
        # AAAXX|Aaaxx
        # YYYBB|Yyybb
        # -----+-----
        # AAAXX|Aaaxx
        # AAAXX|Aaaxx
        # yyybb|yyybb

        # It should return two 2D integer numpy arrays
        # Since it fetch 4 tiles, their #rows=4
        # The first one is the block id of [x,y,channel], so it's 4*3
        # [0,0,0]
        # [0,1,0]
        # [1,0,0]
        # [1,1,0]
        # The first one is the boolean mask of the fetched sub-tiles.
        # In the row-major order.
        # [1,1,1,1]
        # [1,0,1,0]
        # [1,1,0,0]
        # [1,0,0,0]
        # <-------------- first one is the block id ------------> #
        edge_l =  head[0]                     / self.xblk_size_
        edge_r = (head[0] + tile_size[0] - 1) / self.xblk_size_ + 1
        edge_u =  head[1]                     / self.yblk_size_
        edge_d = (head[1] + tile_size[1] - 1) / self.yblk_size_ + 1
        edge_f =  head[2]                     / self.cblk_size_
        edge_b = (head[2] + tile_size[2] - 1) / self.cblk_size_ + 1

        block_ids_mgrid = np.mgrid[edge_f:edge_b, edge_u:edge_d, edge_l:edge_r]
        block_ids = np.column_stack(b.flat for b in block_id_mgrid).astype('i4')
        print(block_ids)

        # <-------------- second one is the boolean mask ------------> #
        # head = (19,16,0), tile_size=(12,12,7)
        boolean_masks = np.empty(
                (block_ids.shape[0], len(self.xsplit_)*len(self.ysplit_)),
                dtype='i4')
        for i, block_id in enumerate(block_ids):
            l_side, u_side, r_side, d_side, x, y = self.FetchSide(block_id, head, tile_size)
            mask = self.CalculateMask(l_side, u_side, r_side, d_side, x, y)
            boolean_masks[i] = mask.flat
        return block_ids, boolean_masks


class CacheLineCalculator(object):
    def __init__(self, indicators, bit_maps):
        self.indicators = indicators
        self.bit_maps = bit_maps
    
    def Fetch(self, block_id, boolean_mask):
        num_cache_line = 0
        for i, block_id_ in enumerate(block_id):
            idx, idy, idc = block_id_[0]*2 , block_id_[1]*2, block_id_[2]
            # print(idx, idy, idc)
            mask = boolean_mask[i]
            num_cache_line += math.ceil(self.indicators[idc][idy][idx]/8) if mask[0] else 0
            num_cache_line += math.ceil(self.indicators[idc][idy][idx+1]/8) if mask[1] else 0
            num_cache_line += math.ceil(self.indicators[idc][idy+1][idx]/8) if mask[2] else 0
            num_cache_line += math.ceil(self.indicators[idc][idy+1][idx+1]/8) if mask[3] else 0

            num_cache_line += math.ceil((self.bit_maps[idc][idy][idx]/16/8).reshape(-1).shape[0]) if mask[0] else 0
            num_cache_line += math.ceil((self.bit_maps[idc][idy][idx+1]/16/8).reshape(-1).shape[0]) if mask[1] else 0
            num_cache_line += math.ceil((self.bit_maps[idc][idy+1][idx]/16/8).reshape(-1).shape[0]) if mask[2] else 0
            num_cache_line += math.ceil((self.bit_maps[idc][idy+1][idx+1]/16/8).reshape(-1).shape[0]) if mask[3] else 0  

        return num_cache_line          
