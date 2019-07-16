import numpy as np
from numpy import newaxis
import math 

class FetchCalculator(object):
    def __init__(self, xsplit, ysplit, csplit):
        # For example, xsplit=(3,2), ysplit=(2,1)
        # xsplit_cumsum=(0,3,5), ysplit_cumsum=(0,2,3)
        #
        # aaaxx
        # aaaxx
        # yyybb
        self.xsplit_ = xsplit
        self.ysplit_ = ysplit
        self.xsplit_cumsum_ = np.cumsum((0,) + xsplit, dtype='i4')
        self.ysplit_cumsum_ = np.cumsum((0,) + ysplit, dtype='i4')
        self.xblk_size_ = sum(xsplit)
        self.yblk_size_ = sum(ysplit)
        self.cblk_size_ = csplit

    def ClampBlock(self, block_id, head, tile_size):
        l_side = block_id[0] * self.xblk_size_
        u_side = block_id[1] * self.yblk_size_
        r_side = l_side + self.xblk_size_
        d_side = u_side + self.yblk_size_

        l_side_clamp = max(l_side, head[0]) - l_side
        u_side_clamp = max(u_side, head[1]) - u_side
        r_side_clamp = min(r_side, head[0] + tile_size[0]) - l_side
        d_side_clamp = min(d_side, head[1] + tile_size[1]) - u_side
        print(f"{l_side_clamp}:{r_side_clamp}")
        print(f"{u_side_clamp}:{d_side_clamp}")

        return l_side_clamp, u_side_clamp, r_side_clamp, d_side_clamp

    def CalculateMask(self, l_side, u_side, r_side, d_side):
        mask_row_l = l_side < self.xsplit_cumsum_[1:  ]
        mask_row_r = r_side > self.xsplit_cumsum_[ :-1]
        mask_col_u = u_side < self.ysplit_cumsum_[1:  ]
        mask_col_d = d_side > self.ysplit_cumsum_[ :-1]

        mask_row = np.bitwise_and(mask_row_l, mask_row_r)
        mask_col = np.bitwise_and(mask_col_u, mask_col_d)[:,newaxis]

        return np.bitwise_and(mask_col, mask_row)
    
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
        edge_l =  head[0]                     // self.xblk_size_
        edge_r = (head[0] + tile_size[0] - 1) // self.xblk_size_ + 1
        edge_u =  head[1]                     // self.yblk_size_
        edge_d = (head[1] + tile_size[1] - 1) // self.yblk_size_ + 1
        edge_f =  head[2]                     // self.cblk_size_
        edge_b = (head[2] + tile_size[2] - 1) // self.cblk_size_ + 1

        block_ids_mgrid = np.mgrid[edge_f:edge_b, edge_u:edge_d, edge_l:edge_r]
        block_ids = np.column_stack(list(b.flat for b in block_ids_mgrid)).astype('i4')

        # <-------------- second one is the boolean mask ------------> #
        # head = (19,16,0), tile_size=(12,12,7)
        boolean_masks = np.empty(
                (block_ids.shape[0], len(self.xsplit_)*len(self.ysplit_)),
                dtype='i4')
        for i, block_id in enumerate(block_ids):
            l_side, u_side, r_side, d_side = self.ClampBlock(block_id, head, tile_size)
            mask = self.CalculateMask(l_side, u_side, r_side, d_side)
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
