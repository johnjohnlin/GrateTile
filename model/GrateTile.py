import numpy as np
import math 
class FetchCalculator(object):
    def __init__(self, xsplit, ysplit, csplit):
        # For example, xsplit=(3,2), ysplit=(2,1)
        #
        # aaaxx
        # aaaxx
        # yyybb
        self.xsplit = xsplit
        self.ysplit = ysplit
        self.xblk_size_ = sum(xsplit)
        self.yblk_size_ = sum(ysplit)
        self.cblk_size_ = csplit
        
    def ClampBlock(self, block_id, head, tile_size):
        index_x, index_y = block_id[0] * self.xblk_size_, block_id[1] * self.yblk_size_
        l_side = block_id[0] * self.xblk_size_
        u_side = block_id[1] * self.yblk_size_
        r_side = l_side + self.xblk_size_
        d_side = u_side + self.yblk_size_

        l_side = max(l_side, head[0])
        u_side = max(u_side, head[1])
        r_side = min(r_side, tile_size[0])
        d_side = min(d_side, tile_size[1])
        
        return l_side, u_side, r_side, d_side, index_x, index_y
        
    def CalculateMask(self, l_side, u_side, r_side, d_side, index_x, index_y):
    
        # mask = np.ones((len(self.ysplit_), len(self.xsplit_)), dtype='i4')
        horizontal_mask = [1,1,1,1]
        vertical_mask   = [1,1,1,1]
        if l_side >= index_x+self.xsplit[0] and r_side >= index_x+self.xsplit[0]:
            horizontal_mask = [0,1,0,1]
        elif l_side <= index_x+self.xsplit[0] and r_side <= index_x+self.xsplit[0]:
            horizontal_mask = [1,0,1,0]
        
        if u_side >= index_y+self.ysplit[0] and d_side >= index_y+self.ysplit[0]:
            vertical_mask = [0,0,1,1]
        elif u_side <= index_y+self.ysplit[0] and d_side <= index_y+self.ysplit[0]:
            vertical_mask = [1,1,0,0]

        mask = [horizontal_mask[i] and vertical_mask[i] for i in range(4)]
        mask = np.array(mask)
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
        # The first one is the block id of [index_x,index_y,channel], so it's 4*3
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
        # print(head, tile_size)
        edge_l =  head[0]            // self.xblk_size_ #
        edge_r = (tile_size[0]-1)    // self.xblk_size_ 
        edge_u =  head[1]            // self.yblk_size_
        edge_d = (tile_size[1]-1)    // self.yblk_size_ 
        edge_f =  head[2]            // self.cblk_size_
        edge_b = (tile_size[2]-1)    // self.cblk_size_ 
        # print(edge_f, edge_b)
        # if (tile_size[0]) // self.xblk_size_ != math.ceil(tile_size[0]/self.xblk_size_)-1:
        #     print((tile_size[0]) , self.xblk_size_ )
        # edge_l = head[0]                 // self.xblk_size_
        # edge_r = math.ceil(tile_size[0]/self.xblk_size_)-1 #
        # edge_u = head[1]                 // self.yblk_size_
        # edge_d = math.ceil(tile_size[1]/self.yblk_size_)-1
        # edge_f = head[2]                 // self.cblk_size_
        # edge_b = math.ceil(tile_size[2]/self.cblk_size_)-1
        block_ids = []
        for c in range(edge_f,edge_b+1):
                for x in range(edge_l,edge_r+1):
                    for y in range(edge_u,edge_d+1):
                        block_ids.append([x,y,c])
        block_ids = np.array(block_ids)
        # <-------------- second one is the boolean mask ------------> #
        # head = (19,16,0), tile_size=(31,28,7)
        boolean_masks = np.empty(
                (block_ids.shape[0], len(self.xsplit)*len(self.ysplit)),
                dtype='i4')
        for i, block_id in enumerate(block_ids):
            l_side, u_side, r_side, d_side, index_x, index_y = self.ClampBlock(block_id,head,tile_size)
            # print(l_side, u_side, r_side, d_side, index_x, index_y)
            mask = self.CalculateMask(l_side, u_side, r_side, d_side, index_x, index_y)
            boolean_masks[i] = mask.flat
        # print(block_ids)
        # raise ValueError('good')
        return block_ids,boolean_masks


class CacheLineCalculator(object):
    def __init__(self, indicators, bit_maps):
        self.indicators = indicators
        self.bit_maps = bit_maps
    
    def Fetch(self, block_id, boolean_mask):
        num_cache_line = 0
        num_cache_line_bmap = 0
        for i in range(block_id.shape[0]):
            block_id_ = block_id[i]
            index_x, index_y, index_c = block_id_[0]*2 , block_id_[1]*2, block_id_[2]
            
            # print(index_x, index_y, index_c)
            # print(len(self.indicators), len(self.indicators[0]), len(self.indicators[0][0]))

            # raise ValueError('good')
            # print(len(self.indicators), len(self.indicators[0]), len(self.indicators[0][0]))
            mask = boolean_mask[i]
            # print(mask)
            num_cache_line += (self.indicators[index_c][index_y]   [index_x]   - 1) // 8 + 1 if mask[0] else 0
            num_cache_line += (self.indicators[index_c][index_y]   [index_x+1] - 1) // 8 + 1 if mask[1] else 0
            num_cache_line += (self.indicators[index_c][index_y+1] [index_x]   - 1) // 8 + 1 if mask[2] else 0
            num_cache_line += (self.indicators[index_c][index_y+1] [index_x+1] - 1) // 8 + 1 if mask[3] else 0

            # num_cache_line_bmap += math.ceil((self.bit_maps[index_c][index_y][index_x]/16/8).reshape(-1).shape[0]) if mask[0] else 0
            # num_cache_line_bmap += math.ceil((self.bit_maps[index_c][index_y][index_x+1]/16/8).reshape(-1).shape[0]) if mask[1] else 0
            # num_cache_line_bmap += math.ceil((self.bit_maps[index_c][index_y+1][index_x]/16/8).reshape(-1).shape[0]) if mask[2] else 0
            # num_cache_line_bmap += math.ceil((self.bit_maps[index_c][index_y+1][index_x+1]/16/8).reshape(-1).shape[0]) if mask[3] else 0  
            num_cache_line_bmap += 4
        return num_cache_line, num_cache_line_bmap       
