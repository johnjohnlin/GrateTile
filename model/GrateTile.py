import numpy as np
from numpy import newaxis
class FetchCalculator(object):
    def __init__(self, wsplit, hsplit, csplit):
        # For example, wsplit=(3,2), hsplit=(2,1), csplit=(8,)
        #
        # aaaxx
        # aaaxx
        # yyybb
        self.wsplit = wsplit
        self.hsplit = hsplit
        self.csplit = csplit
        self.wsplit_cumsum_ = np.cumsum((0,) + wsplit, dtype='i4')  # array([0, 3, 5], dtype=int32)
        self.hsplit_cumsum_ = np.cumsum((0,) + hsplit, dtype='i4')  # array([0, 2, 3], dtype=int32)
        self.csplit_cumsum_ = np.cumsum((0,) + csplit, dtype='i4')
        self.wblk_size_ = sum(wsplit)
        self.hblk_size_ = sum(hsplit)
        self.cblk_size_ = sum(csplit)

    def ClampBlock(self, block_id, head, tile_size):
        l_side = block_id[2] * self.wblk_size_
        u_side = block_id[1] * self.hblk_size_
        f_side = block_id[0] * self.cblk_size_
        r_side = l_side + self.wblk_size_
        d_side = u_side + self.hblk_size_
        b_side = f_side + self.cblk_size_

        l_side_clamp = max(l_side, head[0]) - l_side
        u_side_clamp = max(u_side, head[1]) - u_side
        f_side_clamp = max(f_side, head[2]) - f_side
        r_side_clamp = min(r_side, head[0] + tile_size[0]) - l_side
        d_side_clamp = min(d_side, head[1] + tile_size[1]) - u_side
        b_side_clamp = min(b_side, head[2] + tile_size[2]) - f_side

        return l_side_clamp, r_side_clamp, u_side_clamp, d_side_clamp, f_side_clamp, b_side_clamp

    def CalculateMask(self, l_side, r_side, u_side, d_side, f_side, b_side):
        mask_l = l_side < self.wsplit_cumsum_[1:  ]
        mask_r = r_side > self.wsplit_cumsum_[ :-1]
        mask_u = u_side < self.hsplit_cumsum_[1:  ]
        mask_d = d_side > self.hsplit_cumsum_[ :-1]
        mask_f = f_side < self.csplit_cumsum_[1:  ]
        mask_b = b_side > self.csplit_cumsum_[ :-1]

        mask_row = np.bitwise_and(mask_l, mask_r)   #shape([2,])
        mask_col = np.bitwise_and(mask_u, mask_d)[:,newaxis]    #shape([2,1])
        mask_ch  = np.bitwise_and(mask_f, mask_b)[:,newaxis][:,newaxis] #shape([2,1,1])


        '''
        mask_row
        [T T]: all block
        [T F]: left block
        [F T]: right block

        mask_col
        [[T]
         [T]]: all block
        [[T]
         [F]]: up block
        [[F]
         [T]]: down block
        '''
        mask = np.bitwise_and(mask_col, mask_row)
        mask = np.bitwise_and(mask, mask_ch)

        ############################## testing ################################
        # horizontal_mask = [1,1,1,1]
        # vertical_mask   = [1,1,1,1]
        # if l_side >= self.wsplit[0] and r_side >= self.wsplit[0]:
        #     horizontal_mask = [0,1,0,1]
        # elif l_side <= self.wsplit[0] and r_side <= self.wsplit[0]:
        #     horizontal_mask = [1,0,1,0]

        # if u_side >= self.hsplit[0] and d_side >= self.hsplit[0]:
        #     vertical_mask = [0,0,1,1]
        # elif u_side <= self.hsplit[0] and d_side <= self.hsplit[0]:
        #     vertical_mask = [1,1,0,0]

        # mask = [horizontal_mask[i] and vertical_mask[i] for i in range(4)]
        # test = np.array(mask)

        # if not np.all(test*1 == mask):
        #     print('expect')
        #     print(test)
        #     print('test')
        #     print(mask)
        #     raise ValueError('fall')

        ######################################################################
        return mask

    def Fetch(self, head, tile_size):
        # For example, when wsplit=(3,2), hsplit=(2,1)
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
        # The first one is the block id of [channel, index_h, index_w], so it's 4*3
        # [0,0,0]
        # [0,0,1]
        # [0,1,0]
        # [0,1,1]
        # The first one is the boolean mask of the fetched sub-tiles.
        # In the row-major order.
        # [1,1,1,1]
        # [1,0,1,0]
        # [1,1,0,0]
        # [1,0,0,0]
        # <-------------- first one is the block id ------------> #

        edge_l =  head[0]                     // self.wblk_size_ # floor
        edge_r = (head[0] + tile_size[0] - 1) // self.wblk_size_ # ceil-1
        edge_u =  head[1]                     // self.hblk_size_ # floor
        edge_d = (head[1] + tile_size[1] - 1) // self.hblk_size_ # ceil-1
        edge_f =  head[2]                     // self.cblk_size_ # floor
        edge_b = (head[2] + tile_size[2] - 1) // self.cblk_size_ # ceil-1

        block_ids_mgrid = np.mgrid[edge_f:edge_b+1, edge_u:edge_d+1, edge_l:edge_r+1]
        block_ids = np.column_stack([b.flat for b in block_ids_mgrid]).astype('i4')
        ################### testing ##################### Actually, it is faster than the two lines of numpy code.
        # block_ids = []
        # for c in range(edge_f,edge_b+1):
        #         for x in range(edge_l,edge_r+1):
        #             for y in range(edge_u,edge_d+1):
        #                 block_ids.append([x,y,c])
        # block_ids = np.array(block_ids)

        # if not np.all(test == block_ids):
        #     print('test')
        #     print(block_ids)
        #     print('expect')
        #     print(test)
        #     raise ValueError('Fall')
        #################################################

        # <-------------- second one is the boolean mask ------------> #
        # head = (19,16,0), tile_size=(12,12,7)
        boolean_masks = np.empty(
                (block_ids.shape[0], len(self.wsplit)*len(self.hsplit)*len(self.csplit)), dtype='i4')

        for i, block_id in enumerate(block_ids):
            l_side, r_side, u_side, d_side, f_side, b_side = self.ClampBlock(block_id, head, tile_size)
            mask = self.CalculateMask(l_side, r_side, u_side, d_side, f_side, b_side)
            boolean_masks[i] = mask.flat

        return block_ids, boolean_masks

class CacheLineCalculator(object):
    def __init__(self, indicators, wsplit, hsplit, csplit):
        # assume the block size is 8*8
        self.indicators = indicators
        self.num_wsplit = len(wsplit)
        self.num_hsplit = len(hsplit)
        self.num_csplit = len(csplit)
        # print(wsplit,hsplit,csplit)
        # print(len(indicators), len(indicators[0]), len(indicators[0][0]))

    def Fetch(self, block_ids, boolean_mask):
        num_cache_line = 0
        for i,(index_c, index_h, index_w) in enumerate(block_ids):  # block_ids shape([num_block,3])

            index_w *= self.num_wsplit
            index_h *= self.num_hsplit
            index_c *= self.num_csplit

            subtile_ids_mgrid = np.mgrid[index_c:index_c+self.num_csplit, index_h:index_h+self.num_hsplit, index_w:index_w+self.num_wsplit]
            subtile_ids = np.column_stack([b.flat for b in subtile_ids_mgrid]).astype('i4')  # shape([num_wsplit*num_hsplit, 2])

            mask = boolean_mask[i]  # shape([num_wsplit*num_hsplit,])
            for j, (idc, idh, idw) in enumerate(subtile_ids):
                # print(idc, idh, idw)
                num_cache_line += self.indicators[idc][idh][idw] if mask[j] else 0

        return num_cache_line
