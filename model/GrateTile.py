import numpy as np
import math 
class FetchCalculator(object):
    def __init__(self, xsplit, ysplit, channel):
        # For example, xsplit=(3,2), ysplit=(2,1)
        #
        # aaaxx
        # aaaxx
        # yyybb
        self.xsplit = xsplit
        self.ysplit = ysplit
        self.channel = channel
    def Fetch_side(self,block_id_,head,tsize):
        x = block_id_[0]*sum(self.xsplit)
        y = block_id_[1]*sum(self.ysplit)
        l_side = x
        u_side = y
        if x < head[0]:
            l_side = head[0]
        if y < head[1]:
            u_side = head[1]
        r_side = x+sum(self.xsplit)
        d_side = y+sum(self.ysplit)
        if x+sum(self.xsplit) > tsize[0]:
            r_side = tsize[0]
        if y+sum(self.xsplit) > tsize[1]:
            d_side = tsize[1]
        return l_side, u_side, r_side, d_side, x, y
    def cal_mask(self, l_side, u_side, r_side, d_side, x, y):
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
    
    def Fetch(self, head, tsize):
        # For example, when xsplit=(3,2), ysplit=(2,1)
        # head = (0,0,0), tsize=(6,5,8)
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
        csplit = 8
        edge_l = math.floor(head[0]/sum(self.xsplit)) #
        edge_r = math.ceil(tsize[0]/sum(self.xsplit))-1 #
        edge_u = math.floor(head[1]/sum(self.ysplit)) #
        edge_d = math.ceil(tsize[1]/sum(self.ysplit))-1
        edge_f = math.floor(head[2]/csplit)
        edge_b = math.ceil(tsize[2]/csplit)-1

        block_id = []
        for c in range(edge_f,edge_b+1):
                for x in range(edge_l,edge_r+1):
                    for y in range(edge_u,edge_d+1):
                        block_id.append([x,y,c])
        #block_id = np.array(block_id)
        # <-------------- second one is the boolean mask ------------> #
        # head = (19,16,0), tsize=(31,28,7)
        boolean_mask = []
        for block_id_ in block_id:

            l_side, u_side, r_side, d_side, x, y = self.Fetch_side(block_id_,head,tsize)
            mask = self.cal_mask(l_side, u_side, r_side, d_side, x, y)
            #print(block_id_,boolean_mask)
            boolean_mask.append(mask)
        return block_id,boolean_mask

    