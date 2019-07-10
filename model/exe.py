#!/usr/bin/env python3

if __name__ == '__main__':
    from GrateTile import *
    import numpy as np
    fc = FetchCalculator((3,2), (2,1), 8)
    #xyc, bmask = fc.Fetch((0,0,0), (6,5,8))
    xyc, bmask = fc.Fetch((19,16,0), (31,28,7))
    print(xyc)
    print(bmask)
