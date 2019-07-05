import numpy as np

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

	def Fetch(self, head, tsize):
		# For example, when xsplit=(3,2), ysplit=(2,1)
		# head = (0,0,0), tsize=(6,5,8)
		# The upper case pixels are fetched
		#
		# AAAXX|Aaaxx
		# AAAXX|Aaaxx
		# YYYBB|Yyybb
		# -----+-----
		# AAAXX|Aaaxx
		# AAAXX|Aaaxx
		# yyybb|yyybb
		#
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
		return 1,1

