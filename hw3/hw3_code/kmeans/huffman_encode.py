import numpy as np 
import heapq

class node:
    def __init__(self, freq, left=None, right=None):
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, value):
        return self.freq < value.freq

    def __eq__(self, value):
        return self.freq == value.freq

    def __gt__(self, value):
        return self.freq > value.freq
        
    def calc(self, depth):
        if self.left is None and self.right is None:
            # leaf node
            print('leaf node - frequency:', self.freq, 'depth:', depth)
            return depth * self.freq
        else:
            # non-leaf node
            return self.left.calc(depth+1) + self.right.calc(depth+1)

data = np.load('cnt.npz')['cnt']    # data.shape = (64, )
heap = []
for i in data:
    heap.append(node(i))
Traceback (most recent call last):
  File "huffman_encode.py", line 28, in <module>
    data = np.load('cnt.npz')['cnt']    # data.shape = (64, )
  File "E:\Anaconda\envs\tensorflow\lib\site-packages\numpy\lib\npyio.py", line 423, in load
    magic = fid.read(N)
OSError: [Errno 22] Invalid argument
while len(heap) > 1:
    lnode = heapq.heappop(heap)
    rnode = heapq.heappop(heap)
    heapq.heappush(heap, node(lnode.freq + rnode.freq, lnode, rnode))
root = heapq.heappop(heap)
totalBits = root.calc(0)
print('totalBits:', totalBits, 'Compress Ratio:', totalBits/(root.freq*24))
