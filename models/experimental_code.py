
import torch
import torch.nn as nn
from pprint import pprint 

# code from -  https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

# pool of square window of size=3, stride=2
m1 = nn.AvgPool2d(3, stride=2)
# The below filter has input & output of same shape. Doesn't seem to have any change/impact on input.
# refer - issue reported in same repo - 
# https://github.com/kuangliu/pytorch-cifar/issues/110
m2 = nn.AvgPool2d(1, stride=1)
# pool of non-square window
m3 = nn.AvgPool2d((3, 2), stride=(2, 1))
input = torch.randn(20, 16, 1, 1)
output = m2(input)
pprint(output.shape)


