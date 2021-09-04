# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:11:50 2021

Involution support for Pytorch.

@author: TeCai
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Involution1D(nn.Module):
    __doc__ = r""" Applies Involution1D operation to the input using
    the simplist kernel generation function W_1\sigma(W_0X_{i,j}),
    where W_0 is the reduction matrix and W_1 is the spanning matrix.
    
    Involution operater will generate kernel on each pixel using two matrix above,
    forming new large kernels and broadcast it through channels.
    
    C_in: # Input Channel.
    C_out: # Output Channel.
    r: Reduction_ratio(Compile C_in to C//r, which is the 
    role of W_0).
    K: Kernel_size. On each pixel there would be a K*K kernel.
    B: Batch_Size.
    G: # Group, self.span convert c//r reduced channels into K*G and 
    become involution kernels.
    
    
    The Total kernel size should be of length*K*G size.
    
    The original involution layer requires C_in and C_out
    to be the same, this some how limits the range of application of 
    involution layer. Therefore, a linear transform is applied
    after the involution operation in order to adjust the number of channel.

    Shape:
        - Input: math:'(B,C_in,Length)'
        - Output: math:'(B,C_out,Length)'
    """
    
    # Note that padding should be matched with kernel_size and C should be divisible by C
    
    
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, Group: int,  padding: int , reduction_ratio = 1, dilation = 1 ):
        
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.reduction_ratio = reduction_ratio
        self.Group = Group
        self.reduce = nn.Conv1d(in_channel,in_channel//reduction_ratio,1)
        self.span = nn.Conv1d(in_channel//reduction_ratio, kernel_size*Group, 1)
        self.relu = nn.ReLU()
        if in_channel != out_channel:
            self.transform = nn.Conv1d(in_channel,out_channel,1)
        self.unfold = nn.Unfold(kernel_size = (1,kernel_size), dilation = dilation, padding = (0,padding))
        
        if (in_channel % Group) != 0 :
            raise ValueError('in_channels must be divisible by groups')
        if padding*2+1 != kernel_size:
            raise ValueError('kernel_size must match with padding s.t. kernel_size = padding*2+1 while dilation = 1')
        
    def forward(self,x):
        reduced = self.reduce(x)
        relued_reduced = self.relu(reduced)
        spanned = self.span(relued_reduced)        
        Batch_Size = x.size()[0]
        x_unfolded = self.unfold(x.view(Batch_Size, self.in_channel, 1,-1)) #(B,C*K,Length)
        x_unfolded = x_unfolded.view(Batch_Size, self.Group, self.in_channel//self.Group, self.kernel_size,-1) #(B,G,C//G,Length)
        kernel = spanned.view(Batch_Size, self.Group, self.kernel_size, -1).unsqueeze(2)
        
        #Boardcasting and adding
        
        out = torch.mul(kernel,x_unfolded).sum(dim = 3) #dim of kernel_size
        out = out.view(Batch_Size, self.in_channel, -1)
        if self.in_channel != self.out_channel:
            out = self.transform(out)
            
        
        return out
        
        
class Involution2D(nn.Module):
    __doc__ = r""" Applies Involution2D operation to the input using
    the simplist kernel generation function W_1\sigma(W_0X_{i,j}), 
    where W_0 is the reduction matrix and W_1 is the spanning matrix.
    
    Involution operater will generate kernel on each pixel using two matrix above,
    forming new large kernels and broadcast it through channels.
    
    C_in: # Input Channel.
    C_out: # Output Channel.
    r: reduction_ratio( compile #Channel to C//r, which is the 
    role of W_0).
    K: kernel_size. On each pixel there would be a K*K kernel.
    B: Batch_Size.
    G: # Group, self.span convert c//r reduced channels into K*G and 
    become involution kernel.
    H: The Height of the input, i.e. the first dimension.
    W: The Width of the input, i.e. the second dimension. 
    
    Now Involution2D only support a loacl kernel which has the same 
    size in height and weight 
    
    
    The Total kernel size should be of H*W*K*K*G size.

    The original involution layer requires C_in and C_out
    to be the same, this some how limits the range of application of 
    involution layer. Therefore, a linear transform is applied
    after the involution operation in order to adjust the number of channel.

    Shape:
        - Input: math:'(B,C_in,H,W)'
        - Output: math:'(B,C_out,H,W)'
    """
    def __init__(self, in_channel:int, out_channel: int, kernel_size:int, Group:int,  padding:int , reduction_ratio = 1, dilation = 1 ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.reduction_ratio = reduction_ratio
        self.Group = Group
        self.reduce = nn.Conv2d(in_channel,in_channel//reduction_ratio,1)
        self.span = nn.Conv2d(in_channel//reduction_ratio, kernel_size**2*Group, 1)
        self.relu = nn.ReLU()
        if in_channel != out_channel:
            self.transform = nn.Conv2d(in_channel,out_channel,1)
        self.unfold = nn.Unfold(kernel_size = kernel_size, dilation = dilation, padding = padding)
        
        if (in_channel % Group) != 0 :
            raise ValueError('in_channels must be divisible by groups')
        if padding*2+1 != kernel_size:
            raise ValueError('kernel_size must match with padding s.t. kernel_size = padding*2+1 while dilation = 1')

    def forward(self,x):
        reduced = self.reduce(x)
        relued_reduced = self.relu(reduced)
        spanned = self.span(relued_reduced)        
        [Batch_Size,_,H,W] = x.size()
        x_unfolded = self.unfold(x) #(B,C*K*K,H*W)
        x_unfolded = x_unfolded.view(Batch_Size, self.Group, self.in_channel//self.Group, self.kernel_size**2, H, W) #(B,G,C//G,K*K,H,W)
        kernel = spanned.view(Batch_Size, self.Group, self.kernel_size**2, H, W).unsqueeze(2)
        
        #Boardcasting and adding
        
        out = torch.mul(kernel,x_unfolded).sum(dim = 3) #dim of kernel_size       
        out = out.view(Batch_Size, self.in_channel, H, W)
        if self.in_channel != self.out_channel:
            out = self.transform(out)
        return out





            