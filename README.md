# 1. Involution

The involution layer support for pytorch.

## 1.1. Contents
This module now contains two torch.nn.Modules named 
Involution1D and Involution2D which can be used in Pytorch.

The help file is written below:

Applies InvolutionND operation to the input using
the simplist kernel generation function $W_1\sigma(W_0X_{i,j})$,
where $W_0$ is the reduction matrix and $W_1$ is the spanning matrix.

Involution operater will generate kernel on each pixel using two matrix above, forming new large kernels and broadcast it through channels.

Please Note that in involution layer, the size of input and output are the same, for example the inpnut of Involution1D should be: 

    Shape:
    - Input: (B,C,Length)
    - Output: (B,C,Length)

For further details please refer to the help file in Inv.py.


