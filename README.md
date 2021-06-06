# 1. Involution

The involution layer support for pytorch.

## 1.1. Contents
This module now contains two torch.nn.Modules named 
Involution1D and Involution2D which can be used in Pytorch.

The help file is written below:

Applies InvolutionND operation to the input using
the simplist kernel generation function $W_1\sigma(W_0X_{i,j})$,
where $W_0$ is the reduction matrix and $W_1$ is the spanning matrix.

Involution operater will generate kernel on each pixel using two matrix above, forming new large kernels and broadcasting it through channels.

In the original paper of involution layer, the size of input and output
should the same. So we add a convolution layer of kernel_size $1*1$ to give involution layer the ability of compressing and spanning channels.

For example, the input and output of Involution1D should be: 

    Shape:
    - Input: (B,C_in,Length)
    - Output: (B,C_out,Length)

For further details please refer to the doc of Inv.py.
