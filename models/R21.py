import math
import torch

import torch.nn as nn
from torch.nn.modules.utils import _triple
from torch.autograd import Variable
from torch.nn import functional as F
global tag  
# class ChannelAttention(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias):
#         super(ChannelAttention,self).__init__()
#         self.avg_pool=nn.AdaptiveAvgPool3d((in_channels,None,None))
#         self.max_pool=nn.AdaptiveMaxPool3d((in_channels,None,None))
#         self.fc1=nn.Conv3d(in_channels, out_channels, kernel_size,stride=stride, padding=padding,bias=False)
#         self.relu1=nn.ReLU()
#         self.fc2=nn.Conv3d(out_channels, out_channels, kernel_size,stride=stride, padding=padding,bias=False)
#         self.sigmoid=nn.Sigmoid()
#     def forward(self,x):
#         x=x.permute(0,2,1,3,4)
#         x=self.avg_pool(x)
#         x=x.permute(0,2,1,3,4)
#         avg_out=self.fc2(self.relu1(self.fc1(x)))
#         x=x.permute(0,2,1,3,4)
#         x=self.max_pool(x)
#         x=x.permute(0,2,1,3,4)
#         max_out=self.fc2(self.relu1(self.fc1(x)))
#         # print(avg_out.shape,' and ',max_out.shape)
#         out=avg_out+max_out
#         out=self.sigmoid(out)*out
#         return out
# #空间注意力机制，关注Mi*Ni-1*1*d*d中的d*d
# class SpatialAttention(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias):
#         super(SpatialAttention,self).__init__()
#         self.out_channels=out_channels
#         self.conv1=nn.Conv3d(in_channels+2,out_channels, kernel_size,stride=stride, padding=padding, bias=bias)
#         self.sigmoid=nn.Sigmoid()
#     def forward(self,x):
#         avg_out=torch.mean(x,dim=1,keepdim=True)
#         max_out,_=torch.max(x,dim=1,keepdim=True)
#         x=torch.cat([avg_out,max_out,x],dim=1)
#         x=self.conv1(x)
#         out=self.sigmoid(x)*x
#         return output
# class ChannelAttention(nn.Module):
#     def __init__(self,out_channels,kernel_size):
#         super(ChannelAttention,self).__init__()
#         if out_channels==128:
#         	channels=16 
#         elif out_channels==64:
#         	channels=32  
#         elif out_channels==256:
#             channels=8 
#         else: 
#             channels=4
#         self.avg_pool=nn.AdaptiveAvgPool3d((channels,None,None))
#         self.max_pool=nn.AdaptiveMaxPool3d((channels,None,None))
#         self.overloop_pool=nn.AvgPool3d(3,1,1) 
#         self.fc1=nn.Conv3d(out_channels,out_channels,kernel_size,bias=False)
#         self.relu=nn.ReLU()
#         self.fc2=nn.Conv3d(out_channels,out_channels,kernel_size,bias=False)
#         self.sigmoid=nn.Sigmoid()
#     def forward(self,x): 
#         a=self.avg_pool(x)
#         # avg_out=self.relu1(self.fc1(a))
#         b=self.max_pool(x)
#         # max_out=self.relu1(self.fc1(b))
#         out=0.4*a+0.6*b  
#         out=self.relu(self.fc1(out))
#         out=self.fc2(out) 
#         out=self.overloop_pool(out)  
#         out=self.sigmoid(out)*out
#         return out
# 空间注意力机制，关注Mi*Ni-1*1*d*d中的d*d
class SpatialAttention(nn.Module):
    def __init__(self,out_channels,kernel_size,padding):
        super(SpatialAttention,self).__init__()
        # assert kernel_size in (3,7),'kernel_size must be 3 or 7'
        # padding=3 if kernel_size==7 else 1
        self.conv1=nn.Conv3d(2+out_channels,out_channels,kernel_size,stride=1,padding=1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=torch.mean(x,dim=1,keepdim=True)
        max_out,_=torch.max(x,dim=1,keepdim=True)
        x=torch.cat([x,avg_out,max_out],dim=1)
        # print(4,x.shape)
        out=self.conv1(x)
        # print(5,out.shape)
        out=self.sigmoid(out)*out
        return out
class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, first_conv=False):
        super(SpatioTemporalConv, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_stride = (1, stride[1], stride[2])
        spatial_padding = (0, padding[1], padding[2])

        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)

        if first_conv:
            intermed_channels = 45
        else:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / (
                        kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn1 = nn.BatchNorm3d(intermed_channels)

        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)
        # self.conv=nn.Conv3d(in_channels, out_channels, temporal_kernel_size,
        #                                stride=temporal_stride, padding=temporal_padding, bias=bias)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        # x=self.relu(self.bn2(self.conv(x)))
        return x 

class ResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor
            out_channels (int): Number of channels in the output produced by the block
            kernel_size (int or tuple): Size of the convolving kernels
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(ResBlock, self).__init__()
        # global tag
        # tag=64
        # self.out_channels=out_channels
        self.downsample = downsample
        padding = kernel_size // 2

        if self.downsample:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        # self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.sa=SpatialAttention(out_channels, kernel_size, padding=padding)
        # self.ca=ChannelAttention(out_channels,1)
        self.relu = nn.ReLU()
    def forward(self, x):
        global tag   
        res = self.relu(self.bn1(self.conv1(x)))
        # res = self.conv2(res)
        # print(1,res.shape)
        res=self.sa(res)  
        # print(2,res.shape)
        res=self.bn2(res) 
        # print('res',res.shape)   
        if self.downsample:   
            x = self.downsamplebn(self.downsampleconv(x))
        # print(3,x.shape,res.shape)
        return self.relu(x + res)
   

class ResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor
            out_channels (int): Number of channels in the output produced by the layer
            kernel_size (int or tuple): Size of the convolving kernels
            layer_size (int): Number of blocks to be stacked to form the layer
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """
    def __init__(self, in_channels, out_channels, kernel_size, layer_size, downsample=False):
        super(ResLayer, self).__init__()
        # implement the first block
        self.block1 = ResBlock(in_channels, out_channels, kernel_size, downsample)
        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [ResBlock(out_channels, out_channels, kernel_size)]
    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x

class FeatureLayer(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
    """

    def __init__(self, layer_sizes, input_channel=3):
        super(FeatureLayer, self).__init__()

        self.conv1 = SpatioTemporalConv(input_channel, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                        first_conv=True)
        self.conv2 = ResLayer(64, 64, 3, layer_sizes[0])
        self.conv3 = ResLayer(64, 128, 3, layer_sizes[1],downsample=True)     
        self.conv4 = ResLayer(128, 256, 3, layer_sizes[2], downsample=True)
        self.conv5 = ResLayer(256, 512, 3, layer_sizes[3], downsample=True)
        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)

        return x.view(-1, 512)


class R2Plus1D(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.
        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
        """

    def __init__(self, num_classes, layer_sizes, input_channel=3):
        super(R2Plus1D, self).__init__()

        self.feature = FeatureLayer(layer_sizes, input_channel)
        self.fc = nn.Linear(512, num_classes)

        self.__init_weight()

    def forward(self, x):
        x = self.feature(x)
        logits = self.fc(x)

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):  
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 32,112,112)
    net = R2Plus1D(101, (3, 4, 6, 3))
    outputs = net.forward(inputs)
    print(outputs.size())