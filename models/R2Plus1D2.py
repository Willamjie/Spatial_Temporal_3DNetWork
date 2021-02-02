import math
import torch

import torch.nn as nn
from torch.nn.modules.utils import _triple
from torch.autograd import Variable
from torch.nn import functional as F
# class Self_Attn(nn.Module):
#     """ Self attention Layer"""
#     def __init__(self,in_channels,out_channels,kernel_size,stride, padding):
#         super(Self_Attn,self).__init__()
#         self.out_channels = out_channels
#         self.query_conv = nn.Conv3d(in_channels = in_channels, out_channels =out_channels , kernel_size= kernel_size,stride=stride,padding=padding,bias=False)
#         self.key_conv = nn.Conv3d(in_channels = in_channels, out_channels = out_channels , kernel_size= kernel_size,stride=stride,padding=padding)
#         self.value_conv = nn.Conv3d(in_channels = in_channels , out_channels = out_channels , kernel_size= kernel_size,stride=stride,padding=padding)
#         # self.gamma = nn.Parameter(torch.zeros(1))
#         # self.pooling=nn.MaxPool3d(16,stride=2,return_indices=True)
#         # self.unpooling=nn.MaxUnpool3d(16,stride=2)
#         self.softmax  = nn.Softmax(dim=1) #
#         # print(in_channels,out_channels)
#     def forward(self,x):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature 
#                 attention: B X N X N (N is Width*Height)
#         """
#         # x,indices=self.pooling(Variable(x_in))
#         # print(self.query_conv(x).size()) 
#         # print(type(x))
#         # embed=nn.Embedding(x.shape[0]*x.shape[1]*x.shape[2],1)
#         # x=x.type(torch.LongTensor)
#         # x=embed(x).reshape(x.shape)
#         # x=x.type(torch.cuda.FloatTensor)
#         batch_size,c,t,width ,height = self.query_conv(x).size()
#         proj_query  = self.query_conv(x).view(batch_size*c,t,width*height).permute(0,2,1) # B X CX(N)
#         # print(proj_query.size())
#         proj_key =  self.key_conv(x).view(batch_size*c,t,width*height)# B X C x (*W*H)
#         # print(proj_key.size())
#         energy =  torch.matmul(proj_query,proj_key) # transpose check
#         # print(energy.shape)
#         # print(energy.shape)
#         attention = self.softmax(energy) # BX (N) X (N) 
#         # attention=F.softmax(energy, dim=-1)
#         proj_value = self.value_conv(x).view(batch_size,-1,t,width,height) # B X C X N
#         # print(proj_key.shape,attention.permute(0,2,1).shape)
#         out = torch.matmul(proj_value.view(batch_size*c,t,width*height),attention.permute(0,2,1))
#         out = out.view(batch_size,c,t,width,height)
#         # out=self.unpooling(out, indices)
#         # out = out + x#.view(C,t,width,height)
#         return out
# class PositionalEncoding(nn.Module):
#     "Implement the PE function."
#     def __init__(self, d_model, dropout, max_len=32):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         # print(position)
#         # print(position.shape)
#         div_term = torch.exp(torch.arange(0., d_model, 2)*-(math.log(10000.0) / d_model)).float()
#         # print(pe)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
        
#     def forward(self, x):
#         b,c,t,_=x.size()
#         x= x + Variable(self.pe[:,:x.size()[2]],requires_grad=False)
#         return self.dropout(x).view(b,c,t,112,112)
#channel注意力机制，关注Ni*Mi*t*1*1中的t
class ChannelAttention(nn.Module):
    def __init__(self,out_channels,kernel_size):
        super(ChannelAttention,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool3d(1)
        self.max_pool=nn.AdaptiveMaxPool3d(1)
        self.fc1=nn.Conv3d(out_channels,1,kernel_size,bias=False)
        self.relu1=nn.ReLU()
        self.fc2=nn.Conv3d(1,out_channels,kernel_size,bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out=self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out=avg_out+max_out
        out=self.sigmoid(out)
        return out
#空间注意力机制，关注Mi*Ni-1*1*d*d中的d*d
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size,out_channels):
        super(SpatialAttention,self).__init__()
        # assert kernel_size in (3,7),'kernel_size must be 3 or 7'
        # padding=3 if kernel_size==7 else 1
        self.conv1=nn.Conv3d(2,out_channels,kernel_size,bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=torch.mean(x,dim=1,keepdim=True)
        max_out,_=torch.max(x,dim=1,keepdim=True)
        x=torch.cat([avg_out,max_out],dim=1)
        out=self.conv1(x)
        out=self.sigmoid(out)
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

        self.spatial_atten=SpatialAttention(1,intermed_channels)

        # self.sa=Self_Attn(intermed_channels,out_channels,temporal_kernel_size,temporal_stride, temporal_padding)
        
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.temporal_atten=ChannelAttention(intermed_channels,1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x=self.spatial_conv(x)
        x=self.bn1(x)
        x = self.relu(x) 
        # x=self.spatial_atten(x)*x
        # print('before',self.sa(x).size())
        #x=self.sa(x)
        # print(x.shape)
        # x=self.temporal_atten(x)*x
        x=self.temporal_conv(x)
        # print('after',x.shape)
        x=self.bn2(x)
        x = self.relu(x)
        
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

        self.downsample = downsample
        padding = kernel_size // 2
        
        if self.downsample:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)
    
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.sa1=SpatialAttention(1,out_channels)
        self.ca1=ChannelAttention(out_channels,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        res=self.sa1(res)*res
        res=self.ca1(res)*res
        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

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
        # self.sa1=SpatialAttention(1,64)
        # self.ca1=ChannelAttention(64,1)
        self.conv2 = ResLayer(64, 64, 3, layer_sizes[0])
        self.conv3 = ResLayer(64, 128, 3, layer_sizes[1], downsample=True)
        self.conv4 = ResLayer(128, 256, 3, layer_sizes[2], downsample=True)
        self.conv5 = ResLayer(256, 512, 3, layer_sizes[3], downsample=True)
        # self.sa2=SpatialAttention(1,512)
        # self.ca2=ChannelAttention(512,1)
        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        # self.sa=Self_Attn(3,1)
        # self.position=PositionalEncoding(112*112,0.3)

    def forward(self, x):
        # x=self.sa(x)
        # print(x.shape)
        # if len(x.size())>4:
        #     b,c,t,w,h=x.size()
        # else:
        #     b=1
        #     c,t,w,h=x.size()
        # x_p=self.position(x.view(b,c,t,w*h))
        # # x_sa=self.sa(x)
        # # x=x_p+x_sa+x
        # x=x*x_p+x
        x = self.conv1(x)
        # print(x.shape)
        # x=self.sa1(x)*x
        # x=self.ca1(x)*x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x=self.sa2(x)*x
        # x=self.ca2(x)*x
        # print(x.shape)
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
        # self.conv=nn.Conv3d(32,128,1,2,(1,0,0))
        self.feature = FeatureLayer(layer_sizes, input_channel)
        self.fc = nn.Linear(512, num_classes)

        self.__init_weight()

    def forward(self, x):
        # x=self.conv(x.permute(0,2,1,3,4))
        # # print(x.shape)
        # x=x.permute(0,2,1,3,4)
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