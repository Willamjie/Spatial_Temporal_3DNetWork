import torch.nn as nn
import torch
import math
from torch.autograd import Variable
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=32):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # print(position)
        # print(position.shape)
        div_term = torch.exp(torch.arange(0., d_model, 2)*-(math.log(10000.0) / d_model)).float()
        # print(pe)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        b,c,t,_=x.size()
        x= x + Variable(self.pe[:,:x.size()[2]],requires_grad=False)
        return self.dropout(x).view(b,c,t,112,112)
PE=PositionalEncoding(112*112,0.5)
print(PE(torch.randn(3,3,32,112*112)))
# import time
# #自注意力机制
# class Self_Attn(nn.Module):
#     """ Self attention Layer"""
#     def __init__(self,in_dim,kernel_size):
#         super(Self_Attn,self).__init__()
#         self.chanel_in = in_dim
#         self.query_conv = nn.Conv3d(in_channels = in_dim , out_channels =in_dim , kernel_size= kernel_size,bias=False)
#         self.key_conv = nn.Conv3d(in_channels = in_dim, out_channels = in_dim , kernel_size= kernel_size)
#         self.value_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= kernel_size)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax  = nn.Softmax(dim=1) #
#     def forward(self,x):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature 
#                 attention: B X N X N (N is Width*Height)
#         """

#         batch_size,C,t,width ,height = x.size()
#         # print(self.query_conv(x).shape)
#         proj_query  = self.query_conv(x).view(C,t,width*height).permute(0,2,1) # B X CX(N)
#         # print(proj_query.size())
#         proj_key =  self.key_conv(x).view(C,t,width*height)# B X C x (*W*H)
#         # print(proj_key.size())
#         energy =  torch.matmul(proj_query,proj_key) # transpose check
#         attention = self.softmax(energy) # BX (N) X (N) 
#         proj_value = self.value_conv(x).view(1,-1,t,width,height) # B X C X N
#         out = torch.matmul(proj_value.view(1,C,t,width*height),attention.permute(0,2,1))
#         out = out.view(C,t,width,height)
#         out = self.gamma*out + x#.view(C,t,width,height)
#         return out
# # sa=Self_Attn(3,1)
# # inputs = torch.rand(1,3, 32, 112,112)
# # start=time.time()
# # print(sa(inputs).shape)
# # end=time.time()
# # print(end-start)
