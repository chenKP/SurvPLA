import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, LayerNorm



class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim, drop_out, num_heads = 8):#(C_S, C_T, C_T, C_T,cross_head
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_sizes = in_dim1 // num_heads#
        self.out_dims = out_dim
        self.proj_q1 = nn.Sequential(nn.Linear(in_dim1, self.head_sizes * num_heads))
        self.proj_k2 = nn.Sequential(nn.Linear(in_dim2, self.head_sizes * num_heads))
        self.proj_v2 = nn.Sequential(nn.Linear(in_dim2, self.head_sizes * num_heads)) 
        self.proj_o = nn.Sequential(nn.Linear(self.head_sizes*num_heads, out_dim))
        self.attn_dropout = Dropout(0.25)
        self.proj_dropout = Dropout(drop_out)
    def forward(self, x1, x2, mask=None):

        batch_size, seq_len1, in_dim1 = x1.size()
        _, seq_len2, _ = x2.size()
        q1 = self.proj_q1(x1)
        q1 = q1.view(batch_size, seq_len1, self.num_heads, self.head_sizes).permute(0, 2, 1, 3)#8 4 4097 768
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.head_sizes).permute(0, 2, 3, 1)#8 4 768 4097
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.head_sizes).permute(0, 2, 1, 3)#8 4 4097 768
        attention = torch.matmul(q1, k2) / self.head_sizes ** 0.5#8 4 4096 4096

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(attention, dim=1)
        attention = self.attn_dropout(attention)
        output = torch.matmul(attention, v2).permute(0, 3, 2, 1).contiguous().view(batch_size, -1, self.head_sizes*self.num_heads)#  64 4 49 768
        output = self.proj_o(output)
        output = self.proj_dropout(output)
        output = output.view(batch_size, -1, self.out_dims)
        
        return output, attention


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):

    def __init__(self, in_dim, out_dim, drop_out, num_heads = 8):#(C_S, C_T, C_T, C_T,cross_head
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_sizes = in_dim // num_heads#
        self.v_dim = in_dim // num_heads#输出张量
        self.k_dim = in_dim // num_heads
        
        self.proj_q = nn.Sequential(nn.Linear(in_dim, self.head_sizes * num_heads))
        self.proj_k = nn.Sequential(nn.Linear(in_dim, self.head_sizes * num_heads))
        
        self.proj_v = nn.Sequential(nn.Linear(in_dim, self.head_sizes * num_heads)) 
        
        self.proj_o = nn.Sequential(nn.Linear(self.head_sizes*num_heads, out_dim))
        self.attn_dropout = Dropout(0.25)
        self.proj_dropout = Dropout(drop_out)
        # self.relu = nn.Sigmoid()
    def forward(self, x, mask=None):
        x1 = x2 = x
        batch_size, seq_len1, in_dim1 = x1.size()
        _, seq_len2, _ = x2.size()
        q1 = self.proj_q(x1)
        q1 = q1.view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)#8 4 4097 768
        k2 = self.proj_k(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)#8 4 768 4097
        v2 = self.proj_v(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)#8 4 4097 768
        attention = torch.matmul(q1, k2) / self.k_dim ** 0.5#8 4 4096 4096

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(attention, dim=1)
        attention = self.attn_dropout(attention)
        output = torch.matmul(attention, v2).permute(0, 3, 2, 1).contiguous().view(batch_size, -1, self.v_dim*self.num_heads)#  64 4 49 768
        output = self.proj_o(output)
        output = self.proj_dropout(output)
        output = output.view(batch_size, -1, self.v_dim*self.num_heads)
        
        return output, attention



class FusionModal(nn.Module):
    def __init__(self, path_in_dim, omics_in_dim, out_dim, attn_drop = 0.2, heads = 8):
        super(FusionModal, self).__init__()
        self.num_heads = heads
        self.in_dim = path_in_dim + omics_in_dim
        self.head_sizes = self.in_dim // self.num_heads#
        self.out_dim = out_dim
        self.linear_q = nn.Linear(self.in_dim, self.num_heads*self.head_sizes)
        self.linear_k = nn.Linear(self.in_dim, self.num_heads*self.head_sizes)
        self.linear_v = nn.Linear(self.in_dim, self.num_heads*self.head_sizes)
        self.out = nn.Linear(self.num_heads*self.head_sizes, self.out_dim)

        self.layer_norm = nn.LayerNorm(self.out_dim)
        self.attn_dropout = Dropout(attn_drop)
        self.proj_dropout = Dropout(attn_drop)
    def forward(self, input, mask = None):
        #The code is being organized...

