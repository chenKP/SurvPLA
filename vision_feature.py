import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from einops import rearrange, repeat
from feature_fusion import CrossAttention,SelfAttention,FeedForward
import math

class GatePathToken(nn.Module):
    def __init__(self, in_dim, group_size):
        super(GatePathToken, self).__init__()
        #The code is being organized...

class TextInsertPath(nn.Module):
    def __init__(self, in_dim, text_embed_dim, text_in_dim, out_dim, drop_out = 0.2):
        super(TextInsertPath, self).__init__()
        #The code is being organized...

class EnhancePath(nn.Module):
    def __init__(self, path_in_dim, omics_in_dim,out_dim, attn_drop = 0.2, head = 8):
        super(EnhancePath, self).__init__()
        #The code is being organized...


class PathEncoder(nn.Module):
    def __init__(self, in_dim, text_in_dim, text_embed_dim, out_dim, group_size, drop_out = 0.2):
        super(PathEncoder, self).__init__()
        #The code is being organized...
        

        






