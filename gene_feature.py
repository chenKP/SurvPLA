
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_fusion import CrossAttention,SelfAttention,FeedForward

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

class SNNOmics(nn.Module):
    def __init__(self, omic_input_dim: int, omics_out_dim: int,model_size_omic: str='small', embed_dim = 128):
        super(SNNOmics, self).__init__()
        self.size_dict_omic = {'small': [256, omics_out_dim], 'big': [1024, 1024, 1024, omics_out_dim]}
        self.embed_dim = embed_dim
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        init_max_weights(self)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.repeat(1, self.embed_dim, 1) 
        h_omic = self.fc_omic(x)
        return h_omic


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

class TextInsertGene(nn.Module):

    #The code is being organized...

class EnhanceGene(nn.Module):
    
    #The code is being organized...

class OmicsEncoder(nn.Module):
    def __init__(self,input_dim, text_in_dim, text_embed_dim, out_dim):
    
    #The code is being organized...