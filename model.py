import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViT
from betaVAE import BetaVAE
from gene_feature import OmicsEncoder,EnhanceGene,OmicsEncoder_no_Text
from vision_feature import PathEncoder,EnhancePath, PathEncoder_no_Text
from feature_fusion import FusionModal
import math
def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0
    l1_reg =  l1_reg_all(model.enhence_module) + l1_reg_all(model.gene_embed) + l1_reg_all(model.fusion_module)
    return l1_reg


class SurPLA_FusionVAENet(nn.Module):
    def __init__(self, gene_dim, vision_dim, text_in_dim ,text_embed_dim = 625, head = 8, classes = 4):
        super().__init__()
        
    def forward(self, vision, gene, text):

        #The code is being organized...
