import argparse
import os
import cv2
import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as tF
from torchvision.utils import save_image
from models.vit import ViT
from losses.id_loss import IDLoss


l2_loss = torch.nn.MSELoss(reduction='mean')

""" 
Adapt from One Shot Face Swapping on Megapixels (https://arxiv.org/abs/2105.04932) official repository:
https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/blob/main/inference/inference.py
"""


class DisentanglementId(nn.Module):
    def __init__(self, w_dim=512, stylegan_size=256):
        super().__init__()
        import math
        n_layers = (int(math.log2(stylegan_size)) - 1) * 2  # 256→14
        self.n_layers = n_layers
        self.linear_w = nn.Linear(w_dim, w_dim)
        self.pos_embedding = nn.Parameter(torch.randn(n_layers, 1, w_dim))
        self.layers = nn.Sequential(*[ViT() for _ in range(3)])
        self.split_list = [4, 4, n_layers - 8]  # 14→[4,4,6]

    def split_latent(self, w):
        # 期望 w 的层维度与 n_layers 匹配
        if w.shape[0] != self.n_layers:
            raise ValueError(f"split_latent: w.shape[0]={w.shape[0]} 与 n_layers={self.n_layers} 不一致")
        return torch.split(w, self.split_list, dim=0)

    # def split_latent(self,w):
    #     w_split=torch.split(w,self.split_list,dim=0)#把0维度分块
    #     return w_split
       
    def forward(self,image,e4e,device):

        w_out_list=[]

        w_pos_split =self.split_latent(self.pos_embedding)
        
        
        _,origin_codes,c3,c2,c1 = e4e(image,device,return_latents=True)
   
        codes_temp = origin_codes.permute(1,0,2)#维度换位为了分块处理

        codes_temp = self.linear_w(codes_temp)
  
        codes_split = self.split_latent(codes_temp)

        for i in range(3):

            w_split_temp = codes_split[i] + w_pos_split[i]
          
            w_out_temp=self.layers[i](w_split_temp)
            w_out_list.append(w_out_temp)
        
        w_out = torch.cat(w_out_list,dim=0)
        w_out = w_out.permute(1,0,2)

        
        return origin_codes,w_out,c3,c2,c1


    
