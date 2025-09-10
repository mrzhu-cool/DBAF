import torch
import torch.nn as nn
from models.vit import ViT

l2_loss = torch.nn.MSELoss(reduction='mean')

""" 
Adapt from One Shot Face Swapping on Megapixels (https://arxiv.org/abs/2105.04932) official repository:
https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/blob/main/inference/inference.py
"""


class DisentanglementId(nn.Module):
    def __init__(self):
        super().__init__()
        layer_list=[]
        # Inference Parameters

        self.linear_w = nn.Linear(512,512)
        
        self.pos_embedding=nn.Parameter(torch.randn(14 , 1, 512))#第二维系数为1就每个batch加一遍

        for i in range(3):
            layer_list.append(ViT())


        self.layers = nn.Sequential(*layer_list)

        self.split_list=[4,4,6]

    def split_latent(self,w):
        w_split=torch.split(w,self.split_list,dim=0)#把0维度分块
        return w_split

    
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


    
