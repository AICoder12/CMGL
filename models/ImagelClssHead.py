import os 
import  json 
import argparse 
import numpy as np 
import random 
import os 
import torch 
from torch import nn 
from torch.nn import functional as F 
import torchvision.transforms as transforms 


class LGTI(nn.Module):
    def __init__(self, dim_i, dim_hid, dim_out, k):
        super(LGTI, self).__init__()
        self.fuse_modules = nn.Linear(dim_i * k, dim_hid)
        self.attn = nn.MultiheadAttention(embed_dim=dim_hid, num_heads=4, batch_first=True)
        self.compress =  nn.Linear(dim_hid, 1)
        self.post_process = nn.Linear(dim_hid, dim_out)
        
    def forward(self, inps):
        x = torch.cat(inps, dim = 2)
        x = self.fuse_modules(x)
        x, _ = self.attn(x, x, x)               
        x_temp = self.compress(x)
        attention_weights = nn.Softmax(dim=1)(x_temp) 
        x = torch.sum(attention_weights * x, dim=1)
        x = self.post_process(x)
        return x



class ImagelClssHead(nn.Module):
    def __init__(self, vision_width, text_width, features_list):
        super(ImagelClssHead, self).__init__()
        self.fuse = LGTI(vision_width, vision_width // 2, text_width, k = len(features_list))
        self.class_mapping = nn.Linear(text_width, text_width)
        self.image_mapping = nn.Linear(text_width, text_width)
        self.temperature_image = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.can_k=2000
        self.scale_weights = nn.Parameter(torch.ones(len(features_list)))



    def forward(self, text_embeddings,image_features,patch_tokens,anomaly_maps_final,state):
        text_embeddings_mapping = self.class_mapping(text_embeddings)
        text_embeddings_mapping = text_embeddings_mapping / text_embeddings_mapping.norm(dim = -1, keepdim = True)
        image_embeddings_mapping = self.image_mapping(image_features + self.fuse(patch_tokens))
        image_embeddings_mapping = image_embeddings_mapping / image_embeddings_mapping.norm(dim=-1, keepdim = True)
        pro_img = self.temperature_image.exp() * text_embeddings_mapping @ image_embeddings_mapping.unsqueeze(2) 
        pro_img = pro_img.squeeze(2)
        x_avg=pro_img
        out = F.softmax(x_avg, dim=-1)  

        return out