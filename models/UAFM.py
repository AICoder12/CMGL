import torch 
from torch import Tensor, nn
from typing import Tuple, Type
import numpy as np
from torch.nn import functional as F 
import math


class AFBlock(nn.Module):
    def __init__(self, alpha=2.0, beta=1.0, eps=1e-6, clamp_gate=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.eps = float(eps)
        self.clamp_gate = clamp_gate  

    def forward(self, A1, U, A2):
        if A1.dim() != 3 or A1.shape[-1] != 2:
            raise ValueError("A1 must be shape [B, N, 2]")
        if A2.shape != A1.shape or U.shape != A1.shape:
            raise ValueError("A1, A2, U must have same shape [B, N, 2]")
        B, N, C = A1.shape 
        u_mean = U.mean(dim=1, keepdim=True)  
        u_std  = U.std(dim=1, keepdim=True) + self.eps
        u_norm = (U - u_mean) / (u_std+0.000001)         
        pre = self.beta * (A2 - A1)*  u_norm  
        W = torch.tanh(pre) ** 2
        A_fused = (1-W) * A1 + W * A2

        return A_fused


class UAFM(nn.Module):
    def __init__(self, dim_v,dim_t, dim_out, num_heads = 8, qkv_bias = False, qk_scale = None, attn_drop = 0., proj_drop = 0.):
        super().__init__()
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.layers = 3

        self.q_projs = nn.ModuleList([nn.Conv1d(dim_t, dim_out, 1) for _ in range(self.layers)])
        self.k_projs = nn.ModuleList([nn.Conv1d(dim_v, dim_out, 1) for _ in range(self.layers)])
        self.v_projs = nn.ModuleList([nn.Conv1d(dim_v, dim_out, 1) for _ in range(self.layers)])
        self.proj_post = nn.Conv1d(dim_out, dim_out, 1)

        self.temp_layers = nn.ParameterList([nn.Parameter(torch.ones([]) * np.log(1/0.07)) for _ in range(self.layers)])
        self.beta_t = 1.0
        self.beta_s = 1.0

        self.mha_fusion = AFBlock(alpha=0, beta=0)


    def forward(self, F_t, F_s,anomaly_prior=None,epoch=10):
        B, Nt, Ct = F_t.shape
        B2, Ns, Cv = F_s.shape
        assert B == B2

        anomaly_maps = []
        for l in range(self.layers):
            q = self.q_projs[l](F_t.permute(0,2,1)).permute(0,2,1).reshape(B, Nt, self.num_heads, self.dim_out//self.num_heads)
            k = self.k_projs[l](F_s.permute(0,2,1)).permute(0,2,1).reshape(B, Ns, self.num_heads, self.dim_out//self.num_heads)
            v = self.v_projs[l](F_s.permute(0,2,1)).permute(0,2,1).reshape(B, Ns, self.num_heads, self.dim_out//self.num_heads)

            attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.beta_t
            attn = attn.softmax(dim=-1)
            F_t_a = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, Nt, self.dim_out)
            F_t_a = self.proj_post(F_t_a.permute(0,2,1)).permute(0,2,1)
            F_t_a = F_t_a / (F_t_a.norm(dim=-1, keepdim=True)+ 1e-6)
            anomaly_layer = torch.exp(self.temp_layers[l]) * (F_s @ F_t_a.permute(0,2,1))
            anomaly_maps.append(anomaly_layer)

        anomaly_stack = torch.stack(anomaly_maps, dim=0) 
        A2  = anomaly_stack.mean(dim=0)
        U = anomaly_stack.var(dim=0)
        if anomaly_prior is None:
            anomaly_prior = A2 
        A1 = anomaly_prior

        if epoch>9:
            A1=A1.detach()
            A2=A2.detach()
            U=U.detach()
            anomaly_fused = self.mha_fusion(A2, U, A1)
        else:
            anomaly_fused=A2  

        # anomaly_fused = self.mha_fusion(A2, U, A1)

        return anomaly_fused
