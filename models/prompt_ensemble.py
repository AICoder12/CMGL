import torch
import torch.nn as nn
import numpy as np 

def TextSVG(embeddings, k=4):
    embeddings = embeddings.float()
    U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)
    B = Vh[:k, :].T  
    return B

class LCBlock(nn.Module):
    def __init__(self, 
                 query_num=1, 
                 query_dim=768, 
                 token_dim=1024, 
                 num_layers=4, 
                 num_heads=8, 
                 refine_type="transformer"):
        super().__init__()

        self.queries = nn.Parameter(torch.randn(1, query_num, query_dim))
        self.token_proj = nn.ModuleList([
            nn.Linear(token_dim, query_dim) for _ in range(num_layers)
        ])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fuse = nn.Linear(num_layers * query_dim, query_dim)
        if refine_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=query_dim, nhead=num_heads, batch_first=True)
            self.refine = nn.TransformerEncoder(encoder_layer, num_layers=1)  
        elif refine_type == "gru":
            self.refine = nn.GRU(input_size=query_dim, hidden_size=query_dim, batch_first=True)
        else:
            raise ValueError("refine_type must be 'transformer' or 'gru'")

        self.refine_type = refine_type

    def forward(self, patch_tokens):
        B = patch_tokens[0].size(0)
        Q = self.queries.expand(B, -1, -1)  
        outs = []
        for i, tokens in enumerate(patch_tokens):
            tokens_proj = self.token_proj[i](tokens)  
            out, _ = self.cross_attn[i](Q, tokens_proj, tokens_proj)  
            outs.append(out)
        fused = torch.cat(outs, dim=-1)
        fused = self.fuse(fused)  
        if self.refine_type == "transformer":
            refined = self.refine(fused)  
        else:  
            refined, _ = self.refine(fused)  

        return refined


class Prompt_Ensemble():
    def __init__(self, cla_len, tokenizer):
        super().__init__()
      
        self.special_token = '<|class|>'  

        self.prompt_templates_normal_general = [
            "perfect", "flawless", "intact", "undamaged", "unbroken", "whole", "complete", "sound", "normal", "healthy",
            "good", "reliable", "safe", "standard", "qualified", "stable", "strong", "solid", "durable", "consistent",
            "clean", "smooth", "clear", "uniform", "even", "regular", "neat", "unmarked", "spotless", "unblemished",
            "functional", "working", "operational", "proper", "correct", "balanced", "harmonious"
        ]

        self.prompt_templates_abnormal_general = [
            "abnormal", "defective", "flawed", "imperfect", "faulty", "problematic", "unqualified", "unacceptable", "wrong",
            "damaged", "broken", "cracked", "fractured", "split", "shattered", "ruptured", "weakened", "collapsed",
            "irregular", "distorted", "deformed", "misaligned", "unbalanced", "unstable", "inconsistent",
            "dirty", "stained", "marked", "scratched", "blemished", "rough", "uneven", "worn", "dented", "spotted",
            "failed", "unstable-operation", "malfunctioning", "nonfunctional", "unreliable", "uncontrolled", "inactive"
        ]

        self.prompt_templates_abnormal = ["a photo of a damaged <|class|>"]
        self.prompt_templates_normal = ["a photo of a good <|class|>"]

        self.tokenizer =  tokenizer

    def build_state_pool(self, model,device):
        
        prompted_sentence_normal_general = self.tokenizer(self.prompt_templates_normal_general).to(device)   
        self.normal_embeddings_general = model.encode_text_origanl(prompted_sentence_normal_general)   
    
        prompted_sentence_abnormal_general = self.tokenizer(self.prompt_templates_abnormal_general).to(device)   
        self.abnormal_embeddings_general = model.encode_text_origanl(prompted_sentence_abnormal_general)

        k=4
        self.B_normal_general   = TextSVG(self.normal_embeddings_general, k) 
        self.B_abnormal_general = TextSVG(self.abnormal_embeddings_general, k) 

    def forward_ensemble(self, model,model_tf,image_features,patch_tokens, vison_feature,cls_name, device, prompt_id = 0):

        context_word = model_tf(patch_tokens) 
        normal_state_word = ["context"]*vison_feature.shape[0]
        abnormal_state_word = ["context"]*vison_feature.shape[0]
        prompted_sentence_normal = self.tokenizer(normal_state_word).to(device)  
        prompted_sentence_abnormal = self.tokenizer(abnormal_state_word).to(device) 
        normal_embeddings,normal_state = model.encode_text(context_word,prompted_sentence_normal,vison_feature,self.B_normal_general.unsqueeze(0).expand(vison_feature.shape[0], -1, -1),None) 
        abnormal_embeddings,abnormal_state = model.encode_text(context_word,prompted_sentence_abnormal, vison_feature,None,self.B_abnormal_general.unsqueeze(0).expand(vison_feature.shape[0], -1, -1) )
        text_prompts = torch.cat([normal_embeddings, abnormal_embeddings], dim =1)

        return text_prompts,normal_state,abnormal_state



        