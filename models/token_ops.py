import torch
import torch.nn as nn

from einops import rearrange

from .position_encoding import build_position_encoding

class TokenLearner(nn.Module):
    def __init__(self, out_token, emb):
        super().__init__()
        self.mlps = [nn.Sequential(
            nn.Linear(emb, emb),
            nn.GELU(),
            nn.Linear(emb, 1)).cuda() for _ in range(out_token)]
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x, h, w):    
        """
        Args:
            x (shape [(Ph x Pw) x B x emb]): input tokens
        Returns:
            [elf.out_token x B x emb]: out_tokens
        """
        new_tokens = []
        for mlp in self.mlps:
            # print(f"x device: {x.device}")
            # print(f"weight device: {mlp[0].weight.device}")
            # exit()
            weight_map = mlp(x)
            weighted_x = (x * weight_map).permute(1, 2, 0) # b x emb x (ph x pw)
            weighted_x = rearrange(weighted_x, "b emb (ph pw) -> b emb ph pw", ph=h, pw=w)
            
            token = self.gap(weighted_x).flatten(2).permute(2, 0, 1) # 1 x B x emb
            new_tokens.append(token)
        # print(f"before: {x.shape}")
        # print(f"after: {torch.cat(new_tokens, dim=0).shape}")
        # exit()
        return torch.cat(new_tokens, dim=0) # S x B x emb

class TokenFuser(nn.Module):
    def __init__(self, in_token, emb):
        super().__init__()
        self.mlp = nn.Linear(in_token, in_token)
        self.beta = nn.Linear(emb, in_token)
        
    def forward(self, y, x):
        """
        1) fuse information across the tokens
        2) remap the representation back to its original spatial resolution
        Args:
            y (shape [S x B x emb]): token tensor from a Transformer layer
            x (shape [N x B x emb]): residual input to the previous TokenLearner
        Returns:
        """
        shortcut = x
        y = torch.sigmoid(self.mlp(y.transpose(0, 2))).permute(1, 2, 0) # B x S x emb
        x = self.beta(x).transpose(0, 1) # b x N x S
        
        output = x @ y + shortcut.transpose(0, 1)
        return output.transpose(0, 1)
    
if __name__ == "__main__":
    patch_size = 8
    dummy = torch.randn(patch_size, patch_size, 1, 256)
    H, W, B, emb = dummy.shape
    dummy = dummy.view(H*W, B, -1)
    
    tok_learner = TokenLearner(8, 256)
    tok_fuser = TokenFuser(8, 256)
    
    shortcut = dummy
    
    dummy = tok_learner(dummy, H, W)
    print(dummy.shape)
    dummy = tok_fuser(dummy, shortcut)
    print(dummy.shape)