import torch
import torch.nn as nn

from einops import rearrange

class TokenLearner(nn.Module):
    def __init__(self, out_token, emb, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.mlps = [nn.Sequential(
            nn.Linear(emb, emb),
            nn.GELU(),
            nn.Linear(emb, 1)) for _ in range(out_token)]
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):    
        """
        Args:
            x (shape [b x (Ph x Pw) x emb]): input tokens
        Returns:
            [b x self.out_token x emb]: out_tokens
        """
        new_tokens = []
        for mlp in self.mlps:
            weighted_x = (x * mlp(x)).permute(0, 2, 1) # b x emb x (ph x pw)
            weighted_x = rearrange(weighted_x, "b emb (ph pw) -> b emb ph pw", ph=self.patch_size)
            
            token = self.gap(weighted_x).flatten(2).transpose(-1, -2) # b x 1 x emb
            new_tokens.append(token)
        
        return torch.cat(new_tokens, dim=1)

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
            y (shape [b x S x emb]): token tensor from a Transformer layer
            x (shape [b x N x emb]): residual input to the previous TokenLearner
        Returns:
        """
        shortcut = x
        y = torch.sigmoid(self.mlp(y.transpose(-1, -2))).transpose(-1, -2) # b x S x emb
        x = self.beta(x) # b x N x S
        
        output = x @ y + shortcut
        return output
    
if __name__ == "__main__":
    patch_size = 8
    dummy = torch.randn(1, patch_size*patch_size, 256)
    
    tok_learner = TokenLearner(8, 256, patch_size)
    tok_fuser = TokenFuser(8, 256)
    
    shortcut = dummy
    
    dummy = tok_learner(dummy)
    print(dummy.shape)
    dummy = tok_fuser(dummy, shortcut)
    print(dummy.shape)