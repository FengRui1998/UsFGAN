'''FR——STRB
conv2d + SWIN block + skip connection +conv2d
'''

import torch
import torch.nn as nn
from model.network_swinir import RSTB
import math

def change_C2T(x):
    B, C, H, W = x.shape
    x = x.view(B, H * W, C)
    return x

class STRB_FR(nn.Module):
    def __init__(self, dim, input_resolution, depth=2, num_heads=2, window_size=7):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.strb = RSTB(dim=self.dim, input_resolution=self.input_resolution, depth=self.depth, num_heads=self.num_heads, window_size=self.window_size,
            mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
            drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
            img_size=224, patch_size=4, resi_connection='1conv')
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x):
        x = change_C2T(x)
        x = self.strb(x,self.input_resolution)
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        assert L == H * W, "input feature has wrong size"
        x = x.contiguous().view(B, C, H, W)
        return x








