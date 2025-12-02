import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from model.up import PixelShuffleUpsampleLayer

class DWConvPlusPW(nn.Module):
    def __init__(self, dim=768):
        super(DWConvPlusPW, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                bias=True, groups=dim)
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0,
                                bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()

        x_dw = self.dwconv(x)   
        x_pw = self.pwconv(x)   

        x = x_dw + x_pw         
        x = x.flatten(2).transpose(1, 2)
        return x

class SC_GLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.conv = DWConvPlusPW(hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        # x = self.act(self.dwconv(x, H, W)) * v
        x = self.act(self.conv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class SPBlock(nn.Module):
    def __init__(self,embed_dims, mlp_ratio=4.,out_features=None, act_layer=nn.GELU, drop=0.,
                                    drop_path=0.,i=1):
        super().__init__()
        self.out_features = out_features
        self.act_layer = act_layer
        self.drop = nn.Dropout(drop)
        self.patch_size = 3
        self.stride = 1
        self.mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = SC_GLU(in_features=embed_dims, hidden_features=self.mlp_hidden_dim,act_layer=act_layer, drop=drop)
        self.patch_embed = OverlapPatchEmbed(patch_size=self.patch_size,stride=self.stride,
                                            in_chans=embed_dims,
                                            embed_dim=embed_dims)

    def forward(self, x):
        T = x.shape[0]
        x_res = x
        x, H, W = self.patch_embed(x)
        x = self.mlp(x, H, W)
        x = x.reshape(T, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = x + x_res

        return x


class Down_Branch(nn.Module):
    def __init__(self, embed_dims=[192, 48], mlp_ratio=4., out_features=None,
                 act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        self.out_features = out_features

        self.group1 = nn.ModuleList([
            SPBlock(
                embed_dims=embed_dims[0],  
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop,
                drop_path=drop_path,
                i=1  
            ),
            SPBlock(
                embed_dims=embed_dims[0],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop,
                drop_path=drop_path,
                i=2
            )
        ])

        self.group2 = nn.ModuleList([
            SPBlock(
                embed_dims=embed_dims[1],  
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop,
                drop_path=drop_path,
                i=3
            ),
            SPBlock(
                embed_dims=embed_dims[1],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop,
                drop_path=drop_path,
                i=4
            )
        ])

        self.up = PixelShuffleUpsampleLayer(input_chans=embed_dims[0])

    def forward(self, x, mask_down):
        for block in self.group1:
            x = block(x)

        x = self.up(x)
        x = x * (2 - mask_down)

        for block in self.group2:
            x = block(x)

        x = x * (2 - mask_down)

        return x



