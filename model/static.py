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

# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W).contiguous()
#         x = self.dwconv(x)
#         x = x.flatten(2).transpose(1, 2)
#
#         return x

class DWConvPlusPW(nn.Module):
    """并联DWConv和1x1 PWConv"""
    def __init__(self, dim=768):
        super(DWConvPlusPW, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                bias=True, groups=dim)
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0,
                                bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()

        x_dw = self.dwconv(x)   # 空间建模
        x_pw = self.pwconv(x)   # 通道交互

        x = x_dw + x_pw         # 融合方式：逐元素相加（也可以 torch.cat 再做线性投影）
        x = x.flatten(2).transpose(1, 2)
        return x

class ConvolutionalGLU(nn.Module):
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

class Branch_DOWN(nn.Module):
    def __init__(self,embed_dims, mlp_ratio=4.,out_features=None, act_layer=nn.GELU, drop=0.,
                                    drop_path=0.,i=1):
        super().__init__()
        self.out_features = out_features
        self.act_layer = act_layer
        self.drop = nn.Dropout(drop)
        self.patch_size = 3
        self.stride = 1
        self.mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=embed_dims, hidden_features=self.mlp_hidden_dim,act_layer=act_layer, drop=drop)
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


class BranchDOWN(nn.Module):
    def __init__(self, embed_dims=[192, 48], mlp_ratio=4., out_features=None,
                 act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        self.out_features = out_features

        # 前2个Branch_DOWN：使用embed_dims[0]作为主要维度
        self.group1 = nn.ModuleList([
            Branch_DOWN(
                embed_dims=embed_dims[0],  # 保持输入输出维度对应
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop,
                drop_path=drop_path,
                i=1  # 索引可用于区分不同模块
            ),
            Branch_DOWN(
                embed_dims=embed_dims[0],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop,
                drop_path=drop_path,
                i=2
            )
        ])

        # 后2个Branch_DOWN：使用embed_dims[1]作为主要维度
        self.group2 = nn.ModuleList([
            Branch_DOWN(
                embed_dims=embed_dims[1],  # 主要维度改为embed_dims[1]
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop,
                drop_path=drop_path,
                i=3
            ),
            Branch_DOWN(
                embed_dims=embed_dims[1],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop,
                drop_path=drop_path,
                i=4
            )
        ])

        # 可选：如果需要从embed_dims[0]过渡到embed_dims[1]，添加维度转换卷积
        self.up = PixelShuffleUpsampleLayer(input_chans=embed_dims[0])

    def forward(self, x, mask_down):
        # 处理前2个Branch_DOWN（使用embed_dims[0]）
        for block in self.group1:
            x = block(x)

        # 维度转换：从embed_dims[0]过渡到embed_dims[1]（如果维度不同）
        x = self.up(x)
        x = x * (2 - mask_down)

        # 处理后2个Branch_DOWN（使用embed_dims[1]）
        for block in self.group2:
            x = block(x)

        x = x * (2 - mask_down)

        return x


if __name__ == '__main__':
    x = torch.randn(3, 192, 32, 32)
    model = BranchDOWN()
    y = model(x)
    print(y.shape)
