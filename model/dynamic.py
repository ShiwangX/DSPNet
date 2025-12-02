import torch
import torch.nn as nn
import torch.nn.functional as F
# import swattention
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model.up import PixelShuffleUpsampleLayer

CUDA_NUM_THREADS = 128


def get_seqlen_and_mask(input_resolution, window_size, device):
    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]], device=device), window_size,
                        dilation=1, padding=(window_size // 2, window_size // 2), stride=1)
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask

@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None,
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table

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

class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=4, is_extrapolation=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W
            self.trained_pool_H, self.trained_pool_W = input_resolution[0] // self.sr_ratio, input_resolution[
                1] // self.sr_ratio
            self.trained_pool_len = self.trained_pool_H * self.trained_pool_W

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative_bias_local:
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        B, N, C = x.shape
        pool_H, pool_W = H // self.sr_ratio, W // self.sr_ratio
        pool_len = pool_H * pool_W

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale

        # Generate unfolded keys and values and l2-normalize them
        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
        # Compute local similarity
        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) \
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(padding_mask, float('-inf'))

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(self.act(self.sr(x_)), (pool_H, pool_W)).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        if self.is_extrapolation:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                        relative_pos_index.view(-1)].view(-1, N, pool_len)
        else:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                        relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_pool_len)

            # bilinear interpolation:
            pool_bias = pool_bias.reshape(-1, self.trained_len, self.trained_pool_H, self.trained_pool_W)
            pool_bias = F.interpolate(pool_bias, (pool_H, pool_W), mode='bilinear')
            pool_bias = pool_bias.reshape(-1, self.trained_len, pool_len).transpose(-1, -2).reshape(-1, pool_len,
                                                                                                    self.trained_H,
                                                                                                    self.trained_W)
            pool_bias = F.interpolate(pool_bias, (H, W), mode='bilinear').reshape(-1, pool_len, N).transpose(-1, -2)

        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(
            -2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Foveal3DBlock(nn.Module):
    def __init__(self, in_channels=64, reduce_ratio=8, dilations=[1, 2, 3]):
        super().__init__()
        self.reduce_ratio = reduce_ratio
        self.mid_channels = in_channels // 2  # 中间通道数（in_channel//2）
        self.out_channels = in_channels // reduce_ratio  # 分支输出通道数
        self.num_branches = len(dilations)  # 固定为3个分支

        # 三分支膨胀深度可分离模块
        class TripleBranch3D(nn.Module):
            def __init__(self, mid_channels, out_channels, dilations):
                super().__init__()
                # 生成三个分支（不同膨胀率）
                self.branches = nn.ModuleList()
                for d in dilations:
                    self.branches.append(nn.Sequential(
                        # 时间深度卷积：(3,1,1) + 不同膨胀率，逐通道计算
                        nn.Conv3d(
                            mid_channels, mid_channels,
                            kernel_size=(3, 1, 1),
                            dilation=(d, 1, 1),  # 仅时间维度膨胀
                            padding='same',
                            groups=mid_channels  # 深度卷积（参数降至1/mid_channels）
                        ),
                        nn.GELU(),
                        # 空间深度卷积：(1,3,3) 补充空间感受野
                        nn.Conv3d(
                            mid_channels, mid_channels,
                            kernel_size=(1, 3, 3),
                            padding='same',
                            groups=mid_channels
                        ),
                        nn.GELU(),
                        # 点卷积：融合通道并映射到目标维度
                        nn.Conv3d(
                            mid_channels, out_channels,
                            kernel_size=1, padding=0
                        ),
                        nn.GELU()
                    ))

                # 融合三分支特征（恢复原通道数）
                self.fusion = nn.Sequential(
                    nn.Conv3d(
                        out_channels * 3,  # 3个分支拼接
                        in_channels,
                        kernel_size=1,
                        padding=0
                    ),
                    nn.GELU()
                )

            def forward(self, x):
                # 并行处理三个分支
                branch_outs = [branch(x) for branch in self.branches]
                # 通道维度拼接（B, C*3, T, H, W）
                fused = torch.cat(branch_outs, dim=1)
                return self.fusion(fused)

        # 1x1卷积降维：从in_channels→mid_channels（in_channel//2）
        self.reduce = nn.Conv3d(
            in_channels, self.mid_channels,
            kernel_size=1, padding=0, bias=False
        )

        # 实例化两个三分支模块（保持双块残差结构）
        self.block1 = TripleBranch3D(
            self.mid_channels, self.out_channels, dilations
        )
        self.block2 = TripleBranch3D(
            self.mid_channels, self.out_channels, dilations
        )

        # 残差连接通道匹配（如需）
        if in_channels != self.mid_channels:
            self.residual_adjust = nn.Conv3d(
                in_channels, self.mid_channels, kernel_size=1, bias=False
            )
        else:
            self.residual_adjust = nn.Identity()

    def forward(self, x):  # x: (B, C, T, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(0)  # 处理单样本维度

        # 维度调整（保持原逻辑）
        x = x.permute(0, 2, 1, 3, 4)
        x_res = x  # 原始残差备份

        # 第一次处理：降维→三分支→融合→残差
        x_reduced = self.reduce(x)  # 降维到mid_channels
        block1_out = self.block1(x_reduced)
        out1 = block1_out + x_res  # 与原始x残差

        # 第二次处理：复用降维层→三分支→融合→残差
        out1_reduced = self.reduce(out1)
        block2_out = self.block2(out1_reduced)
        final_out = block2_out + block1_out  # 与第一次输出残差

        return final_out.squeeze(0).permute(1, 0, 2, 3)


class Foveal3DBlock_old(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        # 定义多尺度3D卷积模块（内部包含3个分支和融合层）
        class MultiScale3D(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.branch3x3 = nn.Sequential(
                    nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.GELU()
                )
                self.branch5x5 = nn.Sequential(
                    nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2),
                    nn.GELU()
                )
                self.branch7x7 = nn.Sequential(
                    nn.Conv3d(in_channels, in_channels, kernel_size=7, padding=3),
                    nn.GELU()
                )
                self.fusion = nn.Sequential(
                    nn.Conv3d(in_channels * 3, in_channels, kernel_size=1),
                    nn.GELU()
                )

            def forward(self, x):
                # 多尺度特征提取
                out3 = self.branch3x3(x)
                out5 = self.branch5x5(x)
                out7 = self.branch7x7(x)
                # 融合分支
                fused = torch.cat([out3, out5, out7], dim=1)
                return self.fusion(fused)  # 输出形状：(B, C, T, H, W)

        # 实例化两次多尺度3D卷积模块
        self.block1 = MultiScale3D(in_channels)  # 第一次多尺度卷积
        self.block2 = MultiScale3D(in_channels)  # 第二次多尺度卷积

    def forward(self, x):  # x: (B, C, T, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(0)

        x = x.permute( 0, 2, 1, 3, 4)
        # 第一步：第一次多尺度卷积 + 原始x残差
        block1_out = self.block1(x)  # 第一次多尺度卷积输出
        out1 = block1_out + x  # 中间结果1：第一次输出 + 原始x

        # 第二步：第二次多尺度卷积 + 第一次多尺度输出残差
        block2_out = self.block2(out1)  # 第二次多尺度卷积（输入为中间结果1）
        final_out = block2_out + block1_out  # 最终结果：第二次输出 + 第一次输出

        return final_out.squeeze(0).permute(1 ,0 ,2 ,3)


class FovealAttentionBlock(nn.Module):
    def __init__(self,dim,input_resolution,img_size=512, embed_dims=[192,192],window_size=[3, 3, 3, None],sr_ratio=4,
                 is_extrapolation=True, pretrain_size=None,patch_size=3,):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.window_size = window_size
        i=1
        self.pretrain_size = pretrain_size or img_size
        self.attn = AggregatedAttention(dim=dim, input_resolution=input_resolution,is_extrapolation=is_extrapolation)
        self.norm = nn.LayerNorm(dim)
        self.is_extrapolation = is_extrapolation
        if not self.is_extrapolation:
            relative_pos_index, relative_coords_table = get_relative_position_cpb(
                query_size=to_2tuple(img_size // (2 ** (i + 2))),
                key_size=to_2tuple(img_size // ((2 ** (i + 2)) * self.sr_ratio)),
                pretrain_size=to_2tuple(self.pretrain_size // (2 ** (i + 2))))

            self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
            self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

        self.patch_embed = OverlapPatchEmbed(patch_size=patch_size,
                                        stride=1,
                                        in_chans=dim,
                                        embed_dim=dim)


    def forward(self, x):
        x_res = x
        B = x.shape[0]
        i = 1
        x, H, W = self.patch_embed(x)
        if self.is_extrapolation:
            relative_pos_index, relative_coords_table = get_relative_position_cpb(query_size=(H, W),
                                                                                  key_size=(
                                                                                      H // self.sr_ratio,
                                                                                      W // self.sr_ratio),
                                                                                  pretrain_size=to_2tuple(
                                                                                      self.pretrain_size // (
                                                                                              2 ** (i + 2))),
                                                                                  device=x.device)
        else:
            relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
            relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")
        with torch.no_grad():
            local_seq_length, padding_mask = get_seqlen_and_mask((H, W), self.window_size[i], device=x.device)
            seq_length_scale = torch.log(local_seq_length + (H // self.sr_ratio) * (W // self.sr_ratio))
        x_attn = self.attn(x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask)
        x_attn = self.norm(x_attn)
        x_attn = x_attn.transpose(1, 2).reshape(B, -1, H, W)
        return x_attn


class FovealAttentionBranch(nn.Module):
    def __init__(self, dims=[192, 48], img_size=512, is_extrapolation=True):
        super().__init__()
        # 计算前两组和后两组的不同输入分辨率
        # 前两组：input_res = img_size // (2 **(2 + 2)) = img_size // 16
        input_res_group1 = to_2tuple(img_size // (2 ** 4))
        # 后两组：input_res = img_size // (2 **(1 + 2)) = img_size // 8
        input_res_group2 = to_2tuple(img_size // (2 ** 3))

        self.up = PixelShuffleUpsampleLayer(input_chans=dims[0])

        # 分为两组，每组2个block，使用不同的input_resolution
        self.groups = nn.ModuleList([
            # 第一组（前2个block）：使用input_res_group1
            nn.ModuleList([
                nn.ModuleDict({
                    "conv3d": Foveal3DBlock(in_channels=dims[0]),
                    "attn": FovealAttentionBlock(
                        dim=dims[0],
                        input_resolution=input_res_group1,  # 前两组的分辨率
                        is_extrapolation=is_extrapolation
                    )
                }),
                nn.ModuleDict({
                    "conv3d": Foveal3DBlock(in_channels=dims[0]),
                    "attn": FovealAttentionBlock(
                        dim=dims[0],
                        input_resolution=input_res_group1,  # 前两组的分辨率
                        is_extrapolation=is_extrapolation
                    )
                })
            ]),
            # 第二组（后2个block）：使用input_res_group2
            nn.ModuleList([
                nn.ModuleDict({
                    "conv3d": Foveal3DBlock(in_channels=dims[1]),
                    "attn": FovealAttentionBlock(
                        dim=dims[1],
                        input_resolution=input_res_group2,  # 后两组的分辨率
                        is_extrapolation=is_extrapolation
                    )
                }),
                nn.ModuleDict({
                    "conv3d": Foveal3DBlock(in_channels=dims[1]),
                    "attn": FovealAttentionBlock(
                        dim=dims[1],
                        input_resolution=input_res_group2,  # 后两组的分辨率
                        is_extrapolation=is_extrapolation
                    )
                })
            ])
        ])

    def forward(self, x, mask_up):
        # 输入x的形状假设为：(B, C, T, H, W)
        T, C, H, W = x.shape  # 保持原维度解析逻辑

        # 处理第一组（前2个block，使用input_res_group1）
        for block in self.groups[0]:
            x_res = x
            x = block["conv3d"](x)
            x = block["attn"](x)
            x = x + x_res

        x = self.up(x)
        x = x * (mask_up + 1)

        # 处理第二组（后2个block，使用input_res_group2）
        for block in self.groups[1]:
            x_res = x
            x = block["conv3d"](x)
            x = block["attn"](x)
            x = x + x_res

        x = x * (mask_up + 1)

        return x  # 输出形状：(B, dim, H, W)


if __name__ == '__main__':
    x = torch.randn(4, 192, 32, 32).to(torch.device('cuda'))
    model = FovealAttentionBranch().to(torch.device('cuda'))
    y = model(x)
    print(y.shape)
