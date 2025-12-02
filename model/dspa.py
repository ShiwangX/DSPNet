from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model.shiftvit_T import ShiftVitCount
from model.up import PixelShuffleUpsampleLayer
from model.dynamic import FovealAttentionBranch
from model.static import BranchDOWN
from model.motion_mask import MotionMask_UP,MotionMask_DOWN




class FusionNet(nn.Module):
    def __init__(self,threshold=0.5,img_size=512,is_extrapolation=False,is_train=True) :
        super().__init__()
        self.backbone = ShiftVitCount(is_train=is_train)
        self.upsample = PixelShuffleUpsampleLayer(768)
        self.branch_up = FovealAttentionBranch(img_size=img_size,is_extrapolation=is_extrapolation)
        self.branch_down = BranchDOWN()
        self.mask_up = MotionMask_UP()
        self.mask_down = MotionMask_DOWN(threshold=threshold)
        self.decoder = nn.Conv2d(48,1,1)
        # self.decoder = nn.Conv2d(96, 1, 1)





    def forward(self, x, diff):
        x = self.backbone(x)
        x = self.upsample(x)

        mask_up = self.mask_up(diff)
        mask,mask_down = self.mask_down(diff)

        x_up = self.branch_up(x, mask_up)
        x_down = self.branch_down(x, mask_down)
        # print("x_up shape:", x_up.shape)  # 例如：(B, C, 88, W)
        # print("mask_up shape:", mask_up.shape)

        x = x_up + x_down
        # x = torch.cat([x_up, x_down], dim=1)
        x = self.decoder(x)


        return x




if __name__ == '__main__':
    model = FusionNet(is_extrapolation=True,is_train=False).cuda()

    # # 计算总参数量（可训练参数）
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # # 转换为“百万（M）”单位（除以 1e6），保留 3 位小数
    # total_params_m = total_params / 1e6
    #
    # print(f"总参数量：{total_params}（个）")
    # print(f"总参数量：{total_params_m:.3f} M")  # 表格中常用的单位格式
    x = torch.randn(3, 3, 640, 360).to(torch.device('cuda:0'))
    diff = torch.randn(3, 3, 640, 360).to(torch.device('cuda:0'))
    # print(model)
    out = model(x,diff)
    print(out.shape)