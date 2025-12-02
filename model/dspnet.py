from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model.shiftvit_T import ShiftVitCount
from model.up import PixelShuffleUpsampleLayer
from model.dynamic import Up_Branch
from model.static import Down_Branch
from model.motion_mask import MotionMask_UP,MotionMask_DOWN




class FusionNet(nn.Module):
    def __init__(self,threshold=0.5,img_size=512,is_extrapolation=False,is_train=True) :
        super().__init__()
        self.backbone = ShiftVitCount(is_train=is_train)
        self.upsample = PixelShuffleUpsampleLayer(768)
        self.branch_up = Up_Branch(img_size=img_size,is_extrapolation=is_extrapolation)
        self.branch_down = Down_Branch()
        self.mask_up = MotionMask_UP()
        self.mask_down = MotionMask_DOWN(threshold=threshold)
        self.decoder = nn.Conv2d(48,1,1)





    def forward(self, x, diff):
        x = self.backbone(x)
        x = self.upsample(x)

        mask_up = self.mask_up(diff)
        mask,mask_down = self.mask_down(diff)

        x_up = self.branch_up(x, mask_up)
        x_down = self.branch_down(x, mask_down)

        x = x_up + x_down
        x = self.decoder(x)


        return x


