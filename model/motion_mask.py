import torch
import torch.nn as nn


class MotionMask_UP_old(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64, output_masks=1):
        super(MotionMask_UP_old, self).__init__()

        # 第一层卷积：2倍下采样（stride=2），输出尺寸 H/2, W/2
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,  # 步长=2，第一次下采样
            padding=1  # 保持边缘信息，确保尺寸正确缩小
        )

        # 第二层卷积：再2倍下采样，输出尺寸 H/4, W/4
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,  # 第二次下采样
            padding=1
        )

        # 第三层卷积：再2倍下采样，输出尺寸 H/8, W/8
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=output_masks,
            kernel_size=3,
            stride=2,  # 第三次下采样
            padding=1
        )

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # mask值限制在0~1之间

    def forward(self, x):
        # 输入形状: [B, 3, H, W]（B为批量大小，例如3）

        # 第一层：卷积 + ReLU → 尺寸 H/2, W/2
        x = self.conv1(x)  # 形状: [B, 64, H/2, W/2]
        # x = self.relu(x)

        # 第二层：卷积 + ReLU → 尺寸 H/4, W/4
        x = self.conv2(x)  # 形状: [B, 64, H/4, W/4]
        # x = self.relu(x)

        # 第三层：卷积 + Sigmoid → 尺寸 H/8, W/8
        x = self.conv3(x)  # 形状: [B, 3, H/8, W/8]
        x = self.relu(x)
        masks = self.sigmoid(x)  # 输出mask值在0~1之间

        return masks


class MotionMask_DOWN_old(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64, threshold=0.5):
        super(MotionMask_DOWN_old, self).__init__()
        self.threshold = threshold  # 二值化阈值（超过为1，否则为0）

        # 第一层卷积：升维并提取特征
        # 可以使用排序分块的方法来生成二值掩码
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1  # 保持H、W不变
        )

        # 第二层卷积：进一步提取特征
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # 第三层卷积：降维到单通道（掩码通道）
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,  # 单通道掩码
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.relu = nn.ReLU()  # 中间层激活

    def forward(self, x):
        # 输入形状：[3, 3, H, W]（batch_size=3，3通道）

        # 第一层卷积 + ReLU
        x = self.conv1(x)  # 形状：[3, 64, H, W]
        # x = self.relu(x)

        # 第二层卷积 + ReLU
        x = self.conv2(x)  # 形状：[3, 64, H, W]
        #

        # 第三层卷积（无激活，输出原始分数）
        x = self.conv3(x)  # 形状：[3, 1, H, W]
        x = self.relu(x)

        # 二值化：超过阈值为1，否则为0（训练时保留梯度近似，推理时严格二值化）
        x = torch.sigmoid(x)
        binary_mask = (x > self.threshold).float()


        return binary_mask


import torch
import torch.nn as nn


class MotionMask_UP(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64, output_masks=1):
        super(MotionMask_UP, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=output_masks,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入的高度和宽度（B, C, H, W）
        H_in, W_in = x.shape[2], x.shape[3]

        # 计算目标输出尺寸：(target_h, target_w)
        target_h = (H_in // 32) * 4
        target_w = (W_in // 32) * 4

        # 下采样过程
        x = self.conv1(x)  # [B, 64, H_in/2, W_in/2]
        x = self.relu(x)
        x = self.conv2(x)  # [B, 64, H_in/4, W_in/4]
        x = self.relu(x)
        x = self.conv3(x)  # [B, output_masks, H_in/8, W_in/8]
        x = self.relu(x)
        masks = self.sigmoid(x)

        # 动态裁剪到目标尺寸（裁剪多余的行/列）
        masks = masks[:, :, :target_h, :target_w]  # [B, output_masks, target_h, target_w]

        return masks


class MotionMask_DOWN(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64, threshold=0.5):
        super(MotionMask_DOWN, self).__init__()
        self.threshold = threshold
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # 获取输入的高度和宽度（B, C, H, W）
        H_in, W_in = x.shape[2], x.shape[3]

        # 计算目标输出尺寸：(target_h, target_w)
        target_h = (H_in // 32) * 4
        target_w = (W_in // 32) * 4

        # 下采样过程
        x = self.conv1(x)  # [B, 64, H_in/2, W_in/2]
        x = self.relu(x)
        x = self.conv2(x)  # [B, 64, H_in/4, W_in/4]
        x = self.relu(x)
        x = self.conv3(x)  # [B, 1, H_in/8, W_in/8]
        x = self.relu(x)

        # 二值化
        x = torch.sigmoid(x)
        binary_mask = (x > self.threshold).float()

        # 动态裁剪到目标尺寸
        binary_mask = binary_mask[:, :, :target_h, :target_w]  # [B, 1, target_h, target_w]

        return x,binary_mask



# 测试代码
if __name__ == "__main__":
    # 输入：批量大小=3，3通道，H=256，W=256（可被8整除）
    input_tensor = torch.randn(4, 3, 720, 1280)
    print(f"输入形状: {input_tensor.shape}")  # 应输出 torch.Size([3, 3, 256, 256])

    # model = MotionMask_UP()
    model = MotionMask_DOWN(threshold=0.5)
    output_masks = model(input_tensor)

    print(f"输出形状: {output_masks.shape}")  # 应输出 torch.Size([3, 3, 32, 32])（256/8=32）
    print(f"输出值范围: [{output_masks.min():.4f}, {output_masks.max():.4f}]")  # 应在0~1之间
