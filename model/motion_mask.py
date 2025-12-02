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
        H_in, W_in = x.shape[2], x.shape[3]

        target_h = (H_in // 32) * 4
        target_w = (W_in // 32) * 4

        x = self.conv1(x)  # [B, 64, H_in/2, W_in/2]
        x = self.relu(x)
        x = self.conv2(x)  # [B, 64, H_in/4, W_in/4]
        x = self.relu(x)
        x = self.conv3(x)  # [B, output_masks, H_in/8, W_in/8]
        x = self.relu(x)
        masks = self.sigmoid(x)

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
        H_in, W_in = x.shape[2], x.shape[3]

        target_h = (H_in // 32) * 4
        target_w = (W_in // 32) * 4

        x = self.conv1(x)  # [B, 64, H_in/2, W_in/2]
        x = self.relu(x)
        x = self.conv2(x)  # [B, 64, H_in/4, W_in/4]
        x = self.relu(x)
        x = self.conv3(x)  # [B, 1, H_in/8, W_in/8]
        x = self.relu(x)

        x = torch.sigmoid(x)
        binary_mask = (x > self.threshold).float()

        binary_mask = binary_mask[:, :, :target_h, :target_w]  # [B, 1, target_h, target_w]

        return x,binary_mask



