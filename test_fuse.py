import torch
import os
import numpy as np
from dataset.crowd import CrowdDensity
from model.dspa import FusionNet
import argparse
from glob import glob
import cv2
from torch.utils.data import DataLoader
import h5py
import matplotlib.pyplot as plt
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data-dir', default='',
                        help='training data directory')
    parser.add_argument('--save-dir', default='',
                        help='model directory')
    parser.add_argument('--roi-path', default='',
                        help='roi path')
    parser.add_argument('--frame-number', type=int, default=4,
                        help='the number of input frames')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='thresholds for binary masks')
    args = parser.parse_args()
    return args


def visualize_and_save(model, diff_tensor, batch_idx, save_dir):
    """
    保存可视化结果到文件夹，使用更美观的配色。
    """
    # 1. 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 2. 推理获取 Mask
    model.eval()
    with torch.no_grad():
        # 获取原始 mask
        raw_mask_up,raw_mask_down = model.mask_down(diff_tensor)  # (B, 1, H/8, W/8)

        # 数学运算
        # mask_up (0~1) -> 1+mask_up (1.0~2.0)
        vis_up = raw_mask_up
        # mask_down (0或1) -> 2-mask_down (若原为1则得1，若原为0则得2)
        vis_down = 2 - raw_mask_down

        # 3. 数据转换 (取 Batch 中的第 0 帧)
    # 假设 diff_tensor 是 (B, C, H, W)，取第一张图的均值通道用于显示轮廓
    img_input = diff_tensor[0].mean(dim=0).detach().cpu().numpy()

    # Mask 转换
    img_up = vis_up[0, 0].detach().cpu().numpy()
    img_down = vis_down[0, 0].detach().cpu().numpy()

    # 4. 绘图配置
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 图1: 输入的 Diff 图 (灰度) ---
    axes[0].imshow(img_input, cmap='gray')
    axes[0].set_title(f"Input Diff (Batch {batch_idx})")
    axes[0].axis('off')

    # --- 图2: 1 + Mask UP (连续值) ---
    # 推荐配色: 'magma' (黑紫->亮黄), 'inferno' (黑红->亮黄), 或 'plasma'
    # 这些配色更能体现"能量"或"注意力"的感觉
    im1 = axes[1].imshow(img_up, cmap='magma', vmin=0, vmax=1)
    axes[1].set_title("1 + Mask UP\n(Attention Heatmap)")
    axes[1].axis('off')
    # 添加色条
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # --- 图3: 2 - Mask DOWN (二值) ---
    # 逻辑: 原Mask=1(运动) -> 结果=1; 原Mask=0(静止) -> 结果=2
    # 我们希望 1 (运动) 看起来明显，2 (背景) 看起来淡化
    # 使用 'coolwarm_r' (反转冷暖色):
    # 这样较小的值(1.0)会显示为红色/暖色(强调)，较大的值(2.0)显示为蓝色/冷色(背景)
    im2 = axes[2].imshow(img_down, cmap='coolwarm_r', vmin=1.0, vmax=2.0)
    axes[2].set_title("2 - Mask DOWN\n(Red=1/Motion, Blue=2/Static)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[1, 2])

    # 5. 保存并关闭
    save_path = os.path.join(save_dir, f"vis_batch_{batch_idx:04d}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')  # bbox_inches去除白边
    plt.close(fig)  # 必须关闭，否则内存溢出


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu··
    model = FusionNet(threshold=args.threshold,is_extrapolation=True,is_train=False)
    # model = FusionNet()
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model_27.pth'), device))

    if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'dronecrowd' in args.data_dir:
        sum_res = []
        datasets = [CrowdDensity(args.data_dir+'/'+'test'+'/'+file, is_gray=args.is_gray, method='val',
                          frame_number=args.frame_number, roi_path=args.roi_path)
                    for file in sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)]
        dataloader = [DataLoader(datasets[file], 1, shuffle=False, num_workers=8, pin_memory=False)
                      for file in range(len(os.listdir(os.path.join(args.data_dir, 'test'))))]
        file_list = sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)
        for file in range(len(file_list)):
            epoch_res = []
            for imgs, keypoints,diffs in dataloader[file]:
                b, f, c, h, w = imgs.shape
                assert b == 1, 'the batch size should equal to 1 in validation mode'
                imgs = imgs.to(device).squeeze(0)
                diffs = diffs.to(device).squeeze(0)
                with torch.set_grad_enabled(False):
                    output = model(imgs,diffs)
                    if args.roi_path:
                        mask = np.load(args.roi_path)
                        mask = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                        mask = torch.tensor(mask).to(device)
                        output = output * mask
                    res = keypoints[0].numpy() - torch.sum(output.view(f, -1), dim=1).detach().cpu().numpy()
                    for r in res:
                        epoch_res.append(r)
            epoch_res = np.array(epoch_res)
            if 'fdst' in args.data_dir or 'ucsd' in args.data_dir:
                test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                       key=lambda x: int(x.split('/')[-1].split('.')[0]))
            else:
                test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(test_img_list) % args.frame_number != 0:
                remain = len(test_img_list) % args.frame_number
                epoch_res = np.delete(epoch_res, slice(-1 * args.frame_number, -1 * remain))
            for j, k in enumerate(test_img_list):

                h5_path = k.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])
                if args.roi_path:
                    mask = np.load(args.roi_path)
                    h5_map = h5_map * mask
                count = np.sum(h5_map)

                print(k, epoch_res[j], count, count - epoch_res[j])
            for e in epoch_res:
                sum_res.append(e)
        sum_res = np.array(sum_res)
        mse = np.sqrt(np.mean(np.square(sum_res)))
        mae = np.mean(np.abs(sum_res))
        log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
        print(log_str)

    elif 'venice' in args.data_dir:
        sum_res = []
        datasets = [CrowdDensity(args.data_dir+'/'+'test'+'/'+file, is_gray=args.is_gray, method='val',
                          frame_number=args.frame_number, roi_path=args.roi_path)
                    for file in sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)]
        dataloader = [DataLoader(datasets[file], 1, shuffle=False, num_workers=8, pin_memory=False)
                      for file in range(len(os.listdir(os.path.join(args.data_dir, 'test'))))]
        file_list = sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)
        for file in range(len(file_list)):
            epoch_res = []
            for imgs, keypoints, masks,diffs in dataloader[file]:
                b, f, c, h, w = imgs.shape
                assert b == 1, 'the batch size should equal to 1 in validation mode'
                imgs = imgs.to(device).squeeze(0)
                masks = masks.to(device).squeeze(0)
                diffs = diffs.to(device).squeeze(0)
                with torch.set_grad_enabled(False):
                    output = model(imgs,diffs)
                    output = output * masks
                    res = keypoints[0].numpy() - torch.sum(output.view(f, -1), dim=1).detach().cpu().numpy()
                    for r in res:
                        epoch_res.append(r)
            epoch_res = np.array(epoch_res)
            test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(test_img_list) % args.frame_number != 0:
                remain = len(test_img_list) % args.frame_number
                epoch_res = np.delete(epoch_res, slice(-1 * args.frame_number, -1 * remain))
            for j, k in enumerate(test_img_list):

                h5_path = k.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])
                if args.roi_path:
                    mask = np.load(args.roi_path)
                    h5_map = h5_map * mask
                count = np.sum(h5_map)

                print(k, epoch_res[j], count, count - epoch_res[j])
            for e in epoch_res:
                sum_res.append(e)
        sum_res = np.array(sum_res)
        mse = np.sqrt(np.mean(np.square(sum_res)))
        mae = np.mean(np.abs(sum_res))
        log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
        print(log_str)

    else:
        datasets = CrowdDensity(os.path.join(args.data_dir, 'test'), is_gray=args.is_gray, method='val',
                         frame_number=args.frame_number, roi_path=args.roi_path)
        dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=8, pin_memory=False)
        epoch_res = []
        for i, (imgs, keypoints,diffs) in enumerate(dataloader):
            b, f, c, h, w = imgs.shape
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            imgs = imgs.to(device).squeeze(0)
            diffs = diffs.to(device).squeeze(0)
            visualize_and_save(model, diffs, batch_idx=i, save_dir='mask')
            with torch.set_grad_enabled(False):
                output = model(imgs,diffs)
                if args.roi_path:
                    mask = np.load(args.roi_path)
                    mask = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                    mask = torch.tensor(mask).to(device)
                    output = output * mask
                res = keypoints[0].numpy() - torch.sum(output.view(f, -1), dim=1).detach().cpu().numpy()
                for r in res:
                    epoch_res.append(r)
        epoch_res = np.array(epoch_res)
        test_img_list = sorted(glob(os.path.join(os.path.join(args.data_dir, 'test'), '*.jpg')),
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(test_img_list) % args.frame_number != 0:
            remain = len(test_img_list) % args.frame_number
            epoch_res = np.delete(epoch_res, slice(-1*args.frame_number, -1*remain))
        for j, k in enumerate(test_img_list):

            h5_path = k.replace('jpg', 'h5')
            h5_file = h5py.File(h5_path, mode='r')
            h5_map = np.asarray(h5_file['density'])
            if args.roi_path:
                mask = np.load(args.roi_path)
                h5_map = h5_map * mask
            count = np.sum(h5_map)

            print(os.path.basename(k).split('.')[0], epoch_res[j], count, count-epoch_res[j])
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
        print(log_str)
