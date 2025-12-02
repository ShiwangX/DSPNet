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

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data-dir', default="C:\\File\\mall\\test",
                        help='training data directory')
    parser.add_argument('--save-dir', default="C:\\File\\mall",
                        help='model directory (save visualization here)')
    parser.add_argument('--roi-path', default="C:\\File\\mall\\perspective_roi.npy",
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


# 可视化预测密度图（原函数不变）
def visualize_pred_density(output, test_img_list, start_idx, vis_dir, vis_count):
    f = output.shape[0]
    remaining = 3 - vis_count
    if remaining <= 0:
        return vis_count

    num_vis = min(remaining, f)
    for i in range(num_vis):
        if start_idx + i >= len(test_img_list):
            break

        density_map = output[i].squeeze(0).detach().cpu().numpy()
        if density_map.max() > density_map.min():
            density_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min()) * 255
        else:
            density_norm = np.zeros_like(density_map, dtype=np.uint8)
        density_norm = density_norm.astype(np.uint8)

        heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
        current_img_path = test_img_list[start_idx + i]
        img_basename = os.path.basename(current_img_path).replace('.jpg', '_pred_density.jpg')  # 预测图后缀
        save_path = os.path.join(vis_dir, 'pred', img_basename)
        cv2.imwrite(save_path, heatmap)
        print(f"Saved predicted density map: {save_path}")

        vis_count += 1
        if vis_count >= 3:
            break
    return vis_count


# 新增：可视化真实密度图（从h5文件读取）
def visualize_gt_density(test_img_list, start_idx, vis_dir, vis_count):
    remaining = 3 - vis_count
    if remaining <= 0:
        return vis_count

    num_vis = min(remaining, len(test_img_list) - start_idx)
    for i in range(num_vis):
        current_idx = start_idx + i
        if current_idx >= len(test_img_list):
            break

        # 1. 读取对应的h5文件（真实密度图）
        img_path = test_img_list[current_idx]
        h5_path = img_path.replace('.jpg', '.h5')  # 真实密度图路径
        if not os.path.exists(h5_path):
            print(f"警告：真实密度图文件不存在 {h5_path}")
            continue

        # 2. 从h5文件中提取真实密度图
        with h5py.File(h5_path, 'r') as h5_file:
            gt_density = np.asarray(h5_file['density'])  # 形状: (h_gt, w_gt)

        # 3. 应用ROI mask（与预测图处理一致）
        if args.roi_path:
            mask = np.load(args.roi_path)
            # 真实密度图的mask resize需匹配其尺寸（这里保持原始mask尺寸，若有需要可调整）
            mask = cv2.resize(mask, (gt_density.shape[1], gt_density.shape[0]))
            gt_density = gt_density * mask

        # 4. 归一化到0-255
        if gt_density.max() > gt_density.min():
            gt_norm = (gt_density - gt_density.min()) / (gt_density.max() - gt_density.min()) * 255
        else:
            gt_norm = np.zeros_like(gt_density, dtype=np.uint8)
        gt_norm = gt_norm.astype(np.uint8)

        # 5. 转换为热图并保存
        heatmap = cv2.applyColorMap(gt_norm, cv2.COLORMAP_JET)
        img_basename = os.path.basename(img_path).replace('.jpg', '_gt_density.jpg')  # 真实图后缀
        save_path = os.path.join(vis_dir, 'gt', img_basename)
        cv2.imwrite(save_path, heatmap)
        print(f"Saved ground truth density map: {save_path}")

        vis_count += 1
        if vis_count >= 3:
            break
    return vis_count


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    model = FusionNet(threshold=args.threshold, is_extrapolation=True, is_train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 检查模型文件
    model_path = os.path.join(args.save_dir, 'best_model_27.pth')
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 创建可视化目录（分预测和真实子目录）
    vis_root = os.path.join("C:\\File\\mall", 'density_visualization')
    vis_pred_dir = os.path.join(vis_root, 'pred')  # 预测密度图目录
    vis_gt_dir = os.path.join(vis_root, 'gt')      # 真实密度图目录
    os.makedirs(vis_pred_dir, exist_ok=True)
    os.makedirs(vis_gt_dir, exist_ok=True)
    print(f"预测密度图保存到：{vis_pred_dir}")
    print(f"真实密度图保存到：{vis_gt_dir}")
    vis_count = 0  # 计数：只处理前三帧


    # 处理 mall 数据集
    # 加载测试集
    datasets = CrowdDensity(
        args.data_dir,  # 直接使用test目录
        is_gray=args.is_gray,
        method='val',
        frame_number=args.frame_number,
        roi_path=args.roi_path
    )
    dataloader = DataLoader(datasets, 1, shuffle=False, num_workers=8, pin_memory=False)
    epoch_res = []

    # 加载测试图像列表
    test_img_list = sorted(
        glob(os.path.join(args.data_dir, '*.jpg')),
        key=lambda x: int(x.split('_')[-1].split('.')[0])  # 按seq_xxxxxx中的数字排序
    )
    print(f"测试集图像数量：{len(test_img_list)}")
    if len(test_img_list) == 0:
        print("错误：未找到测试图像")
        exit(1)

    # 先可视化前三帧的真实密度图（独立于模型预测，提前处理）
    visualize_gt_density(test_img_list, start_idx=0, vis_dir=vis_root, vis_count=0)

    start_idx = 0
    for imgs, keypoints, diffs in dataloader:
        if vis_count >= 3:
            break
        b, f, c, h, w = imgs.shape
        assert b == 1, 'batch size must be 1'
        imgs = imgs.to(device).squeeze(0)
        diffs = diffs.to(device).squeeze(0)

        with torch.set_grad_enabled(False):
            output = model(imgs, diffs)
            if args.roi_path:
                mask = np.load(args.roi_path)
                mask = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                mask = torch.tensor(mask).to(device)
                output = output * mask

            # 可视化预测密度图
            vis_count = visualize_pred_density(
                output=output,
                test_img_list=test_img_list,
                start_idx=start_idx,
                vis_dir=vis_root,
                vis_count=vis_count
            )

            res = keypoints[0].numpy() - torch.sum(output.view(f, -1), dim=1).detach().cpu().numpy()
            for r in res:
                epoch_res.append(r)

        start_idx += f

    # 后续误差计算逻辑（保持不变）
    epoch_res = np.array(epoch_res)
    if len(test_img_list) % args.frame_number != 0:
        remain = len(test_img_list) % args.frame_number
        epoch_res = np.delete(epoch_res, slice(-1 * args.frame_number, -1 * remain))

    for j, k in enumerate(test_img_list):
        h5_path = k.replace('jpg', 'h5')
        if not os.path.exists(h5_path):
            print(f"警告：h5文件不存在 {h5_path}")
            continue
        h5_file = h5py.File(h5_path, mode='r')
        h5_map = np.asarray(h5_file['density'])
        if args.roi_path:
            mask = np.load(args.roi_path)
            h5_map = h5_map * mask
        count = np.sum(h5_map)
        print(os.path.basename(k).split('.')[0], epoch_res[j], count, count - epoch_res[j])

    if len(epoch_res) > 0:
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        print(f'Final Test (mall): mae {mae}, mse {mse}')
    else:
        print("警告：未计算出MAE和MSE")