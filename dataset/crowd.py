from PIL import Image
import torch.utils.data as data
import os
from glob import glob
from torchvision import transforms
import numpy as np
import h5py
import torch
from scipy.io import loadmat
import cv2
import random
import torch.nn.functional as F

class CrowdDensity(data.Dataset):
    def __init__(self, root_path, is_gray=False, method='train', frame_number=3,
                 crop_height=512, crop_width=512, roi_path=None):
        self.root_path = root_path
        self.frame_number = frame_number
        self.diff_number = frame_number + 1
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.roi_path = roi_path
        if 'fdst' in self.root_path or 'ucsd' in self.root_path:
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')),
                                  key=lambda x: int(x.split('/')[-1].split('.')[0]))
        else:
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')),
                                  key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method
        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        if self.method == 'train':
            return len(self.im_list) - self.frame_number + 1
        elif self.method == 'val':
            return len(self.im_list) // self.frame_number + (len(self.im_list) % self.frame_number != 0)

    def __getitem__(self, item):
        img_list = []
        diff_list = []
        target_list = []
        keypoint_list = []
        mask_list = []
        if self.method == 'train' and 'venice' not in self.root_path:
            total_frames = len(self.im_list)  # 1272
            max_index = total_frames - 1
            width, height = Image.open(self.im_list[0]).convert('RGB').size
            new_width = self.crop_width
            new_height = self.crop_height
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            right = left + new_width
            bottom = top + new_height
            rate = random.random()
            if self.roi_path:
                mask = np.load(self.roi_path)
                mask = mask[top:bottom, left:right]
                if rate > 0.5:
                    mask = np.fliplr(mask)
                mask = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
            else:
                mask = np.ones((new_height // 32 * 4, new_width // 32 * 4), dtype=int)
            k_start = item
            k_end = item + self.diff_number
            if k_end > max_index + 1:  # 因为range是左闭右开，k_end需要 <= max_index + 1
                k_start = max_index - self.diff_number + 1  # 1271 - 4 + 1 = 1268
                k_end = k_start + self.diff_number

            for k in range(k_start, k_end - 1):
                cur_path = self.im_list[k]
                cur_frame = Image.open(cur_path).convert('RGB')
                cur_frame = cur_frame.crop((left, top, right, bottom))
                np_ones = np.ones_like(cur_frame).astype('float32')
                if rate > 0.5:
                    cur_frame = cur_frame.transpose(Image.FLIP_LEFT_RIGHT)
                if k < max_index:
                    next_path = self.im_list[k+1]
                    next_frame = Image.open(next_path).convert('RGB')
                    next_frame = next_frame.crop((left, top, right, bottom))
                    if rate > 0.5:
                        next_frame = next_frame.transpose(Image.FLIP_LEFT_RIGHT)
                    diff = np.abs(np.log(np.array(cur_frame) + np_ones) - np.log(np.array(next_frame) + np_ones))
                    diff_tensor = torch.from_numpy(diff).permute(2, 0, 1).float()  # 假设是图像格式(H,W,C)
                    diff_list.append(diff_tensor)

            for q in range(item, item + self.frame_number):
                img_path = self.im_list[q]
                img = Image.open(img_path).convert('RGB')
                img = img.crop((left, top, right, bottom))
                if rate > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = self.trans(img)
                img_list.append(img)

                target_path = img_path.replace('jpg', 'h5')
                target_file = h5py.File(target_path, mode='r')
                target_ori = np.asarray(target_file['density'])
                target_ori = target_ori[top:bottom, left:right]
                if rate > 0.5:
                    target_ori = np.fliplr(target_ori)
                target = cv2.resize(target_ori, (target_ori.shape[1] // 32 * 4, target_ori.shape[0] // 32 * 4),
                                    interpolation=cv2.INTER_CUBIC) * (
                                 (target_ori.shape[0] * target_ori.shape[1]) / (
                                 (target_ori.shape[1] // 32 * 4) * (target_ori.shape[0] // 32 * 4)))
                if self.roi_path:
                    target = target * mask
                keypoint = np.sum(target)
                keypoint_list.append(keypoint)
                target_list.append(torch.from_numpy(target.copy()).float().unsqueeze(0))
            return torch.stack(img_list, dim=0), torch.stack(target_list, dim=0), torch.tensor(
                keypoint_list), torch.tensor(mask) ,torch.stack(diff_list, dim=0)


        elif self.method == 'train' and 'venice' in self.root_path:
            total_frames = len(self.im_list)
            max_index = total_frames - 1
            width, height = Image.open(self.im_list[0]).convert('RGB').size
            new_width = self.crop_width
            new_height = self.crop_height
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            right = left + new_width
            bottom = top + new_height
            rate = random.random()
            k_start = item
            k_end = item + self.diff_number
            if k_end > max_index + 1:  # 因为range是左闭右开，k_end需要 <= max_index + 1
                k_start = max_index - self.diff_number + 1  # 1271 - 4 + 1 = 1268
                k_end = k_start + self.diff_number

            for k in range(k_start, k_end - 1):
                cur_path = self.im_list[k]
                cur_frame = Image.open(cur_path).convert('RGB')
                cur_frame = cur_frame.crop((left, top, right, bottom))
                np_ones = np.ones_like(cur_frame).astype('float32')
                if rate > 0.5:
                    cur_frame = cur_frame.transpose(Image.FLIP_LEFT_RIGHT)
                if k < max_index:
                    next_path = self.im_list[k + 1]
                    next_frame = Image.open(next_path).convert('RGB')
                    next_frame = next_frame.crop((left, top, right, bottom))
                    if rate > 0.5:
                        next_frame = next_frame.transpose(Image.FLIP_LEFT_RIGHT)
                    diff = np.abs(np.log(np.array(cur_frame) + np_ones) - np.log(np.array(next_frame) + np_ones))
                    diff_tensor = torch.from_numpy(diff).permute(2, 0, 1).float()  # 假设是图像格式(H,W,C)
                    diff_list.append(diff_tensor)

            for q in range(item, item + self.frame_number):
                img_path = self.im_list[q]
                img = Image.open(img_path).convert('RGB')
                img = img.crop((left, top, right, bottom))
                if rate > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = self.trans(img)
                img_list.append(img)

                roi_path = img_path.replace('jpg', 'mat')
                mask = loadmat(roi_path)['roi']
                mask = mask[top:bottom, left:right]
                if rate > 0.5:
                    mask = np.fliplr(mask)
                mask = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                mask_list.append(torch.from_numpy(mask.copy()).float().unsqueeze(0))

                target_path = img_path.replace('jpg', 'h5')
                target_file = h5py.File(target_path, mode='r')
                target_ori = np.asarray(target_file['density'])
                target_ori = target_ori[top:bottom, left:right]
                if rate > 0.5:
                    target_ori = np.fliplr(target_ori)
                target = cv2.resize(target_ori, (target_ori.shape[1] // 32 * 4, target_ori.shape[0] // 32 * 4),
                                    interpolation=cv2.INTER_CUBIC) * (
                                 (target_ori.shape[0] * target_ori.shape[1]) / (
                                 (target_ori.shape[1] // 32 * 4) * (target_ori.shape[0] // 32 * 4)))
                target = target * mask
                keypoint = np.sum(target)
                keypoint_list.append(keypoint)
                target_list.append(torch.from_numpy(target.copy()).float().unsqueeze(0))
            return torch.stack(img_list, dim=0), torch.stack(target_list, dim=0), torch.tensor(
                keypoint_list), torch.stack(mask_list, dim=0),torch.stack(diff_list, dim=0)

        elif self.method == 'val' and 'venice' not in self.root_path:
            item = item * self.frame_number
            if item + self.frame_number > len(self.im_list):
                item = len(self.im_list) - self.frame_number

            step = self.frame_number  # k的步长与q保持一致（确保样本对齐）
            max_group_start = len(self.im_list) - self.diff_number # 最后一组的起始帧（确保4帧）
            # 根据item计算k的起始帧（与q_start对齐）
            k_start = min(item, max_group_start)
            k_end = k_start + self.diff_number  # 每组4帧（如0-4→0,1,2,3）

            for k in range(k_start, k_end -1):
                # 加载当前帧（无裁剪，直接使用原始图像）
                cur_path = self.im_list[k]
                cur_frame = Image.open(cur_path).convert('RGB')  # val模式不裁剪
                if 'class' in self.root_path:
                    # resize参数为(width, height)，这里设置为(1280, 704)
                    cur_frame = cur_frame.resize((1280, 704))
                # cur_frame = self.trans(cur_frame)
                # cur_frame = F.interpolate(cur_frame, size=(512, 512), mode='bilinear', align_corners=False)
                np_ones = np.ones_like(cur_frame).astype('float32')
                # 计算当前帧与下一帧的差异（无翻转，保持原始顺序）
                if k < len(self.im_list) - 1:  # 确保k+1不越界
                    next_path = self.im_list[k + 1]
                    next_frame = Image.open(next_path).convert('RGB')  # val模式不裁剪
                    if 'class' in self.root_path:
                        next_frame = next_frame.resize((1280, 704))
                    # next_frame = self.trans(next_frame)
                    # next_frame = F.interpolate(next_frame, size=(512, 512), mode='bilinear', align_corners=False)

                    diff = np.abs(np.log(np.array(cur_frame) + np_ones) - np.log(np.array(next_frame) + np_ones))  # 现在是3维：[C, H, W]


                    # 2. Resize 到 512×512（使用双线性插值，适合连续值）
                    # diff_resized_hwc = cv2.resize(
                    #     diff,
                    #     dsize=(512,512),
                    #     interpolation=cv2.INTER_LINEAR  # 双线性插值，适合平滑过渡
                    # )

                    # 3. 恢复通道顺序为 [C, 512, 512]
                    diff = diff.transpose(2, 0, 1)

                    diff = torch.from_numpy(diff).float()
                    diff_list.append(diff)

            for q in range(item, item + self.frame_number):
                img_path = self.im_list[q]
                img = Image.open(img_path).convert('RGB')
                if 'class' in self.root_path:
                    img = img.resize((1280, 704))
                img = self.trans(img)

                # img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
                img_list.append(img)

                h5_path = img_path.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])
                if self.roi_path:
                    mask = np.load(self.roi_path)
                    h5_map = h5_map * mask
                keypoint = np.sum(h5_map)

                keypoint_list.append(keypoint)

            return torch.stack(img_list, dim=0), torch.tensor(keypoint_list),torch.stack(diff_list, dim=0)


        elif self.method == 'val' and 'venice' in self.root_path:
            item = item * self.frame_number
            if item + self.frame_number > len(self.im_list):
                item = len(self.im_list) - self.frame_number
            max_group_start = len(self.im_list) - self.diff_number
            k_start = min(item, max_group_start)
            k_end = k_start + self.diff_number  # 每组4帧（如0-4→0,1,2,3）
            for k in range(k_start, k_end -1):
                # 加载当前帧（无裁剪，直接使用原始图像）
                cur_path = self.im_list[k]
                cur_frame = Image.open(cur_path).convert('RGB')  # val模式不裁剪
                if 'class' in self.root_path:
                    # resize参数为(width, height)，这里设置为(1280, 704)
                    cur_frame = cur_frame.resize((1280, 704))
                # cur_frame = self.trans(cur_frame)
                # cur_frame = F.interpolate(cur_frame, size=(512, 512), mode='bilinear', align_corners=False)
                np_ones = np.ones_like(cur_frame).astype('float32')
                # 计算当前帧与下一帧的差异（无翻转，保持原始顺序）
                if k < len(self.im_list) - 1:  # 确保k+1不越界
                    next_path = self.im_list[k + 1]
                    next_frame = Image.open(next_path).convert('RGB')  # val模式不裁剪
                    if 'class' in self.root_path:
                        next_frame = next_frame.resize((1280, 704))
                    # next_frame = self.trans(next_frame)
                    # next_frame = F.interpolate(next_frame, size=(512, 512), mode='bilinear', align_corners=False)

                    diff = np.abs(np.log(np.array(cur_frame) + np_ones) - np.log(np.array(next_frame) + np_ones))  # 现在是3维：[C, H, W]
                    # 2. Resize 到 512×512（使用双线性插值，适合连续值）
                    # diff_resized_hwc = cv2.resize(
                    #     diff,
                    #     dsize=(512,512),
                    #     interpolation=cv2.INTER_LINEAR  # 双线性插值，适合平滑过渡
                    # )
                    # 3. 恢复通道顺序为 [C, 512, 512]
                    diff = diff.transpose(2, 0, 1)
                    diff = torch.from_numpy(diff).float()
                    diff_list.append(diff)


            for q in range(item, item + self.frame_number):
                img_path = self.im_list[q]
                img = Image.open(img_path).convert('RGB')
                img = self.trans(img)
                img_list.append(img)

                h5_path = img_path.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])

                roi_path = img_path.replace('jpg', 'mat')
                mask = loadmat(roi_path)['roi']
                mask_resize = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                mask_list.append(torch.from_numpy(mask_resize.copy()).float().unsqueeze(0))
                h5_map = h5_map * mask
                keypoint = np.sum(h5_map)

                keypoint_list.append(keypoint)

            return torch.stack(img_list, dim=0), torch.tensor(keypoint_list), torch.stack(mask_list, dim=0),torch.stack(diff_list, dim=0)



mall_root = "C:\\File\\mall\\train"
crowd = CrowdDensity(mall_root,method='train',crop_height=384,crop_width=384,frame_number=4)
# print(len(crowd))
img,target,keypoint,mask,diff = crowd[0]
print(img.shape)
print(target.shape)
print(diff.shape)
print(mask.shape)
print(keypoint)
# mall_root = "C:\\File\\class\\val"
# crowd_val = CrowdDensity(mall_root,method='val',diff_number=4,crop_height=512,crop_width=512,frame_number=4)
# print(len(crowd_val))
# img , keypoint , diff = crowd_val[15]
# print(img.shape)
# print(keypoint)
# print(diff.shape)

