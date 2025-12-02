from pathlib import Path
import argparse
import threading
import multiprocessing
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import scipy
import shutil
import matplotlib.pyplot as plt
import h5py
import re
from PIL import Image
import cv2


class MultiProcessor:
    def __init__(self):
        self.args = self.parse_args()
        self.root = Path(self.args.root) 
        self.num_workers = self.args.num_workers if self.args.num_workers > 0 else multiprocessing.cpu_count()
        self.resize = True if self.args.resize.lower() == 'true' else False

        self.task_list = []
        for phase in ['train', 'test']:
            phase_path = self.root / phase
            for video_path in phase_path.iterdir():
                video_name = video_path.name
                if video_name[0] == '.':  
                    continue
                for img_path in video_path.glob('*.jpg'):
                    if len(re.findall(r'(\d+)\.\w+$', str(img_path))) == 0 or img_path.name[0] == '.':
                        continue
                    img_name = img_path.name
                    self.task_list.append([phase, video_name, img_name])

        self.threads = [threading.Thread(target=self.task, args=(i,)) for i in range(self.num_workers)]
        self.present_index = 0
        self.done_number = 0
        self.lock = threading.RLock()
        print(f'resize: {self.resize}')
        print(f'count of threads: {self.num_workers}')

    def task(self, processor_id):
        while True:
            self.lock.acquire()
            if self.present_index >= len(self.task_list):
                self.lock.release()
                return
            task = self.task_list[self.present_index]
            self.present_index += 1
            self.lock.release()

            img_path = self.root / task[0] / task[1] / task[2]
            json_path = self.root / task[0] / task[1] / task[2].replace('jpg', 'json')

            gt_save_path = img_path.parent / task[2].replace('.jpg', '.h5')


            with open(json_path, 'r') as jsf:
                jss = jsf.read()
                state_dict = json.loads(jss)
                key = list(state_dict.keys())[0]
                points = []
                for e in state_dict[key]['regions']:
                    points.append([e['shape_attributes']['x'], e['shape_attributes']['y']])

            img_ori = plt.imread(img_path)
            gt_density_map = self.gaussian_filter_density(img_ori, points, None, mode='gt')
            raw_h, raw_w = gt_density_map.shape

            if self.resize:
                img = Image.open(str(img_path))
                img = img.resize((640, 360))
                img.save(str(img_path), quality=95) 
                gt_density_map = cv2.resize(gt_density_map, (640, 360)) * (raw_h / 360) * (raw_w / 640)

            h5f = h5py.File(str(gt_save_path), 'w')
            h5f.create_dataset('density', data=gt_density_map)
            h5f.close()

            self.lock.acquire()
            self.done_number += 1
            print(f'Thread {processor_id:<2}: [{self.done_number}/{len(self.task_list)}] '
                  f'{img_path} 处理完成，密度图保存至 {gt_save_path}')
            self.lock.release()

    def start(self):
        for thread in self.threads:
            thread.daemon = True
            thread.start()

        for thread in self.threads:
            thread.join()

    @staticmethod
    def gaussian_filter_density(img, points, multiplying_power, mode='gt'):
        img_shape = [img.shape[0], img.shape[1]]
        density = np.zeros(img_shape, dtype=np.float32)
        gt_count = len(points)
        if gt_count == 0:
            return density

        leafsize = 2048
        tree = KDTree(points.copy(), leafsize=leafsize)
        distances, locations = tree.query(points, k=4)

        for i, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
                pt2d[int(pt[1]), int(pt[0])] = 1.
            else:
                continue
            if gt_count > 1:
                sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            else:
                sigma = np.average(np.array(pt.shape)) / 2. / 2.
            sigma = min(20., sigma)
            if mode == 'amb':
                sigma *= multiplying_power
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        density = np.clip(density, 0., 1.)
        return density

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-r', '--root', help='原始数据根目录',
                            default=r'/ssd4/shiwang/fdst')  # 例如C:\File\FDST
        parser.add_argument('-mp', '--multiplying_power', type=float, default=1.5, help='amb模式的高斯核倍数（当前未使用）')
        parser.add_argument('-rs', '--resize', type=str, default='True', help='是否将图片resize为960x540（True/False）')
        parser.add_argument('-nw', '--num_workers', metavar='NE', type=int, help='线程数量', default=16)
        args = parser.parse_args()
        return args


if __name__ == '__main__':
    mp = MultiProcessor()

    mp.start()
