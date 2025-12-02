# DSPNet

official code for Dual-branch Dynamic-Static Perception Network for Video Crowd Counting.

## Datasets

- Canteen: [BaiduNetDisk](https://pan.baidu.com/s/18XtesjJTBolXMwHZFoazVw?pwd=yi7b).
- Classroom: [BaiduNetDisk](https://pan.baidu.com/s/1ZbD3aLNuu7syw86a7UQe-g?pwd=z3q8).

## Pretraining Weight

-  ShiftVit-Tiny: [ShiftVit-T-G.pth](https://pan.baidu.com/s/1faf5lFmemvptlAaYd19EZw?pwd=ftvt).

## Install dependencies

torch >= 1.0, torchvision, opencv, numpy, scipy, etc.

## Take training and testing of Bus dataset for example:

1. Download Canteen.

2. Preprocess Canteen to generate ground-truth density maps.

   ```
   python generate_h5.py
   ```

3. Divide the last 10% of the training set into the validation set. The folder structure should look like this:

   ```
   Canteen
   ├──train
       ├──1.img
       ├──1.h5
   ├──val
   ├──test
   ├──canteen_roi.npy
   ```

4. Train Canteen.

   ```
   python train.py (dataset path) (directory for saving model weights)
   ```

5. Test Canteen.

   ```
   python test.py (dataset path) (directory for saving model weights)
   ```
