from __future__ import division
import torch
import math
import random
import numpy as np
import numbers
import cv2
import matplotlib.pyplot as plt
import os
if (not ("DISPLAY" in os.environ)):
    plt.switch_backend('agg')
    print("Environment variable DISPLAY is not present in the system.")
    print("Switch the backend of matplotlib to agg.")

import time

from PIL import Image
# ===== general functions =====


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size

    """

    def __init__(self, scale=2):#scale=2
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        self.downscale = 1.0/scale

    def __call__(self, sample):#对'flow''intrinsic''fmask'降采样  默认1/4
        if self.downscale != 1 and 'img1' in sample:
            sample['img1'] = cv2.resize(sample['img1'],
                                        (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
        if self.downscale != 1 and 'img2' in sample:
            sample['img2'] = cv2.resize(sample['img2'],
                                        (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
        if self.downscale != 1 and 'flow' in sample:
            sample['flow'] = cv2.resize(sample['flow'],
                                        (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        if self.downscale != 1 and 'intrinsic' in sample:
            sample['intrinsic'] = cv2.resize(sample['intrinsic'],
                                             (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        if self.downscale != 1 and 'fmask' in sample:
            sample['fmask'] = cv2.resize(sample['fmask'],
                                         (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        return sample


class CropCenter(object):#中心裁剪
    """Crops the a sample of data (tuple) at center
    if the image size is not large enough, it will be first resized with fixed ratio
    """

    def __init__(self, size):#h w
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        kks = list(sample.keys())
        th, tw = self.size
        h, w = sample[kks[0]].shape[0], sample[kks[0]].shape[1]
        if w == tw and h == th:
            return sample

        # resize the image if the image size is smaller than the target size
        scale_h, scale_w, scale = 1., 1., 1.
        if th > h:
            scale_h = float(th)/h
        if tw > w:
            scale_w = float(tw)/w
        if scale_h > 1 or scale_w > 1:
            scale = max(scale_h, scale_w)
            w = int(round(w * scale))  # w after resize
            h = int(round(h * scale))  # h after resize

        # x1 = random.randint(int((w-tw)*0.2), int((w-tw)*0.8))
        # y1 = random.randint(int((h-th)*0.2), int((h-th)*0.8))
        x1 = int((w - tw) / 2)
        y1 = int((h - th) / 2)

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape) == 3:
                if scale > 1:
                    img = cv2.resize(
                        img, (w, h), interpolation=cv2.INTER_LINEAR)
                sample[kk] = img[y1:y1+th, x1:x1+tw, :]
            elif len(img.shape) == 2:
                if scale > 1:
                    img = cv2.resize(
                        img, (w, h), interpolation=cv2.INTER_LINEAR)
                sample[kk] = img[y1:y1+th, x1:x1+tw]

        return sample


class ToTensor(object):
    """如果数据是三维的，将其转置为通道优先的顺序（C x H x W）。

    如果数据是二维的，将其变形为一个带有单一通道的三维数组（1 x H x W）。

    如果数据是三维的且具有三个通道，将像素值进行归一化，使其在 0 到 1 之间。

    将数据复制为连续的内存块，然后将其转换为 PyTorch 的张量格式
    """
    def __call__(self, sample):
        sss = time.time()

        kks = list(sample)

        for kk in kks:
            data = sample[kk]
            data = data.astype(np.float32)
            if len(data.shape) == 3:  # transpose image-like data
                data = data.transpose(2, 0, 1)
            elif len(data.shape) == 2:
                data = data.reshape((1,)+data.shape)

            # normalization of rgb images
            if len(data.shape) == 3 and data.shape[0] == 3:
                data = data/255.0

            # copy to make memory continuous
            sample[kk] = torch.from_numpy(data.copy())

        return sample


class SampleNormalize(object):
    # Numpy input datatype
    # Normalize optical flow & pose before input into network
    # NN output normalized flow & pose
    """用于对输入样本中的光流（flow）和姿态（motion）进行归一化处理。主要目的是在将这些数据输入神经网络之前，对其进行预处理，以便更好地适应网络的训练过程"""
    def __init__(self, pose_std: np.ndarray = None, flow_norm: float = 20.0) -> None:
        # Set default value
        if pose_std is None:
            # the output scale factor
            self.pose_std = np.array(
                [0.13,  0.13,  0.13,  0.013,  0.013,  0.013], dtype=np.float32)
        else:
            self.pose_std = pose_std
        self.flow_norm = flow_norm  # scale factor for flow

    def __call__(self, sample: np.ndarray):
        if "flow" in sample:
            sample["flow"] = sample["flow"] / self.flow_norm
        if "motion" in sample:
            sample["motion"] = sample["motion"] / self.pose_std
        return sample


def tensor2img(tensImg, mean, std):
    """
    convert a tensor a numpy array, for visualization
    """
    # undo normalize
    for t, m, s in zip(tensImg, mean, std):
        t.mul_(s).add_(m)
    tensImg = tensImg * float(255)
    # undo transpose
    tensImg = (tensImg.numpy().transpose(1, 2, 0)).astype(np.uint8)
    return tensImg


def bilinear_interpolate(img, h, w):
    # assert round(h)>=0 and round(h)<img.shape[0]
    # assert round(w)>=0 and round(w)<img.shape[1]

    h0 = int(math.floor(h))
    h1 = h0 + 1
    w0 = int(math.floor(w))
    w1 = w0 + 1

    a = h - h0
    b = w - w0

    h0 = max(h0, 0)
    w0 = max(w0, 0)
    h1 = min(h1, img.shape[0]-1)
    w1 = min(w1, img.shape[1]-1)

    A = img[h0, w0, :]
    B = img[h1, w0, :]
    C = img[h0, w1, :]
    D = img[h1, w1, :]

    res = (1-a)*(1-b)*A + a*(1-b)*B + (1-a)*b*C + a*b*D

    return res


def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    """用于根据水平和垂直方向的光流分量（du和dv）计算角度和距离"""
    a = np.arctan2(dv, du)

    angleShift = np.pi

    if (True == flagDegree):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt(du * du + dv * dv)

    return a, d, angleShift


def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0):#光流可视化
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv(
        flownp[:, :, 0], flownp[:, :, 1], flagDegree=False)

    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((ang.shape[0], ang.shape[1], 3), dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[:, :, 0] = np.remainder((ang + angShift) / (2*np.pi), 1)
    hsv[:, :, 1] = mag / maxF * n
    hsv[:, :, 2] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 1) * hueMax
    hsv[:, :, 1:3] = np.clip(hsv[:, :, 1:3], 0, 1) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if (mask is not None):
        mask = mask != 255
        bgr[mask] = np.array([0, 0, 0], dtype=np.uint8)

    return bgr


def dataset_intrinsics(dataset='tartanair'):
    if dataset == 'kitti':
        focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
    elif dataset == 'euroc':
        focalx, focaly, centerx, centery = 458.6539916992, 457.2959899902, 367.2149963379, 248.3750000000
    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0
    else:
        return None
    return focalx, focaly, centerx, centery


def plot_traj(gtposes, estposes, vis=False, savefigname=None, title=''):
    fig = plt.figure(figsize=(4, 4))
    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    plt.plot(gtposes[:, 0], gtposes[:, 1], linestyle='dashed', c='k')
    plt.plot(estposes[:, 0], estposes[:, 1], c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth', 'VO'])
    plt.title(title)
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)


def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5)/fx
    hh = (hh.astype(np.float32) - oy + 0.5)/fy
    intrinsicLayer = np.stack((ww, hh)).transpose(1, 2, 0)

    return intrinsicLayer


def load_kiiti_intrinsics(filename):
    '''
    load intrinsics from kitti intrinsics file
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    cam_intrinsics = lines[2].strip().split(' ')[1:]
    focalx, focaly, centerx, centery = float(cam_intrinsics[0]), float(
        cam_intrinsics[5]), float(cam_intrinsics[2]), float(cam_intrinsics[6])

    return focalx, focaly, centerx, centery
