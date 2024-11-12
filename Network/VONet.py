import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .gmflow.gmflow import GMF as FlowNet
from .vit import VOT as FlowPoseNet

class VONet(nn.Module):
    def __init__(self):
        super(VONet, self).__init__()
        self.flowNet = FlowNet()

        self.FlowFeatureExtractor = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.fusionFlowCorner = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )



        self.flowPoseNet = FlowPoseNet()




    def forward(self, x):


        def detect_corners(image_tensor):
            device_tensor = image_tensor
            image_tensor = image_tensor.cpu()
            image_array = image_tensor.numpy()
            image_array = np.transpose(image_array, (0, 2, 3, 1))
            image_array = (image_array * 255).astype(np.uint8)
            corner_maps = []
            for i in range(image_array.shape[0]):
                gray = cv2.cvtColor(image_array[i], cv2.COLOR_RGB2GRAY)
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=3)
                if type(corners) == type(None):
                    corner_map = np.zeros_like(gray)
                else:
                    corners = np.int0(corners)
                    corner_map = np.zeros_like(gray)
                    for corner in corners:
                        x, y = corner.ravel()
                        corner_map[y, x] = 1

                corner_maps.append(corner_map)
            corner_maps_tensor = torch.from_numpy(np.stack(corner_maps, axis=0))
            corner_maps_tensor = torch.unsqueeze(corner_maps_tensor, dim=1)
            corner_maps_tensor = corner_maps_tensor.to(device_tensor.device)
            corner_maps_tensor = corner_maps_tensor.float()
        
            return corner_maps_tensor
        
        def process_corner_map(corner_map):
            _, _, height, width = corner_map.size()
            left_width = int(0.1 * width)
            right_width = int(0.9 * width)
            up_height = int(0.1 * height)
            down_height = int(0.9 * height)
            processed_corner_map = torch.zeros_like(corner_map)
            processed_corner_map[:, :, up_height:down_height, left_width:right_width] = corner_map[:, :,
                                                                                        up_height:down_height,
                                                                                        left_width:right_width]
            return processed_corner_map
        
        
        
        make_corner = x[0]
        corner_map = detect_corners(make_corner)
        corner_map = process_corner_map(corner_map)


        flow = self.flowNet(x[0], x[1])
        corner_flow = flow * corner_map
        flow = self.FlowFeatureExtractor(flow)
        corner_flow = self.FlowFeatureExtractor(corner_flow)
        input = flow + corner_flow
        input = self.fusionFlowCorner(input)


        pose = self.flowPoseNet(input)

        return pose






