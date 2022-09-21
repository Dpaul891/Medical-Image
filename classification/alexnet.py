#!/usr/bin/env python
# coding=utf-8
"""
@FilePath: model.py
@Author: Xu Mingyu
@Date: 2022-03-26 21:33:28
@LastEditTime: 2022-03-26 22:43:46
@Description: 
@Copyright 2022 Xu Mingyu, All Rights Reserved. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class AlexNet(nn.Module):
    def __init__(self, class_num, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # [64, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # [64, 27, 27]
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # [192, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # [192, 13, 13]
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # [384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # [256, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # [256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # [256, 6, 6]
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(64, class_num),
        )
        if init_weights:
            self._initialize_weights()
        ### End your code ###
        
        
    def forward(self, x):
        print(x.shape)
        bs = x.shape[0]
        x = self.features(x)
        print(x.shape)
        x = x.view(bs, -1)
    
        # x = torch.flatten(self.pooling(x), start_dim=1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
