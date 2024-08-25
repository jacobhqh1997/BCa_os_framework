import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models
import albumentations as A
import matplotlib.pyplot as plt
import os
import re
import sys
import pandas as pd
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os.path as osp
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from albumentations.pytorch import ToTensorV2
import cv2
import json
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon, Polygon, Point
from torch.utils.data import DataLoader
import torch
from ctran import ctranspath 
from torchvision import transforms
from PIL import Image
from collections import defaultdict

class ImageFeatureDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = self.collect_images(image_dir)

    def collect_images(self, path):
        imgs = []
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.png'):
                    imgs.append(os.path.join(dirpath, filename))
        return imgs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        match = re.search(r"patch_x(\d+)_y(\d+).*\.png", os.path.basename(img_path))
        x, y = int(match.group(1)), int(match.group(2))
        if self.transform:
            image = self.transform(image)
        return image, (x, y), img_path

# 初始化变量和目录
image_dir = 'path/to/patch_img/'
output_dir = 'path/to/save_img/'
os.makedirs(output_dir, exist_ok=True)

# 设置转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# 加载数据集
dataset = ImageFeatureDataset(image_dir, transform)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=0)

# 初始化模型
model = ctranspath()
model.head = nn.Identity()
td = torch.load('path/to/ctranspath.pth')
model.load_state_dict(td['model'], strict=False)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 存储特征和坐标
features_dict = defaultdict(lambda: {'features': [], 'coords': []})

with torch.no_grad():
    for patches, (y_coords, x_coords), paths in dataloader:

        patches = patches.to(device)
        features = model(patches)
        for feature, y, x, path in zip(features, y_coords, x_coords, paths):
            subdir = os.path.dirname(path).split('/')[-1]
            features_dict[subdir]['features'].append(feature.cpu())
            features_dict[subdir]['coords'].append((y.item(), x.item())) 

# 保存特征和坐标到文件
for subdir, data in features_dict.items():
    filename = f"{subdir}_features.pt"
    result_path = os.path.join(output_dir, filename)
    torch.save({
        'features': torch.stack(data['features']),
        'coords': data['coords']
    }, result_path)
