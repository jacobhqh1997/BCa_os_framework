import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models
import albumentations as A
import re
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
import os
import torch
from torchvision import transforms
import timm


model_virchow = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model_virchow = model_virchow.eval()
transforms_virchow = create_transform(**resolve_data_config(model_virchow.pretrained_cfg, model=model_virchow))

local_dir = "path/to/UNI/"
model_uni = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)
model_uni.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
transform_uni = transforms.Compose(
    [
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model_uni.eval()

class ImageFeatureDataset(Dataset):
    def __init__(self, image_dir, transform_uni=None):
        self.image_dir = image_dir
        self.transform_uni= transform_uni
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
        # match = re.search(r"patch_x(\d+)_y(\d+).png", os.path.basename(img_path))
        match = re.search(r"patch_x(\d+)_y(\d+).*\.png", os.path.basename(img_path))
        x, y = int(match.group(1)), int(match.group(2))
        image_uni = self.transform_uni(image)

        return image_uni, (x, y), img_path


image_dir = 'path/to/patch_img/'
output_dir_uni = 'path/to/save_img/'
os.makedirs(output_dir_uni, exist_ok=True)


dataset = ImageFeatureDataset(image_dir, transform_uni)
dataloader = DataLoader(dataset, batch_size=700, shuffle=False, num_workers=0)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_uni.to(device)



features_dict_uni = defaultdict(lambda: {'features': [], 'coords': []})

# 
with torch.no_grad():
    for patches_uni, (y_coords, x_coords), paths in dataloader:

        patches_uni = patches_uni.to(device)
        features_uni = model_uni(patches_uni)

        for feature_uni, y, x, path in zip(features_uni, y_coords, x_coords, paths):
            subdir = os.path.dirname(path).split('/')[-1]
            features_dict_uni[subdir]['features'].append(feature_uni.cpu())
            features_dict_uni[subdir]['coords'].append((y.item(), x.item())) 




for subdir, data in features_dict_uni.items():
    filename = f"{subdir}_features.pt"
    result_path_uni = os.path.join(output_dir_uni, filename)
    torch.save({
        'features': torch.stack(data['features']),
        'coords': data['coords']
    }, result_path_uni)
