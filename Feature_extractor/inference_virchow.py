import torch
from torchvision import datasets, models
import os
import re
from openslide.deepzoom import DeepZoomGenerator
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image
from collections import defaultdict
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
import torch
import timm

model_virchow = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model_virchow = model_virchow.eval()
transforms_virchow = create_transform(**resolve_data_config(model_virchow.pretrained_cfg, model=model_virchow))


class ImageFeatureDataset(Dataset):
    def __init__(self, image_dir, transform_virchow=None):
        self.image_dir = image_dir
        self.transform_virchow = transform_virchow
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
        match = re.search(r"patch_x(\d+)_y(\d+).png", os.path.basename(img_path))
        x, y = int(match.group(1)), int(match.group(2))
        image_virchow = self.transform_virchow(image)

        return  image_virchow, (x, y), img_path


image_dir = 'path/to/patch_img/'
output_dir_virchow = 'path/to/save_img/'
os.makedirs(output_dir_virchow, exist_ok=True)



dataset = ImageFeatureDataset(image_dir, transforms_virchow)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=0)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_virchow.to(device)

features_dict_virchow = defaultdict(lambda: {'features': [], 'coords': []})
# 
with torch.no_grad():
    for patches_virchow, (y_coords, x_coords), paths in dataloader:

        patches_virchow = patches_virchow.to(device)
        output = model_virchow(patches_virchow).to(device)
        features_vorchow = torch.cat([output[:, 0], output[:, 5:].mean(1)], dim=-1)
       
        for feature_virchow, y, x, path in zip( features_vorchow, y_coords, x_coords, paths):
            subdir = os.path.dirname(path).split('/')[-1]
            features_dict_virchow[subdir]['features'].append(feature_virchow.cpu())
            features_dict_virchow[subdir]['coords'].append((y.item(), x.item()))


for subdir, data in features_dict_virchow.items():
    filename = f"{subdir}_features.pt"
    result_path_virchow = os.path.join(output_dir_virchow, filename)
    torch.save({
        'features': torch.stack(data['features']),
        'coords': data['coords']
    }, result_path_virchow) 
