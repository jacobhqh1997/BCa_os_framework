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
from torchvision import transforms

tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
tile_encoder.eval()

class ImageFeatureDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform_giga = transform
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
        image_giga = self.transform_giga(image)

        return  image_giga, (x, y), img_path

image_dir = 'path/to/patch_img/'
output_dir_giga = 'path/to/save_img/'
os.makedirs(output_dir_giga, exist_ok=True)



dataset = ImageFeatureDataset(image_dir, transform)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tile_encoder.to(device)

features_dict_giga = defaultdict(lambda: {'features': [], 'coords': []})

with torch.no_grad():
    for patches_giga, (y_coords, x_coords), paths in dataloader:

        patches_giga = patches_giga.to(device)
        features_giga = tile_encoder(patches_giga)
        
       
        for feature_giga, y, x, path in zip( features_giga, y_coords, x_coords, paths):
            subdir = os.path.dirname(path).split('/')[-1]
            features_dict_giga[subdir]['features'].append(feature_giga.cpu())
            features_dict_giga[subdir]['coords'].append((y.item(), x.item()))


for subdir, data in features_dict_giga.items():
    filename = f"{subdir}_features.pt"
    result_path_giga = os.path.join(output_dir_giga, filename)
    torch.save({
        'features': torch.stack(data['features']),
        'coords': data['coords']
    }, result_path_giga) 
