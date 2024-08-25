import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from Macro_networks import resnext50_32x4d

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd 
import os
from monai.data import CacheDataset
import pickle
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Transform

from lifelines.utils import concordance_index
INFO_PATH = 'path/to/clinical_information/'#clinical info

EPOCH = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cindex_test_max = 0
cindex_binary_max =0
cudnn.deterministic = True
torch.cuda.manual_seed_all(2024)
torch.manual_seed(2024)
random.seed(2024)
from torchvision.transforms import Compose, Resize, ToTensor    
model = resnext50_32x4d()
checkpoint = torch.load('path/to/macro_pkl/best_model.pkl')
model.load_state_dict(checkpoint['model_state_dict'])
model = nn.DataParallel(model, device_ids=[0])
model = model.to(device)
max_cindex =0

from monai.transforms import Compose
from monai.transforms import Compose, Lambda, MapTransform

albu_transform = A.Compose([
    A.Resize(448, 448),
    ToTensorV2(),
])

class AlbumentationsTransform(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.transform = albu_transform

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(image=d[key])["image"]
        return d

train_transform = Compose([
    AlbumentationsTransform(keys=["image"]),
])


def prepare_data_list(seg_filepaths):
    data_list = []
    for seg_filepath in seg_filepaths:
        seg = np.load(seg_filepath)
        seg_filepath = seg_filepath
        hospital = seg_filepath.split('/')[-2]
        base_dir = INFO_PATH
        data = pd.read_csv(base_dir + hospital + '.csv')
        # print('data:',data)
        ID = seg_filepath.split('/')[-1][:-4]
        # print('ID:',ID)
        pd_index = data[data['WSIs'].isin([ID])].index.values[0]
        T = data['death_time'][pd_index]
        O = data['death_status'][pd_index]
        O = torch.tensor(O).type(torch.FloatTensor)
        T = torch.tensor(T).type(torch.FloatTensor)
        data_list.append({"image": seg, "death_time": T, "death_status": O,"seg_filepath":seg_filepath})
    return data_list


import os

def get_files(path, rule=".npy"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all

def path_cleaning(macro_path, info_df):
    cleaned_path = []
    seg_list = get_files(macro_path)
    # print('seg_list:',seg_list)
    info_list = list(info_df['WSIs'])
    for i in seg_list:
        if os.path.splitext(os.path.basename(i))[0] in info_list:
            cleaned_path.append(i)
    return cleaned_path

def filter_values(risk_pred_all, censor_all, survtime_all, file_path_all, wsis_values):
    risk_pred_filtered, censor_filtered, survtime_filtered, file_path_filtered = [], [], [], []
    for risk_pred, censor, survtime, file_path in zip(risk_pred_all, censor_all, survtime_all, file_path_all):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        if filename in wsis_values:
            risk_pred_filtered.append(risk_pred)
            censor_filtered.append(censor)
            survtime_filtered.append(survtime)
            file_path_filtered.append(filename)
    return np.array(risk_pred_filtered), np.array(censor_filtered), np.array(survtime_filtered), file_path_filtered



if __name__ == '__main__':
    macro_test= 'path/to/npy/'
    info_val =  pd.read_csv('path/to/npy.csv')
    Val_list = path_cleaning(macro_test, info_val)
    val_data_list = prepare_data_list(Val_list)

    val_dataset = CacheDataset(data=val_data_list, transform=train_transform, cache_num=10, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    metric_logger = {'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    
    for epoch in tqdm(range(EPOCH)):
        model.eval()
        file_path_all = []  
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])  
        for batch_idx, batch in enumerate(val_loader):
            x_path = batch['image'].type(torch.FloatTensor).to(device)
            survtime = batch['death_time'].to(device)
            censor = batch['death_status'].to(device)
            filepaths = batch['seg_filepath']
            preds, _ = model(x_path)
            filename = os.path.basename(filepaths[0]).replace('.npy', '.pt')
            save_path = os.path.join('path/to/Macro_feature/', filename)
            torch.save(preds.cpu(), save_path)
    

