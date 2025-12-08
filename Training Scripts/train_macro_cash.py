import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from network.Macro_networks import resnext50_32x4d, regularize_path_weights
from utils import CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters
import copy
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

from sklearn.metrics import roc_curve
EPOCH = 150
LR = 5e-3
LAMBDA_COX = 1
LAMBDA_REG = 3e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cindex_test_max = 0
cindex_binary_max =0
cudnn.deterministic = True
torch.cuda.manual_seed_all(2024)
torch.manual_seed(2024)
random.seed(2024)
from torchvision.transforms import Compose, Resize, ToTensor    
model = resnext50_32x4d()
model = nn.DataParallel(model, device_ids=[0])
model = model.to(device)
optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
print("Number of Trainable Parameters: %d" % count_parameters(model))
from monai.transforms import Compose
from monai.transforms import Compose, Lambda, MapTransform

albu_transform = A.Compose([
    A.Resize(336, 336),
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
        ID = seg_filepath.split('/')[-1][:-4]
        pd_index = data[data['WSIs'].isin([ID])].index.values[0]
        T = data['death_time'][pd_index]
        O = data['death_status'][pd_index]
        O = torch.tensor(O).type(torch.FloatTensor)
        T = torch.tensor(T).type(torch.FloatTensor)
        data_list.append({"image": seg, "T": T, "O": O, "seg_filepath":seg_filepath})
    return data_list


import os


import pandas as pd

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
            file_path_filtered.append(file_path)
    return np.array(risk_pred_filtered), np.array(censor_filtered), np.array(survtime_filtered), file_path_filtered



if __name__ == '__main__':
    macro_train= 'path/to/train/'
    macro_test= 'path/to/valid/'
    info_train = pd.read_csv('path/to/train.csv')
    info_val =  pd.read_csv('path/to/test.csv')

    Train_list = path_cleaning(macro_train,info_train)
    Val_list = path_cleaning(macro_test,info_val)
    train_data_list = prepare_data_list(Train_list)
    val_data_list = prepare_data_list(Val_list)

    train_dataset = CacheDataset(data=train_data_list, transform=train_transform, cache_num=10, num_workers=0)
    val_dataset = CacheDataset(data=val_data_list, transform=train_transform, cache_num=10, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=0)

    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    
    for epoch in tqdm(range(EPOCH)):
        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        file_path_all = []  
        loss_epoch = 0
        print('train_model_before_weight')
        print(list(model.parameters())[-1])
        for batch_idx, batch in enumerate(train_loader):
            x_path = batch['image'].type(torch.FloatTensor).to(device)

            survtime = batch['T'].to(device)
            censor = batch['O'].to(device)
            weights = batch['weight'].type(torch.float).to(device)
            filepath = batch['seg_filepath']

            _, pred = model(x_path)

            weights = weights.type(torch.float).to(device)
  

            loss_cox = CoxLoss(survtime, censor, pred, device)
            loss_reg = regularize_path_weights(model=model)
            loss = LAMBDA_COX*loss_cox  + LAMBDA_REG*loss_reg  
            loss_epoch += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   
            file_path_all += filepath

        scheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

        loss_epoch /= len(train_loader.dataset)

        max_cindex = 0
        best_threshold = 0
        cindex_epoch_train = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        pvalue_epoch_train  = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_epoch_train  = accuracy_cox(risk_pred_all, censor_all)

        model.eval()
        file_path_all = []  
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])  
        for batch_idx, batch in enumerate(val_loader):
            loss_test = 0
            x_path = batch['image'].type(torch.FloatTensor).to(device)

            survtime = batch['T'].to(device)
            censor = batch['O'].to(device)
            filepath = batch['seg_filepath']
            _, pred = model(x_path)
            file_path_all += filepath
            loss_cox = CoxLoss(survtime, censor, pred, device)
            loss_reg = regularize_path_weights(model=model)
            loss = LAMBDA_COX*loss_cox + LAMBDA_REG*loss_reg
            loss_test += loss.data.item()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   
        cindex_test_total = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        pvalue_test_total = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_test_total = accuracy_cox(risk_pred_all, censor_all)

    
        save_path =  'path/to/metric_macro/'
        if not os.path.exists(save_path): os.makedirs(save_path)

        epoch_idx = epoch

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metric_logger}, 
            save_path + '/model_epoch_{}.pkl'.format(epoch))       
          
        if cindex_test_max < cindex_test_total:
            cindex_test_max = cindex_test_total
            print('current max cindex_test_max:',cindex_test_max)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metric_logger}, 
                save_path + '/best_model.pkl') 
