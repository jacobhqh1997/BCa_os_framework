import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from utils import CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters
import copy
import albumentations as A
import os
from monai.data import CacheDataset
import pickle
from torch.utils.data import DataLoader
from monai.transforms import Transform
from network.transmil import TransMIL
import pandas as pd

INFO_PATH ='path/to/clinical_information/'
EPOCH = 140
LR = 5e-3
LAMBDA_COX = 1
LAMBDA_REG = 3e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.deterministic = True
torch.cuda.manual_seed_all(3407)
torch.manual_seed(3407)
random.seed(3407)
from torchvision.transforms import Compose   

Argument = {
    'input_dim': 3072,  
    'num_classes': 1,  
    'pos_enc': 'PPEG', 
}
model = TransMIL(**Argument) 
model = nn.DataParallel(model, device_ids=[0])
model = model.to(device)
from lifelines.utils import concordance_index
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
print("Number of Trainable Parameters: %d" % count_parameters(model))
cindex_test_max = 0
cindex_binary_max =0



def prepare_data_list(seg_filepaths):
    data_list = []
    for seg_filepath in seg_filepaths:
        seg = torch.load(seg_filepath)
        features = seg['features']
        features = features.cpu() 
        seg_filepath = seg_filepath
        hospital = seg_filepath.split('/')[-2]
        base_dir = INFO_PATH
        data = pd.read_csv(base_dir + hospital + '.csv')
        ID = seg_filepath.split('/')[-1][:-12]

        pd_index = data[data['WSIs'].isin([ID])].index.values[0]
        T = data['death_time'][pd_index]
        O = data['death_status'][pd_index]
        O = torch.tensor(O).type(torch.FloatTensor)
        T = torch.tensor(T).type(torch.FloatTensor)
        data_list.append({"features": features, "death_time": T, "death_status": O, "images_filepath":seg_filepath})
    return data_list


def get_files(path, rule=".pt"):
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
  
    info_list = list(info_df['WSIs'])
    for i in seg_list:
        filename = os.path.splitext(os.path.basename(i))[0]
        filename = filename.replace('_features', '')
        # print(filename)
        if filename in info_list:
            cleaned_path.append(i) 
    return cleaned_path
    

def filter_values(risk_pred_all, censor_all, survtime_all, file_path_all, wsis_values):
    risk_pred_filtered, censor_filtered, survtime_filtered, file_path_filtered = [], [], [], []
    for risk_pred, censor, survtime, file_path in zip(risk_pred_all, censor_all, survtime_all, file_path_all):
        filename = os.path.splitext(os.path.basename(file_path))[0][:-9]
        # print('filename:',filename)
        if filename in wsis_values:
            risk_pred_filtered.append(risk_pred)
            censor_filtered.append(censor)
            survtime_filtered.append(survtime)
            file_path_filtered.append(file_path)
    return np.array(risk_pred_filtered), np.array(censor_filtered), np.array(survtime_filtered), file_path_filtered


if __name__ == '__main__':
    micro_train= 'path/to/train/'
    micro_test = 'path/to/valid/'
    info_train = pd.read_csv('path/to/train.csv')
    info_val =  pd.read_csv('path/to/test.csv')


    Train_list = path_cleaning(micro_train,info_train)
    Val_list = path_cleaning(micro_test,info_val)
    train_data_list = prepare_data_list(Train_list)
    val_data_list = prepare_data_list(Val_list)


    train_dataset = CacheDataset(data=train_data_list, cache_num=10, num_workers=0)
    val_dataset = CacheDataset(data=val_data_list,  cache_num=10, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0)

    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    
    for epoch in tqdm(range(EPOCH)):
        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])    
        file_path_all = []  
        loss_epoch = 0
        print('train_model_before_weight')
        print(list(model.parameters())[-1])
        for batch_idx, batch in enumerate(train_loader):
            graph_data = batch['features'].to(device)
           
            survtime = batch['death_time'].to(device)
            censor = batch['death_status'].to(device)
            pred = model(graph_data)
           
            loss_cox = CoxLoss(survtime, censor, pred, device)
            loss = LAMBDA_COX * loss_cox 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()           
            loss_epoch += loss.item()           

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
            file_path_all += batch['images_filepath']


        scheduler.step(loss_epoch)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}, Learning Rate: {lr}, Loss: {loss_epoch / len(train_loader.dataset)}')

        max_cindex = 0
        best_threshold = 0
        cindex_epoch_train = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        pvalue_epoch_train  = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_epoch_train  = accuracy_cox(risk_pred_all, censor_all)
        
        model.eval()
        file_path_all = [] 
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])  
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                loss_test = 0
                graph_data = batch['features'].to(device)

                survtime = batch['death_time'].to(device)
                censor = batch['death_status'].to(device)

                pred = model(graph_data)
                loss_cox = CoxLoss(survtime, censor, pred, device)
                loss = LAMBDA_COX * loss_cox

                risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
                survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
                file_path_all += batch['images_filepath']                
        cindex_test_total = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        pvalue_test_total = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_test_total = accuracy_cox(risk_pred_all, censor_all)

    
        save_path =  'path/to/metric_uni/'
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


