from __future__ import print_function
from __future__ import division
from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import albumentations as A
from monai.transforms import Compose, RandGaussianNoise, RandRotate90, RandGaussianSmooth, ScaleIntensity, RandAdjustContrast, ToTensor,Transpose
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Transform
from sklearn.metrics import confusion_matrix
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.data import CacheDataset
from sklearn.metrics import roc_auc_score


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss   


class AlbumentationsTransform(Transform):
    """
    Wrap Albumentations transforms to be used in MONAI workflows.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        # Apply the transform and return the modified data
        # Assuming the input data is a dictionary with an 'image' key
        image = data['image']
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array but got {type(image)}")
        transformed = self.transform(image=image)
        data['image'] = transformed['image']
        return data
    
def prepare_data_list(images_filepaths, lv_multiplier):
    data_list = []
    for image_filepath in images_filepaths:
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            print(f"Failed to read image at {image_filepath}")
            continue
        label_name = os.path.normpath(image_filepath).split(os.sep)[-2]
        label_map = {"LV": 5, "EMP": 1, "Stroma": 2, "adipose": 7, "Immune_cells": 6, "Tumor": 0, "non-ROI": 3, "Muscularis": 4}
        multiplier = lv_multiplier if label_name == "LV" else 1
        for _ in range(multiplier):
            data_list.append({"image": image, "label": label})
    return data_list


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    early_stopping = EarlyStopping(patience=30, verbose=True,path='path/to/checkpoint8.pt')
    val_acc_history = []
    red_lr = ReduceLROnPlateau(optimizer, patience=5, verbose=1, factor=0.8)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            if early_stopping.early_stop:
                print("Early stopping")
                break                

            # Iterate over data.
            for batch_data in dataloaders[phase]:
                inputs = batch_data['image'].to(device)
                labels = batch_data['label'].to(device)


                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_auc = roc_auc_score(all_labels, all_preds)
            if phase == 'val':
                early_stopping(epoch_loss, model)  # Consider using AUC for early stopping if applicable
                red_lr.step(epoch_loss)

            print('{} Loss: {:.4f} AUC: {:.4f}'.format(phase, epoch_loss, epoch_auc))

            # deep copy the model
            if phase == 'val' and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_auc_history.append(epoch_auc)

            # if phase == 'val':
            #     early_stopping(epoch_loss, model)
            #     red_lr.step(epoch_loss)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val AUC: {:4f}'.format(best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, time_elapsed

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract):
    model_ft = models.resnext50_32x4d(pretrained=False)  
    weights_path = "path/to/weight.pth"
    model_ft.load_state_dict(torch.load(weights_path)) 
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    input_size = 128

    return model_ft, input_size


if __name__ == '__main__':
    import os 
    import glob
    def get_image_paths(directory):
        return glob.glob(os.path.join(directory, '**', '*.png'), recursive=True)   
    train_path = 'path/to/patch'
    val_path = 'path/to/valid'

    train_images_filepaths = get_image_paths(train_path)
    val_images_filepaths = get_image_paths(val_path)  

    train_list = prepare_data_list(train_images_filepaths, lv_multiplier=3)
    val_list = prepare_data_list(val_images_filepaths, lv_multiplier=1)    

    print(len(train_images_filepaths), len(val_images_filepaths))
    batch_size = 200   
    train_transform = A.Compose(
        [
            A.Resize(128, 128),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
            A.Blur(blur_limit=3),
            A.RandomRotate90(p = 1),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
            A.RandomBrightnessContrast(p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_transformss = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
    ])
    train_transforms =AlbumentationsTransform(transform=train_transformss)
    val_transforms = AlbumentationsTransform(transform=val_transformss)

    train_dataset = CacheDataset(data=train_list, transform=train_transforms, cache_num=10, num_workers=12)
    val_dataset = CacheDataset(data=val_list, transform=val_transforms,  cache_num=10, num_workers=12)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=8000, shuffle=False, num_workers=12)


    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    num_classes = 8
    num_epochs = 60
    feature_extract = False
    models_name = ["resnext50_32x4d"]
    time_list = []
    for model_name in models_name:
        print(model_name)
        model_ft, input_size = initialize_model(num_classes, feature_extract)
        dataloaders_dict = {
            'train': train_loader,
            'val': val_loader
        }
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = nn.DataParallel(model_ft, device_ids=[0])
        model_ft = model_ft.to(device)
        cudnn.benchmark = True
        params_to_update = model_ft.parameters()
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        model_ft, hist, time_elapsed = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
        nb_classes = 8
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        model_ft.eval()
        with torch.no_grad():
            for i,batch_data in enumerate(dataloaders_dict['val']):
                inputs = batch_data['image'].to(device)
                classes =  batch_data['label'].to(device)
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)
        save_path = 'path/to/save_results/{}_{}'.format(model_name,train_path.split('/')[-1][:-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        np.save(save_path + '/confusion_matrix-{}.npy'.format(model_name), confusion_matrix.numpy())
        print(confusion_matrix.diag()/confusion_matrix.sum(1))
        per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
        np.save(save_path + '/per_class_acc-{}.npy'.format(model_name), per_class_acc.numpy())

        val_acc_history = np.array([h.cpu().numpy() for h in hist])
        np.save(save_path + '/val_acc_history-{}.npy'.format(model_name), val_acc_history)

        torch.save(model_ft.module.state_dict(), save_path + '/8params-{}.pkl'.format(model_name),_use_new_zipfile_serialization=False)

        time_list.append(time_elapsed)
        print(time_list)
        np.save(save_path + '/time_list-{}.npy'.format(model_name), np.array(time_list))
