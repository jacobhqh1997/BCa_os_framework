import numpy as np
import pandas as pd
import os
def TIM_Score(prob_map_path, cell_size):
    # prob_map: MxNx8 numpy array contains the probabilities
    # cell_size: number of patch to be consider as one grid-cell
    prob_map = np.load(prob_map_path)
    pred_map = np.zeros((prob_map.shape[0],prob_map.shape[1]))
    for i in range(prob_map.shape[0]):
        for j in range(prob_map.shape[1]):
            pred_map[i][j] = np.argmax(prob_map[i,j,:])
            # print(multi_classes[i][j])
            if prob_map[i,j,0] == 0 and pred_map[i][j] == 0:
                pred_map[i][j] = 1
    T = np.int8(pred_map == 0)  # patches predicted as tumour
    M = np.int8(pred_map == 4)  # patches predicted as musele
    [rows, cols] = T.shape
    stride = np.int32(cell_size / 2)
    t = np.zeros(len(range(0, rows - cell_size + 1, stride))*len(range(0, cols - cell_size + 1, stride)))
    m = np.zeros(len(range(0, rows - cell_size + 1, stride))*len(range(0, cols - cell_size + 1, stride)))
    k = 0
    # probability of tumour and lymphocytes/necrosis in each grid cell
    for i in range(0, rows - cell_size + 1, stride):
        for j in range(0, cols - cell_size + 1, stride):
            t[k] = np.mean(np.mean(T[i:i + cell_size, j:j + cell_size]))
            m[k] = np.mean(np.mean(M[i:i + cell_size, j:j + cell_size]))
            k += 1
    # removing grid cells with no tumour and lymphocytes/necrosis regions
    index = np.logical_and(t == 0, m == 0)
    index = np.where(index)[0]
    t = np.delete(t, index)
    m = np.delete(m, index)
    tim_score = 0.0
    coloc_score = 0.0 
    if len(t) == 0:  # ideally each WSI should have some tumour or lymphocyte region to get a sensible TILAb-score 
        tim_score = 0  # if there is no tumour then its good for patients long term survival
    else:
        # normalizaing the percentage of tumour and lymphocyte range to [0-1] in a grid-cell
        t = t/(t + m)
        m = m/(t + m)
        # Morisita-Horn Index based colocalization socre
        coloc_score = (2 * sum(t*m)) / (sum(t**2) + sum(m**2))
        if np.sum(t) == 0:
            tim_score = 1 # when only lymphocytes are present
        else:
            l2t_ratio = np.sum(m) / np.sum(t)  # lymphocyte to tumour ratio
            tim_score = 0.5 * coloc_score * l2t_ratio  
    return tim_score,coloc_score

def calculate_tumor_score(prob_map_path):
   
    prob_map = np.load(prob_map_path)
    pred_map = np.zeros((prob_map.shape[0],prob_map.shape[1]))
    for i in range(prob_map.shape[0]):
        for j in range(prob_map.shape[1]):
            pred_map[i][j] = np.argmax(prob_map[i,j,:])
            # print(multi_classes[i][j])
            if prob_map[i,j,0] == 0 and pred_map[i][j] == 0:
                pred_map[i][j] = 1    
                
    pred_map_copy = pred_map.copy()

   
    pred_map_copy[(pred_map == 1) | (pred_map == 3)] = -1

    
    num_tumor = (pred_map_copy == 0).sum()
    num_remaining = (pred_map_copy != -1).sum()

   
    tumor_score = num_tumor / num_remaining if num_remaining != 0 else 0

    return tumor_score



prob_map_path = "path/to/your/prob_map.npy"  
cell_size = 10  

tim_score, coloc_score = TIM_Score(prob_map_path, cell_size)
TFS_score = calculate_tumor_score(prob_map_path, cell_size)

IMTS_score =tim_score*TFS_score*100



