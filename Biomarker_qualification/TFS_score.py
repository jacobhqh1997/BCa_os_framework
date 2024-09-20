import numpy as np

def calculate_tumor_score(prob_map_path):

    prob_map = np.load(prob_map_path)
    pred_map = np.zeros((prob_map.shape[0],prob_map.shape[1]))
    for i in range(prob_map.shape[0]):
        for j in range(prob_map.shape[1]):
            pred_map[i][j] = np.argmax(prob_map[i,j,:])
           
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


TFS_score = calculate_tumor_score(prob_map_path, cell_size)

print(f"IFS_score: {TFS_score}")