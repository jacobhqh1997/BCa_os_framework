import numpy as np

def calculate_immune_score(prob_map_path):

    prob_map_immune = np.load(prob_map_path)
    pred_map = np.zeros((prob_map_immune.shape[0],prob_map_immune.shape[1]))
    for i in range(prob_map_immune.shape[0]):
        for j in range(prob_map_immune.shape[1]):
            pred_map[i][j] = np.argmax(prob_map_immune[i,j,:])

            if prob_map_immune[i,j,0] == 0 and pred_map[i][j] == 0:
                pred_map[i][j] = 1    
                
    pred_map_copy = pred_map.copy()

    pred_map_copy[(pred_map == 1) | (pred_map == 3)] = -1

    num_immune = (pred_map_copy == 6).sum()
    num_remaining = (pred_map_copy != -1).sum()

    immune_score = num_immune / num_remaining if num_remaining != 0 else 0

    return immune_score


prob_map_path = "path/to/your/prob_map.npy"  
cell_size = 10  


IFS_score = calculate_immune_score(prob_map_path, cell_size)

print(f"IFS_score: {IFS_score}")