import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def cut_empty(matrix_path, save_base_dir):
    matrix = np.load(matrix_path)
    empty = matrix[:,:,1]
    coor = np.where(empty<1)
    top = min(coor[0])
    bottom = max(coor[0])
    left = min(coor[1])
    right = max(coor[1])
    matrix = matrix[top:bottom,left:right]

    height, width, _ = matrix.shape
    new_size = 0

    if height >= width:
        new_size = height+8
        new_matrix = np.zeros((new_size, new_size,8))
        new_matrix[:,:,1] = 1
        dealt_width = int((height-width)/2)
        new_matrix[4:-4, dealt_width+4:width+(dealt_width+4)] = matrix
    else:
        new_size = width+8
        new_matrix = np.zeros((new_size, new_size,8))
        new_matrix[:,:,1] = 1
        dealt_height = int((width-height)/2)
        new_matrix[dealt_height+4:height+(dealt_height+4),4:-4] = matrix
  
    save_path = os.path.join(save_base_dir, '{}.npy'.format(os.path.basename(matrix_path)[:-4]))
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    np.save(save_path, new_matrix)


def get_files(path, rule=".npy"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all


if __name__ == '__main__':


    TCGA_path = 'path/to/org'

    TCGA = get_files(TCGA_path)

    save_base_dir = 'path/to/Macro'
    num_processes = multiprocessing.cpu_count()
    for i in TCGA:
        cut_empty(i,save_base_dir)






