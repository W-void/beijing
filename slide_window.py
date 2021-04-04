# %%
import os
import cv2 as cv
import numpy as np
from osgeo import gdal

# %%
def slideWindow(root):
    seq_win_size = 10
    spa_win_size = 3
    fileNames = os.listdir(root)
    fileNames = [root + f for f in fileNames]

    for f in fileNames:
        tif = gdal.Open(f).ReadAsArray()

#%%
seq_win_size = 5
assert len(tif_list) > seq_win_size
filter_d = 5
sigma_color = 5
sigma_space = 5
seq_data = np.zeros((seq_win_size, bands, mask_row, mask_col), np.float32)
seq_mean_sum = np.zeros((bands, mask_row, mask_col), np.float32)
seq_squa_sum = np.zeros((bands, mask_row, mask_col), np.float32)

tif = tif.astype(np.float32)
seq_data[k % seq_win_size] = tif
seq_mean_sum += tif
seq_squa_sum += tif * tif

if k >= seq_win_size:
    # seq_std = np.std(seq_data, 0) # 计算时序维的标准差
    # seq_median = np.median(seq_data, 0) # [c, w, h]
    seq_mean = seq_mean_sum / seq_win_size
    seq_std = np.sqrt(seq_squa_sum / seq_win_size - seq_mean * seq_mean)
    flag = np.sum(tif > seq_mean + seq_std) # [w, h]
    cloud = np.where(flag > bands/2, np.int8(1), np.int8(0))
    cv.namedWindow('cloud')
    cv.imshow('cloud', cloud[::10, ::10])
    cv.waitKey(0)
    cv.destroyAllWindows()