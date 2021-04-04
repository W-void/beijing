#%%
import os
import numpy as np
from osgeo import gdal
from glob import glob
import matplotlib.pyplot as plt

#%%
img_path = 'D:/beijing/bjtif/LC08_L1TP_123032_20130613_20170504_01_T1_sr_band1-7.tif'
img = gdal.Open(img_path).ReadAsArray()
print(img.shape)

#%%
img = img.reshape((7, -1))
idx = np.where(np.sum(img <= 0, 0) == 0)

#%%
for channel in img:
    c = channel[idx]
    plt.hist(c, bins=100)
    plt.show()

# %%
rb = np.clip(img[3][idx] / img[1][idx], 0.5, 2.5) # 必须要clip，否则直方图会是一条线
plt.hist(rb, bins=100)
plt.show()

# %%
