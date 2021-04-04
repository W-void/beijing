#%%
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import gdal

#%%
tif = gdal.Open('D:/beijing/bjtif/LC08_L1TP_123032_20130613_20170504_01_T1_sr_band1-7.tif').ReadAsArray(3500, 4000, 500, 500)
img = np.clip(np.transpose(tif[1:4], (1, 2, 0))*5e-4, 0, 1)
segments = slic(img, n_segments=60, compactness=10)
out=mark_boundaries(img, segments)
plt.imshow(out)
plt.show()
# %%
