# %%
import os
import numpy as np
from osgeo import gdal

# %%
root_tif = 'D:/landsat/bjtif/'
root_qa = 'D:/landsat/bjqa/'
fileNames = os.listdir(root_qa)
fileNames = [root_qa + f for f in fileNames]

#%%
fileName = fileNames[1]
tif = gdal.Open(fileName).ReadAsArray()
qa = tif[3000:4000, 3000:4000]

# %%
import cv2 as cv

cloud_flag = np.where(qa >= 4096, 1, 0)
cv.namedWindow('cloud', 0)
cv.imshow('cloud', cloud_flag)
k=cv.waitKey(0)

if k == 27:
    cv.destroyAllWindows()
elif k == ord('s'):
    cv.imwrite('cloud.png',cloud_flag)
    cv.destroyAllWindows()

# %%
