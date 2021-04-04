#%%
import os 
import random
import numpy as np
from glob import glob
from osgeo import gdal
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

#%%
net = torch.load('./checkpoints/CAM_6399.pt')
net.eval()

# %%
vals = glob('./WDCD/validation/image/*.tiff')
img_size = 250
for val in vals:
    tif = gdal.Open(val).ReadAsArray()
    