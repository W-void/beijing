#%%
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from dataloader import dataloader
import model
from sklearn import metrics
from osgeo import gdal

#%%
net = torch.load('./checkpoints/CAM_6399.pt')
net.eval()

input_img = []
features_blobs = []
def hook_feature(module, input, output):
    input_img.append(input[0][:, :3].data.cpu().numpy())
    features_blobs.append(output.data.cpu().numpy())
net._modules.get('features_conv').register_forward_hook(hook_feature)

params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())


#%%
def returnCAM(feature_conv, weight_softmax):
    size_upsample = (250, 250)
    nc, h, w = feature_conv.shape
    cam = weight_softmax @ feature_conv.reshape((nc, h * w))
    cam = cam.reshape(h, w) 
    cam_img = (cam - cam.min()) / (cam.max() - cam.min())  
    cam_img = np.uint8(255 * cam_img)
    output_cam = cv2.resize(cam_img, size_upsample)
    return output_cam


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i, (imgs, labels) in enumerate(dataloader):
    output = net(imgs.to(device))
    for idx in range(imgs.shape[0]):
        print('labes:%d, predict:%f'%(labels[idx].data,output[idx].data))
        CAMs = returnCAM(features_blobs[0][idx], weight_softmax)
        b, c, width, height = imgs.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs, (width, height)), cv2.COLORMAP_JET)
        result = heatmap / 255 * 0.2 + np.transpose(input_img[0][idx], (1, 2, 0)) * 0.8
        cv2.namedWindow('heatmap', 0)
        cv2.imshow('heatmap', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    break
# %%
