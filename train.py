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


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = model.vgg11_bn().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-3)
modelName = 'CAM'

#%%
for epo in range(2): 
    net.train()
    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = net(imgs)
        loss = criterion(output.flatten(), labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_loss = loss.item()
        if np.mod(i, 15) == 0:
            acc = metrics.accuracy_score(labels.cpu(), output.cpu() > 0.5)
            auc = metrics.roc_auc_score(labels.cpu(), output.detach().cpu())
            print('epoch {}, {:03d}/{},train loss is {:.4f}, acc is {:.4f}, auc is {:.4f}'.format(epo, i, len(dataloader), iter_loss, acc, auc), end="\n")
        if np.mod(i+1, 100) == 0:
            savePath = './checkpoints/'
            torch.save(net, savePath + modelName + '_{}.pt'.format(i))
