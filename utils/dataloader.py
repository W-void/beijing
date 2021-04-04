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
transform = transforms.Compose([
    transforms.ToTensor()
])

class BagDataset(Dataset):
    def __init__(self, root='./WDCD/training/'):
        self.root = root
        self.cloud_root = self.root + 'cloud/'
        self.non_cloud_root = self.root + 'non-cloud/'
        self.initialize()
    
    def __len__(self):
        return len(self.imgs_and_labels)

    def __getitem__(self, idx):
        img = self.readTif(self.imgs_and_labels[idx][0])
        img = transform(img.transpose((1, 2, 0)) * 1e-3)
        img = img.float()
        label = self.imgs_and_labels[idx][1]
        return img, label

    def initialize(self):
        cloud_imgs = glob(self.cloud_root+'*.tiff')
        non_cloud_imgs = glob(self.non_cloud_root+'*.tiff')
        imgs = cloud_imgs + non_cloud_imgs

        cloud_labels = [1] * len(cloud_imgs)
        non_cloud_labels = [0] * len(non_cloud_imgs)
        labels = cloud_labels + non_cloud_labels
        
        self.imgs_and_labels = list(zip(imgs, labels))
        random.shuffle(self.imgs_and_labels)

    def readTif(self, img_path):
        data = gdal.Open(img_path).ReadAsArray()
        return data


dataset = BagDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


if __name__ == '__main__':
    for train_batch in dataloader:
        imgs = train_batch[0]
        labels = train_batch[1]
        print(imgs.shape)  # [b, c, h, w] -> [b, 4, 250, 250]
        print(labels)
        print(imgs.dtype, labels.dtype)
        break
# %%
