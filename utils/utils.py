'''
Author: wang shuli
Date: 2020-12-25 09:53:29
LastEditTime: 2021-01-15 13:16:34
LastEditors: your name
Description: 
'''
import os
import re
import glob
import random
import cv2
from osgeo import gdal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# import multiprocessing  # 解决VSCode对多线程支持不好的问题
# multiprocessing.set_start_method('spawn',True)

transform = transforms.Compose([
    transforms.Resize((256, 256))
])

class BagDataset(Dataset):

    def __init__(self, transform):

        self.root = 'D:/data/wdcd/training'
        self.img_dirs = self.get_img_dirs(self.root)
        self.transform = transform

   
    def get_img_dirs(self, image_dir):
        
        # load background images
        cloud_img_dirs = glob.glob(os.path.join(image_dir, 'cloud/*.tiff'))
        non_cloud_img_dirs = glob.glob(os.path.join(image_dir, 'non-cloud/*.tiff'))
        m_tr_imgs = max(len(cloud_img_dirs), len(non_cloud_img_dirs))

        cloud_img_dirs = cloud_img_dirs[:m_tr_imgs]
        non_cloud_img_dirs = non_cloud_img_dirs[:m_tr_imgs]

        return list(zip(cloud_img_dirs, non_cloud_img_dirs))
        
 
    def __len__(self):
        
        return len(self.img_dirs)


    def __getitem__(self, idx):

        def readTif(fileName):

            im_data = torch.from_numpy(gdal.Open(fileName).ReadAsArray() / 255)
            im_data = self.transform(im_data)

            return im_data 
        
        cloud_img_dir, non_cloud_img_dir = self.img_dirs[idx]
        cloud_img = readTif(cloud_img_dir)
        non_cloud_img = readTif(non_cloud_img_dir)

        return cloud_img, non_cloud_img



if __name__ =='__main__':

    train_dataset = BagDataset(transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)

    for train_batch in train_dataloader:
        print(train_batch[0].shape, train_batch[0].dtype)
        print(train_batch[1].shape, train_batch[1].dtype)
        break