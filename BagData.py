import os
import re
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
import cv2
from osgeo import gdal
# from gdalconst import *
from sklearn.preprocessing import OneHotEncoder
# import multiprocessing  # 解决VSCode对多线程支持不好的问题
# multiprocessing.set_start_method('spawn',True)


transform = transforms.Compose([
    transforms.ToTensor() # totensor 会改变shape！！！
    , transforms.Normalize(
        mean=[0.04654*2, 0.04435*2, 0.04013*2, 0.04112*2, 0.04776*2, 0.02371*2, 0.01906*2, 0.0038*2, 0.1909*2, 0.17607*2], 
        std=[1370*16e-6, 1414*16e-6, 1385*16e-6, 1488*16e-6, 1522*16e-6, 998*16e-6, 821*16e-6, 292*16e-6, 2561*16e-6, 2119*16e-6]
        # mean=[0.04435*2, 0.04013*2, 0.04112*2],
        # std=[1414*16e-10, 1385*16e-10, 1488*16e-10]
        )
])

senceList = ["Barren", "Forest", "Grass/Crops","Shrubland", "Snow/Ice", "Urban", "Water", " Wetlands"]
f = open('./dataLoad/urls.txt', "r")
lines = f.readlines()
senceDict = {}
for i, line in enumerate(lines):
    senceId = re.split('[./]', line)[-3]
    senceDict[senceId] = i//12


class BagDataset(Dataset):

    def __init__(self, tr='train', transform=transform, grep=-1):
        self.transform = transform
        self.type = tr
        self.root = 'D:/landsat8data/'
        self.imgPath = self.root + self.type + '/image/'
        self.maskPath = self.root + self.type + '/label/'
        self.qaPath = self.root + self.type + '/image_qa/'
        self.imgFiles = os.listdir(self.imgPath)
        if grep != -1:
            self.imgFiles = [i for i in self.imgFiles if senceDict[i.split('_')[0]] == grep]
        
    def __len__(self):
        return len(self.imgFiles)

    def readTif(self, fileName):
        im_data = gdal.Open(fileName).ReadAsArray()
        return im_data #[1:4]

    def __getitem__(self, idx):
        # img_name = '%05d'%idx
        # img = self.readTif(self.imgPath+img_name+'.tiff')
        # label = cv2.imread(self.maskPath+img_name+'.png', 0) # 灰度图
        img = self.readTif(self.imgPath + self.imgFiles[idx])
        label = cv2.imread(self.maskPath + self.imgFiles[idx][:-4]+'png', 0)
        qa = cv2.imread(self.qaPath + self.imgFiles[idx][:-4]+'png', 0)
        # 调整
        label = label > 128
        qa = qa > 128
        # label = torch.FloatTensor(label)
        #print(imgB.shape)
        if self.transform:
            img = self.transform(img.transpose(1,2,0) * 2e-5)   
        # print(img.shape, label.shape)
        img = img.float()
        label = torch.tensor(label, dtype=torch.float)
        qa = torch.tensor(qa, dtype=torch.float)
        return self.imgFiles[idx], img, label

# bag = BagDataset(transform)
# train_size = int(0.6 * len(bag))
# test_size = len(bag) - train_size
# train_dataset, test_dataset = random_split(bag, [train_size, test_size])
train_dataset = BagDataset(tr='train', transform=transform)
test_dataset = BagDataset(tr='val', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
# all_dataloader = DataLoader(bag, batch_size=4, shuffle=False, num_workers=4)

if __name__ =='__main__':
    # for i, batch in enumerate(all_dataloader):
    #     if torch.any(torch.isnan(batch[0])):
    #         print("NO.{} have nan !!!".format(i))

    for i in range(2):
        print(i)
        for train_batch in train_dataloader:
            continue
            print(train_batch[0])
            print(train_batch[1].shape)
            print(train_batch[2].shape)

        for test_batch in test_dataloader:
            # continue
            # print(test_batch[0])
            # print(test_batch[1].shape)
            print(test_batch[2].shape)

