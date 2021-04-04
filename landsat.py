# %%
import os
import re
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from BagData import test_dataloader, train_dataloader
from model import ssnet


# %%
def load_checkpoint(net, net_pretrained=None):
    if net_pretrained == None:
        # net.apply(weights_init)
        return net
    else:
        net_dict = net.state_dict()
        net_pretrained_dict = net_pretrained.state_dict()
        # pretrained_dict = {k: v for k, v in net_pretrained_dict.items() if k[:3] == 'inc'}
        pretrained_dict = {k: v for k, v in net_pretrained_dict.items() if k in net_dict.keys()}
        # pretrained_dict.pop('outc.conv.weight')
        # pretrained_dict.pop('outc.conv.bias')
        print('Total : {}, update: {}'.format(len(net_pretrained_dict), len(pretrained_dict)))
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        for d, p in zip(net_dict, net.parameters()):
            if(d[:3] == 'inc'):
                p.requires_grad = False

        print("loaded finished!")
        return net

# %%
senceList = ["Barren", "Forest", "Grass/Crops","Shrubland", "Snow/Ice", "Urban", "Water", "Wetlands"]
def read_list(path='./dataLoad/urls.txt'):
    f = open(path, "r")
    lines = f.readlines()
    senceDict = {}
    for i, line in enumerate(lines):
        senceId = re.split('[./]', line)[-3]
        senceDict[senceId] = i//12
    return senceDict

# %%
def train(epo_num=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_pretrained = torch.load('./checkpoints/ssnet_0.pt')
    net = ssnet(10, 1)
    modelName = 'ssnet'
    
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
    net = load_checkpoint(net, net_pretrained)
    # print(net.state_dict().keys())
    net = net.to(device)
    net = net.float()
    criterion = nn.BCELoss().to(device)
    optimizer = optim.RMSprop(net.parameters(), lr=1e-4, momentum=0.9)

    all_train_iter_loss = []
    all_test_iter_loss = []
    result = []
    global_step = 0
    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):
        train_loss = 0
        all_recall = 0.
        all_precision = 0.
        net.train()
        for index, (_, bag, bag_msk) in enumerate(train_dataloader):
            bag = bag.to(device)
            bag_msk = bag_msk.to(device)
            output = net(bag)
            regularization_loss = 0
            loss = criterion(output, bag_msk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_loss = loss.item()

            if np.mod(index, 15) == 0:
                print('epoch {}, {:03d}/{},train loss is {:.4f}'.format(epo, index, len(train_dataloader), iter_loss), end="\n        ")

        if np.mod(epo+1, 1) == 0:
            savePath = './checkpoints/'
            if not os.path.exists(savePath):
                os.makedirs(savePath)

            torch.save(net, savePath + modelName + '_{}.pt'.format(epo))
            print('saveing ' + savePath + modelName + '_{}.pt'.format(epo))

        test_loss = 0
        all_recall_test = 0.
        all_precision_test = 0.
        evaluateArray = np.zeros((4))
        senceDict = read_list()
        predEvalArray = np.zeros((8, 5))
        net.eval()
        with torch.no_grad():
            for index, (names, bag, bag_msk) in enumerate(test_dataloader):
                bag = bag.to(device)
                optimizer.zero_grad()
                output = net(bag)
                outputData = output.data.cpu()
                outputData = torch.where(outputData > 0.5, 1, 0)

                acc_test, recall_test, precision_test = get_acc_recall_precision(evaluateArray, bag_msk.data, outputData)

                if np.mod(index, 15) == 0:
                    print('epoch {}, {:03d}/{}'.format(epo, index, len(test_dataloader)), end="        ")
                    print('acc: {:.4}, recall: {:.4}, precision: {:.4}, f-score: {:.4f}'.format(acc_test/(index+1), 
                        recall_test, precision_test, 2*(recall_test*precision_test)/(recall_test+precision_test)))

                for idx, name in enumerate(names):
                    senceId = re.split('[_]', name)[0]
                    y = bag_msk.data[idx].cpu()
                    y_ = outputData[idx]
                    tmpList = evaluate(y, y_)
                    predEvalArray[senceDict[senceId]] += np.array(tmpList)


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        np.save('./log/' + modelName + '_{}.npy'.format(epo), predEvalArray)
        npy2tex(predEvalArray)

        print('epoch test acc: {:.4}, recall: {:.4}, precision: {:.4}, f-score: {:.4f}'.format(acc_test/(index+1), 
                        recall_test, precision_test, 2*(recall_test*precision_test)/(recall_test+precision_test)))
        
        print('time: %s'%(time_str))

#%%
def get_acc_recall_precision(arr, y, y_):
    arr[0] += y.sum()
    arr[1] += y_.sum()
    arr[2] += (y * y_).sum()
    arr[3] += (y == y_).sum().to(torch.float64) / (256*256) / y.shape[0]
    recall = arr[2] / arr[0]
    precision = arr[2] / arr[1]
    return arr[3], recall, precision

def evaluate(y, y_):
    correction = (y_ * y).sum()
    sumY_ = y_.sum()
    sumY = y.sum()
    sumEqual = (y_ == y).sum()
    acc_ = sumEqual.to(torch.float64) / (256*256)
    return [1, acc_, correction, sumY, sumY_]

def npy2tex(arr):
    # arr = np.load(path)
    arr = arr[:8]
    arr = arr.T
    l = []
    l.append([a for a in arr[1]/arr[0]] + [arr[1].sum()/arr[0].sum()]) #acc
    l.append(np.array([a for a in arr[2]/arr[3]] + [arr[2].sum()/arr[3].sum()])) #rec.
    l.append([a for a in arr[2]/arr[4]] + [arr[2].sum()/arr[4].sum()]) #prec.
    l.append(a for a in 2*l[1]*l[2]/(l[1]+l[2])) #f1    
    name = ['Acc  ', 'Rec. ', 'Prec.', 'F1   ']
    for i, eval in enumerate(l):
        print('&%s'%name[i], end=' ')
        for e in eval:
            print('&%.2f'%(e*100), end=' ')
        print("\\\\")

# %%
if __name__ == "__main__":
    train()
