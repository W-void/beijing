#%%
import torch
from torch import nn
import torchvision as ptv
import torch.optim as optim
import numpy as np
from time_cut import find_common_name


class MLP(nn.Module):
    def __init__(self, n_feature):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(n_feature, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, din):
        dout = nn.functional.relu(self.fc1(din))
        dout = nn.functional.relu(self.fc2(dout))
        return nn.functional.sigmoid(self.fc3(dout))

model = MLP(10).cuda()
print(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3)