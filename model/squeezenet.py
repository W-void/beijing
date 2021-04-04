import torch.nn as nn
from torchvision.models import squeezenet1_1


class squeezenet(nn.Module):
    def __init__(self):
        super(squeezenet, self).__init__()
        self.net = squeezenet1_1(pretrained=False)
        self.net.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2))
        self.net.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = torch.squeeze(self.net(x))
        return x