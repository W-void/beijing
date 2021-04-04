import math
import torch
import torch.nn as nn
from .utils import DoubleConv


class ssnet(nn.Module):

    def __init__(self, n_channels=10, n_classes=1, init_weights=True):
        super(ssnet, self).__init__()
        self.cfg = [32, 64, 128, 64, 32]
        self.in_channels = n_channels
        self.num_classes = n_classes
        self._make_layers()
        

        self.down_1 = nn.MaxPool2d(2)
        self.down_2 = nn.MaxPool2d(2)
        self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=2*self.cfg[-1], out_channels=self.num_classes, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        st = self.conv_layers['spect'][0](x) 
        sp = self.conv_layers['space'][0](x)
        
        st_ = torch.cat([st, sp], 1)
        sp_ = torch.cat([self.down_1(sp), self.down_up_ss['down'][0](st)], 1)
        st = self.conv_layers['spect'][1](st_)
        sp = self.conv_layers['space'][1](sp_)

        st_ = torch.cat([st, self.down_up_ss['up'][0](sp)], 1)
        sp_ = torch.cat([self.down_2(sp), self.down_up_ss['down'][1](st)], 1)
        st = self.conv_layers['spect'][2](st_)
        sp = self.conv_layers['space'][2](sp_)

        st_ = torch.cat([st, self.down_up_ss['up'][1](sp)], 1)
        sp_ = torch.cat([self.up_1(sp), self.down_up_ss['down'][2](st)], 1)
        st = self.conv_layers['spect'][3](st_)
        sp = self.conv_layers['space'][3](sp_)

        st_ = torch.cat([st, self.down_up_ss['up'][2](sp)], 1)
        sp_ = torch.cat([self.up_2(sp), st], 1)
        st = self.conv_layers['spect'][4](st_)
        sp = self.conv_layers['space'][4](sp_)
        
        x = self.classifier(torch.cat([st, sp], 1))
        return torch.squeeze(x, dim=1)

    def _make_layers(self):
        in_channels = self.in_channels
        spect_layers, space_layers = [], []
        for c in self.cfg:
            spect_layers.append(DoubleConv(in_channels, c, 1))
            space_layers.append(DoubleConv(in_channels, c, 3))
            in_channels = c * 2
        spect_layers = nn.ModuleList(spect_layers)
        space_layers = nn.ModuleList(space_layers)
        self.conv_layers = nn.ModuleDict({'spect':spect_layers, 'space':space_layers})
        
        down_ss, up_ss = [], []
        for scale in [2, 4, 2]:
            down_ss.append(nn.MaxPool2d(scale))
            up_ss.append(nn.Upsample(scale_factor=scale, mode='bilinear'))
        down_ss = nn.ModuleList(down_ss)
        up_ss = nn.ModuleList(up_ss)
        self.down_up_ss = nn.ModuleDict({'down':down_ss, 'up':up_ss})

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    net = ssnet()
    print(net)
