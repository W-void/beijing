'''
Author: wang shuli
Date: 2020-12-25 09:41:18
LastEditTime: 2021-01-15 20:41:37
LastEditors: your name
Description: 
'''

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import utils

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=4, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--save_path", type=str, default="checkpoints", help="model save dir")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def conv_block(in_filters, out_filters, sample=None):
 
            block = []
            
            if sample == 'down':
                block += [nn.MaxPool2d(2)]
            elif sample == 'up':
                block += [nn.Upsample(scale_factor=2, mode='bilinear')]
            
            for _ in range(2):
                block += [nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.Tanh()]
                in_filters = out_filters

            block = nn.Sequential(*block)
            return block  

        self.filter_num = [16, 32, 64, 128]
        self.net_a_1 = conv_block(opt.channels, self.filter_num[0])
        self.net_a_2 = conv_block(self.filter_num[0], self.filter_num[1], 'down')
        self.net_a_3 = conv_block(self.filter_num[1], self.filter_num[2], 'down')
        self.net_a_4 = conv_block(self.filter_num[2], self.filter_num[3], 'down')
        self.net_1 = conv_block(self.filter_num[3], self.filter_num[2], 'up')
        self.net_2 = conv_block(self.filter_num[2], self.filter_num[1], 'up')
        self.net_3 = conv_block(self.filter_num[1], self.filter_num[0], 'up')
        
        self.reflectance = nn.Sequential(
            nn.Conv2d(self.filter_num[0], opt.channels, 1),
            nn.Sigmoid()
        )
        self.alpha = nn.Sequential(
            nn.Conv2d(self.filter_num[0], 1, 1),
            nn.Sigmoid()
        )
        # self.mask = nn.Sequential(
        #     nn.Conv2d(self.filter_num[0], opt.channels, 3),
        # )

    def forward(self, img):

        a_1 = self.net_a_1(img)
        a_2 = self.net_a_2(a_1)
        a_3 = self.net_a_3(a_2)
        a_4 = self.net_a_4(a_3)
        net_1 = self.net_1(a_4)
        net_1 = net_1 + a_3
        net_2 = self.net_2(net_1)
        net_2 = net_2 + a_2
        net_3 = self.net_3(net_2)
        net_3 = net_3 + a_1
        refle_ = self.reflectance(net_3)
        alpha_ = self.alpha(net_3)

        return refle_, alpha_


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss functions
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
transform = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size))
])
dataset = utils.BagDataset(transform)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    
    mm = 1

    for i, (clouds, nonclouds) in enumerate(dataloader):

        batch_size = opt.batch_size

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        clouds = Variable(clouds.type(FloatTensor))
        nonclouds = Variable(nonclouds.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------
        for _ in range(mm):
            optimizer_G.zero_grad()

            # Generate a batch of images
            refle, alpha = generator(clouds)
            gen_cloud_imgs = refle + alpha * nonclouds
            gen_noncloud_mask = torch.where(alpha > 0.8)
            gen_noncloud_imgs = torch.where(alpha > 0.8, clouds/alpha, FloatTensor([0]))

            # Loss measures generator's ability to fool the discriminator
            validity_cloud = discriminator(gen_cloud_imgs)
            validity_noncloud = discriminator(gen_noncloud_imgs)
            g_loss = 0.5 * (adversarial_loss(validity_cloud, valid) + adversarial_loss(validity_noncloud, fake))

            g_loss.backward()
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real = discriminator(nonclouds)
        nonclouds_rmv = torch.where(alpha > 0.8, nonclouds, FloatTensor([0]))
        real_rmv = discriminator(nonclouds_rmv)
        d_real_loss = 0.5 * (adversarial_loss(real, valid) + adversarial_loss(real_rmv, valid))
        
        # Loss for fake images
        fake_cloud= discriminator(gen_cloud_imgs.detach())
        fake_noncloud = discriminator(gen_noncloud_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_cloud, fake) + adversarial_loss(fake_noncloud, fake)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        mm = 5 if d_loss < 0.4 else 1

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        batches_done = epoch * len(dataloader) + i + 1
        if batches_done % opt.sample_interval == 0:
            img = torch.cat((clouds[:, :3]/2, refle[:, :3], alpha.repeat(1, 3, 1, 1), gen_cloud_imgs), dim=0)
            save_image(img.data, 'images/%d_cloud.png' % batches_done, nrow=batch_size)
            
            torch.save(generator, os.path.join(opt.save_path, '{}_Gen.pt'.format(batches_done)))
            torch.save(discriminator, os.path.join(opt.save_path, '{}_Dis.pt'.format(batches_done)))
            print('saveing ' + os.path.join(opt.save_path, '{}.pt'.format(batches_done)))