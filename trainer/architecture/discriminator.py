import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv256_128 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv128_64 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv64_32 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv32_16 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_layer = nn.Conv2d(512, 1, kernel_size=4, padding=1)
    
    def forward(self, x):
        out = self.conv256_128(x)
        out = self.conv128_64(out)
        out = self.conv64_32(out)
        out = self.conv32_16(out)
        out = self.out_layer(out)
            
        # Patch GAN
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
        return out