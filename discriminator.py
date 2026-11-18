import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64):
        super().__init__()

        self.conv1 = nn.Conv2d(num_in_ch, num_feat, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_feat, num_feat * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=4, stride=2, padding=1)

        self.conv4 = nn.Conv2d(num_feat * 4, 1, kernel_size=4, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))  
        x = self.lrelu(self.conv2(x)) 
        x = self.lrelu(self.conv3(x))  
        out = self.conv4(x)           
        return out


if __name__ == "__main__":
    disc = Discriminator()
