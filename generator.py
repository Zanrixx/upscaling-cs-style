import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat, num_grow_ch):
        #num_feat число признаков на входе, а num_grow_ch сколько мы добавим за ещё один слой новых признаков
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1) #получили 64 на вход вышло 32 новых
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch,      num_grow_ch, 3, 1, 1) #получили 64+32=96 на вход вышло 32 новых
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch,  num_grow_ch, 3, 1, 1) #получили 64+32*2=128 на вход вышло 32 новых
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch,  num_grow_ch, 3, 1, 1) #получили 64+32*3=160 на вход вышло 32 новых
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch,  num_feat,    3, 1, 1)   #получили 64+32*4=192 на вход вышло 64 новых
        self.lrelu = nn.LeakyReLU(0.2, inplace=True) #отрицательные числа на 0.2 умножаем, inplace=True значит что мы не создаём новый тензор, а перезаписываем старый

    def forward(self, x): #отдаём 64 временно растим её до 192 но потом запихиваем обратно в 64 только с уточненными значеними
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))   # без lrelu
        return x5 * 0.2 + x 
        # веса живут внутри слоёв (self.conv1...)(наугад), а в x, x1, x2 лежат признаки, которые через них проходят
        # x1 = lrelu(conv1(x)): свёртка по x даёт 32 новых признака
        # cat склеивает признаки по каналам (x + x1 = 96), поэтому у каждой следующей свёртки вход больше; в конце x5 * 0.2 + x это остаток
        
class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        return feat

if __name__ == "__main__":
    model = RRDBNet()
    x = torch.rand(1, 3, 32, 32)
    print(model(x).shape)