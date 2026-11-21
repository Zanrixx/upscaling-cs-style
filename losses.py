import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        #при первом запуске torchvision скачает веса VGG19
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        #features[:35] слои до conv5_4 включительно без ReLU после него (как в ESRGAN)
        self.features = vgg19(weights=weights).features[:35].eval()
        for p in self.features.parameters():
            p.requires_grad = False                 

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        sr_feat = self.features((sr - self.mean) / self.std)   
        with torch.no_grad():                                  
            hr_feat = self.features((hr - self.mean) / self.std)
        return F.l1_loss(sr_feat, hr_feat)


if __name__ == "__main__":
    loss_fn = VGGPerceptualLoss()
    a = torch.rand(2, 3, 128, 128)
    b = torch.rand(2, 3, 128, 128)
    print(loss_fn(a, b).item())      #какое-то положительное число
    print(loss_fn(a, a).item())      #должно быть 0.0