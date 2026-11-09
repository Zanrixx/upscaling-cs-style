import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def random_pair(lr, hr):
    w, h = hr.size
    x = random.randint(0, (w - 128) // 4) * 4 #патч не должен быть больше 1920-128 и чтобы не было **,24 пикселя разделим на 4 при выборе и после просто домножим
    y = random.randint(0, (h - 128) // 4) * 4
    lx, ly = x // 4, y // 4
    hr_patch = hr.crop((x, y, x + 128, y + 128)) #.crop вырезает патч (квадратик)
    lr_patch = lr.crop((lx, ly, lx + 32, ly + 32))
    return lr_patch, hr_patch

def list_images(folder):
    folder = Path(folder)
    return sorted(i for i in folder.rglob("*") if i.is_file() and i.suffix.lower() in ".jpg")

def augment_pair(lr, hr):
    if random.randint(0, 1) < 0.5:
        
        lr = TF.hflip(lr)
        hr = TF.hflip(hr)
    k = random.randint(0, 3)
    lr = TF.rotate(lr, k*90)
    hr = TF.rotate(hr, k*90)
    return lr, hr #8 вариантов из одного патча получается 



class SRdataset(Dataset):
    def __init__(self, hr_folder, lr_folder):
        self.hr_images = list_images(hr_folder)
        self.lr_images = list_images(lr_folder)

        if len(self.hr_images) != len(self.lr_images):
            raise ValueError("Количество HR и LR изображений не совпадает")
        for h, l in zip(self.hr_images, self.lr_images):
            if h.stem != l.stem:
                raise ValueError(f"Несовпадение имен файлов: {h.name} и {l.name}")
    def __len__(self):
        return len(self.hr_images)
    def __getitem__(self, i):
        hr = Image.open(self.hr_images[i]).convert("RGB")
        lr = Image.open(self.lr_images[i]).convert("RGB")
        lr, hr = random_pair(lr, hr)
        lr, hr = augment_pair(lr, hr)
        return TF.to_tensor(lr), TF.to_tensor(hr) #переводит в тензоры [3, высота, ширина]

if __name__ == "__main__":
    ds = SRdataset("data/TEST/hr", "data/TEST/lr")
    lr, hr = ds[0]
    print(lr.size, hr.size)
