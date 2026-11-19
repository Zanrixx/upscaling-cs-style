#1 стадия будет только генератор, потеря l1 и выбор лучшей точки по PSNR
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from discriminator import Discriminator
from metrics import evaluate
from utils import save_checkpoint, load_checkpoint
from dataset import SRdataset
from generator import RRDBNet
import torch.nn as nn



def train_stage2(hr_dir="data/TEST/hr", lr_dir="data/TEST/lr",
                 val_hr_dir="data/TEST/hr", val_lr_dir="data/TEST/lr",
                 ckpt_dir="checkpoints/stage2_test",
                 stage1_ckpt="checkpoints/stage1_test/best.pth",
                 num_block=2, batch_size=2, total_steps=100, every=10,
                 lr=1e-4, lambda_gan=5e-3, eta=1e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #latest.pth будет перезаписываться (туда сохраняются веса)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "latest.pth")
    best_ckpt = os.path.join(ckpt_dir, "best.pth")

    loader = DataLoader(SRdataset(hr_dir, lr_dir), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SRdataset(val_hr_dir, val_lr_dir, batch_size=batch_size, shuffle=True))

    
    gen = RRDBNet(num_block=num_block).to(device)
    disc = Discriminator().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr)


    bce_loss = nn.BCEWithLogitsLoss()

    step = 0
    best_lpips = 0.0

    if os.path.exists(ckpt):
        step = load_checkpoint(ckpt, gen, opt_g, device)
    elif os.path.exists(stage1_ckpt):
            load_checkpoint(stage1_ckpt, gen, None, device)
            #1 стадия будет только генератор, потеря l1 и выбор лучшей точки по PSNR
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from discriminator import Discriminator
from metrics import evaluate
from utils import save_checkpoint, load_checkpoint
from dataset import SRdataset
from generator import RRDBNet
import torch.nn as nn



def train_stage2(hr_dir="data/TEST/hr", lr_dir="data/TEST/lr",
                 val_hr_dir="data/TEST/hr", val_lr_dir="data/TEST/lr",
                 ckpt_dir="checkpoints/stage2_test",
                 stage1_ckpt="checkpoints/stage1_test/best.pth",
                 num_block=2, batch_size=2, total_steps=100, every=10,
                 lr=1e-4, lambda_gan=5e-3, eta=1e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #latest.pth будет перезаписываться (туда сохраняются веса)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "latest.pth")
    best_ckpt = os.path.join(ckpt_dir, "best.pth")

    loader = DataLoader(SRdataset(hr_dir, lr_dir), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SRdataset(val_hr_dir, val_lr_dir, batch_size=batch_size, shuffle=True))

    
    gen = RRDBNet(num_block=num_block).to(device)
    disc = Discriminator().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr)


    bce_loss = nn.BCEWithLogitsLoss()

    step = 0
    best_lpips = 0.0

    if os.path.exists(ckpt):
        step = load_checkpoint(ckpt, gen, opt_g, device)
    elif os.path.exists(stage1_ckpt):
            load_checkpoint(stage1_ckpt, gen, None, device)
            gen.load_state_dict(torch.load(stage1_ckpt, map_location=device, weights_only=True)["model"])
            print(f"Веса {stage1_ckpt} загружены")
    while step < total_steps:
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr = gen(lr_img)

            #вот этот блок пересмотреть 
            pred_real = disc(hr_img)
            pred_fake = disc(sr.detach())

            loss_d_real = bce_loss(pred_real, torch.ones_like(pred_real))
            loss_d_fake = bce_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) / 2
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()


            pred_fake_for_g = disc(sr)
            loss_g_gan = bce_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g_l1 = F.l1_loss(sr, hr_img)

            loss_g = loss_g_l1 + 0.005 * loss_g_gan

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()


            step += 1
            if step % every == 0:

            if step >= total_steps:
                break

    save_checkpoint(ckpt, gen, opt_g, step)
   

if __name__ == "__main__":
    train_stage2()


if __name__ == "__main__":
    train_stage2()