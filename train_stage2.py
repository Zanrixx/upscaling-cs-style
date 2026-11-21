#генератор стартует с весов стадии 1 и учится вместе с дискриминатором.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from discriminator import Discriminator
from utils import save_checkpoint_gan, load_checkpoint_gan
from dataset import SRdataset
from generator import RRDBNet
from losses import VGGPerceptualLoss
from metrics import evaluate_gan
import lpips


def train_stage2(hr_dir="data/TEST/hr", lr_dir="data/TEST/lr",
                 val_hr_dir="data/TEST/hr", val_lr_dir="data/TEST/lr",
                 ckpt_dir="checkpoints/stage2_test",
                 stage1_ckpt="checkpoints/stage1_test/best.pth",
                 num_block=2, batch_size=2, total_steps=100, every=10,
                 lr=1e-4, lambda_gan=5e-3, eta=1e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #latest.pth будет перезаписываться
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "latest.pth")
    best_ckpt = os.path.join(ckpt_dir, "best.pth")

    loader = DataLoader(SRdataset(hr_dir, lr_dir), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SRdataset(val_hr_dir, val_lr_dir, train=False), batch_size=1)

    gen = RRDBNet(num_block=num_block).to(device)   
    disc = Discriminator().to(device)      
    perceptual = VGGPerceptualLoss().to(device) #VGG будет сравнить две большие картинки, чтобы генератор понимал куда идти(к 0 стремится крч)        
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    #оптимизаторы
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr)

    #потеря сигмоиду она делает сама
    bce_loss = nn.BCEWithLogitsLoss()

    step = 0
    best_lpips = float("inf")   

    if os.path.exists(best_ckpt):                  
        best_lpips = torch.load(best_ckpt, map_location="cpu")["lpips"]


    if os.path.exists(ckpt):
        step = load_checkpoint_gan(ckpt, gen, disc, opt_g, opt_d, device)
        print("продолжаем с шага", step)
    elif os.path.exists(stage1_ckpt):
        gen.load_state_dict(torch.load(stage1_ckpt, map_location=device, weights_only=True)["model"])
        print(f"Веса {stage1_ckpt} загружены")
    else:
        raise FileNotFoundError(f"не найдены веса стадии 1: {stage1_ckpt}")

    while step < total_steps:
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr = gen(lr_img)                      

            #1)учим ДИСКРИМИНАТОР
            pred_real = disc(hr_img)                                             #оценка настоящих кадров
            pred_fake = disc(sr.detach())                                        #оценка подделок
            loss_d_real = bce_loss(pred_real, torch.ones_like(pred_real))        #настоящие -> 1
            loss_d_fake = bce_loss(pred_fake, torch.zeros_like(pred_fake))       #подделки -> 0
            loss_d = (loss_d_real + loss_d_fake) / 2
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            #2)учим ГЕНЕРАТОР
            pred_fake_for_g = disc(sr)                  
            loss_g_gan = bce_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))  
            loss_g_l1 = F.l1_loss(sr, hr_img)      
            loss_g_per = perceptual(sr, hr_img) 
            loss_g = loss_g_per + lambda_gan * loss_g_gan + eta * loss_g_l1   #формула из статьи ESRGAN
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            step += 1
            if step % every == 0:
                cur_psnr, cur_lpips = evaluate_gan(gen, val_loader, device, lpips_fn)
                print(f"шаг:{step} D:{loss_d.item():.4f} G:{loss_g.item():.4f} (per {loss_g_per.item():.4f}) val psnr: {cur_psnr:.2f} lpips: {cur_lpips:.4f}")
                save_checkpoint_gan(ckpt, gen, disc, opt_g, opt_d, step)
                if cur_lpips < best_lpips:  
                    best_lpips = cur_lpips
                    torch.save({"model": gen.state_dict(), "lpips": cur_lpips,
                                "psnr": cur_psnr, "step": step}, best_ckpt)
                    print(f"новый бест чекпоинт: {best_ckpt}")
            if step >= total_steps:
                break

    save_checkpoint_gan(ckpt, gen, disc, opt_g, opt_d, step)


if __name__ == "__main__":
    train_stage2()