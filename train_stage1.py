#1 стадия будет только генератор, потеря l1 и выбор лучшей точки по PSNR
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from metrics import evaluate
from utils import save_checkpoint, load_checkpoint, convert_keys
from dataset import SRdataset
from generator import RRDBNet


def train_stage1(hr_dir="data/TEST/hr", lr_dir="data/TEST/lr",
                 val_hr_dir="data/TEST/hr", val_lr_dir="data/TEST/lr",
                 ckpt_dir="checkpoints/stage1_test",
                 num_block=2, batch_size=2, total_steps=100, every=10, lr=2e-4, pretrained=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #latest.pth будет перезаписываться (туда сохраняются веса)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "latest.pth")
    best_ckpt = os.path.join(ckpt_dir, "best.pth")

    loader = DataLoader(SRdataset(hr_dir, lr_dir), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SRdataset(val_hr_dir, val_lr_dir, train=False), batch_size=1)
    model = RRDBNet(num_block=num_block).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0
    best_psnr = 0.0
    if os.path.exists(best_ckpt):
        best_psnr = torch.load(best_ckpt, map_location="cpu")["psnr"]
    #вырубается программа - делаем чекпоинт дай бог сохранит
    if os.path.exists(ckpt):
        step = load_checkpoint(ckpt, model, optimizer, device)
    elif pretrained:
        w = torch.load(pretrained, map_location=device, weights_only=True)
        model.load_state_dict(convert_keys(w))
        print("загружены готовые веса")
        best_psnr = evaluate(model, val_loader, device)      
        print("psnr до обучения:", best_psnr)

    while step < total_steps:
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            sr = model(lr_img)
            loss = F.l1_loss(sr, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % every == 0:
                current_psnr = evaluate(model, val_loader, device)
                print(f"шаг:{step} потеря: {loss.item():.4f} psnr: {current_psnr:.2f}")
                save_checkpoint(ckpt, model, optimizer, step)
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    torch.save({"model": model.state_dict(), "psnr": current_psnr, "step": step}, best_ckpt)
                    print(f"новый бест чекпоинт: {best_ckpt}")
            if step >= total_steps:
                break

    save_checkpoint(ckpt, model, optimizer, step)