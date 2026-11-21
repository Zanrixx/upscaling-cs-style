import torch
import math


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item() #mean среднее значение по всем элементам тензора (но тензор типа 0 мерный, поэтому нужно .item() чтобы получить число)
    if mse == 0:
        return float("inf")      
    return 10 * math.log10(1 / mse)


if __name__ == "__main__":
    a = torch.rand(3, 64, 64)
    print(psnr(a, a))                                              


def evaluate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr = model(lr_img).clamp(0, 1)
            total_psnr += psnr(sr, hr_img)
    model.train()
    return total_psnr / len(loader)


def evaluate_gan(model, loader, device, lpips_fn):
    model.eval()
    total_psnr = 0.0
    total_lpips = 0.0
    with torch.no_grad():
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr = model(lr_img).clamp(0, 1)
            total_psnr += psnr(sr, hr_img)
            total_lpips += lpips_fn(sr * 2 - 1, hr_img * 2 - 1).mean().item()
    model.train()
    n = len(loader)
    return total_psnr / n, total_lpips / n