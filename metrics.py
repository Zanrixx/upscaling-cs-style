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
