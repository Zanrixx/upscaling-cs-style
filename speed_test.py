import os
import time
import shutil
import torch
from train_stage1 import train_stage1
from train_stage2 import train_stage2
from utils import convert_keys

TRAIN_HR = "data/SRGD/Train/train-1080p"
TRAIN_LR = "data/SRGD/Train/train-270p"
WEIGHTS = "weights/RRDB_PSNR_x4.pth"
START = "checkpoints/speed_start.pth"      


def make_start_ckpt():
    if not os.path.exists(START):
        os.makedirs("checkpoints", exist_ok=True)
        w = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        torch.save({"model": convert_keys(w)}, START)


def measure_stage1(batch, ckpt_dir):
    train_stage1(hr_dir=TRAIN_HR, lr_dir=TRAIN_LR, ckpt_dir=ckpt_dir,
                 num_block=23, batch_size=batch, total_steps=100, every=100,
                 pretrained=WEIGHTS)


def measure_stage2(batch, ckpt_dir):
    train_stage2(hr_dir=TRAIN_HR, lr_dir=TRAIN_LR, ckpt_dir=ckpt_dir,
                 stage1_ckpt=START, num_block=23, batch_size=batch,
                 total_steps=100, every=100)


def timed(name, fn, batch, results):
    #запускает fn на 100 шагов, замеряет время и память; возвращает False, если не хватило памяти
    ckpt_dir = f"checkpoints/speed_{name}_bs{batch}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)      
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        fn(batch, ckpt_dir)
    except torch.cuda.OutOfMemoryError:
        results.append(f"{name}, батч {batch}: НЕ ВЛЕЗ В ПАМЯТЬ")
        torch.cuda.empty_cache()
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        return False
    sec = time.time() - t
    mem = torch.cuda.max_memory_allocated() / 1e9
    results.append(f"{name}, батч {batch}: {sec:.0f} сек, {sec / 100:.2f} сек/шаг, "
                   f"{batch * 100 / sec:.1f} кадр/сек, пик памяти {mem:.1f} ГБ")
    shutil.rmtree(ckpt_dir, ignore_errors=True)       
    return True


def main():
    print("видеокарта:", torch.cuda.get_device_name(0))
    make_start_ckpt()
    results = []

    for batch in [1, 2, 4, 8]:
        if not timed("s1", measure_stage1, batch, results):
            break
    for batch in [1, 2, 4, 8]:
        if not timed("s2", measure_stage2, batch, results):
            break

    print("\n===== ИТОГИ =====")
    print("видеокарта:", torch.cuda.get_device_name(0))
    for line in results:
        print(line)


if __name__ == "__main__":
    main()