import time
import shutil
import torch
from train_stage1 import train_stage1


def main():
    print("видеокарта:", torch.cuda.get_device_name(0))

    for batch in [2, 4, 8]:
        ckpt_dir = f"checkpoints/speed_bs{batch}"
        shutil.rmtree(ckpt_dir, ignore_errors=True)   # иначе возьмёт старый чекпоинт и выйдет ложно быстро
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            train_stage1(
                hr_dir="data/SRGD/Train/train-1080p",
                lr_dir="data/SRGD/Train/train-270p",
                ckpt_dir=ckpt_dir,
                num_block=23, batch_size=batch, total_steps=100, every=100,
                pretrained="weights/RRDB_PSNR_x4.pth",
            )
        except torch.cuda.OutOfMemoryError:
            print(f"батч {batch}: не влез в память")
            torch.cuda.empty_cache()
            break
        sec = time.time() - t
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"батч {batch}: {sec:.0f} сек, {sec / 100:.2f} сек/шаг, "
              f"{batch * 100 / sec:.1f} кадр/сек, пик памяти {mem:.1f} ГБ")


if __name__ == "__main__":
    main()