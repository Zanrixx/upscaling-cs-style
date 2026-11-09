from pathlib import Path
from PIL import Image

SCALE = 4 #1080/4=270 

def downscale(img):
    w, h = img.size
    return img.resize((w // SCALE, h // SCALE), Image.BICUBIC) #есть ещё nearest bilenear lanczos мб потом протестировать их

def process_folder(src, dst):
    folder = Path(src)
    dst = Path(dst)
    files = sorted(i for i in folder.rglob("*") if i.is_file())
    print(f"в папке {src} найдено файлов: {len(files)} ")

    for i in range(len(files)):
        src_i = files[i]
        out = dst / (src_i.stem + ".jpg")
        if out.exists():
            continue
        try:
            img = Image.open(src_i).convert("RGB")
            img_lr = downscale(img)
            img_lr.save(out, quality=95, subsampling=0)
            print(f"файл {out} сохранен")
        except Exception as e:
            print(f"ошибка на {src_i}: {e}")
            

if __name__ == "__main__":
    #process_folder("data/SRGD/val/val-1080p", "data/SRGD/val/val-270p")
    process_folder("data/SRGD/train/train-1080p", "data/SRGD/train/train-270p")