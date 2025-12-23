import random
import shutil
from pathlib import Path

def make_train_subset(dataset_dir="dataset", out_dir="dataset_fast", frac=0.25, seed=42):
    random.seed(seed)
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)

    src_img = dataset_dir / "train" / "images"
    src_lbl = dataset_dir / "train" / "labels"

    dst_img = out_dir / "train" / "images"
    dst_lbl = out_dir / "train" / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    # valid/test копируем полностью (чтобы метрики сравнивались честно)
    for split in ["valid", "test"]:
        for kind in ["images", "labels"]:
            s = dataset_dir / split / kind
            d = out_dir / split / kind
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)

    images = sorted(list(src_img.glob("*.jpg"))) + sorted(list(src_img.glob("*.png"))) + sorted(list(src_img.glob("*.jpeg")))
    if not images:
        raise RuntimeError(f"Не найдено изображений в {src_img}")

    random.shuffle(images)
    keep = images[: max(1, int(len(images) * frac))]

    for p in keep:
        shutil.copy2(p, dst_img / p.name)
        label = src_lbl / (p.stem + ".txt")
        if label.exists():
            shutil.copy2(label, dst_lbl / label.name)
        else:
            # если вдруг нет txt — создадим пустой (YOLO так умеет)
            (dst_lbl / (p.stem + ".txt")).write_text("", encoding="utf-8")

    print(f"Готово: {len(keep)} из {len(images)} train-изображений -> {out_dir}")

if __name__ == "__main__":
    make_train_subset(frac=0.25)
