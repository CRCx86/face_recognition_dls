import cv2
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

# Пути
base_dir = Path("E:\\Deep Learning School\\FR\\celeba-original-wild-images")
selected_file = Path("E:\\Deep Learning School\\FR\\result\\selected_images\\selected_files_10k.txt")

out_dir = Path("E:\\Deep Learning School\\FR\\result\\cropped")
out_dir.mkdir(parents=True, exist_ok=True)

bbox_file = base_dir / "list_bbox_celeba.csv"
lm_file   = base_dir / "list_landmarks_celeba.csv"

def crop_img():
    # Загружаем bbox + landmarks
    df_bbox = pd.read_csv(bbox_file)
    df_lm   = pd.read_csv(lm_file)

    df = df_bbox.merge(df_lm, on="image_id").set_index("image_id")

    # Читаем выбранные файлы
    with open(selected_file) as f:
        selected_paths = [line.strip() for line in f]

    padding = 0.15

    for path in selected_paths:
        img_path = Path(path)
        image_id = img_path.name

        if image_id not in df.index:
            continue

        row = df.loc[image_id]

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        bx, by, bw, bh = row["x_1"], row["y_1"], row["width"], row["height"]

        pad_w = int(bw * padding)
        pad_h = int(bh * padding)

        x1 = max(0, bx - pad_w)
        y1 = max(0, by - pad_h)
        x2 = min(w, bx + bw + pad_w)
        y2 = min(h, by + bh + pad_h)

        crop = img[y1:y2, x1:x2]

        landmarks = np.array([
            [row["lefteye_x"],   row["lefteye_y"]],
            [row["righteye_x"],  row["righteye_y"]],
            [row["nose_x"],      row["nose_y"]],
            [row["leftmouth_x"], row["leftmouth_y"]],
            [row["rightmouth_x"],row["rightmouth_y"]],
        ], dtype=np.float32)

        landmarks[:, 0] -= x1
        landmarks[:, 1] -= y1

        # сохраняем
        cv2.imwrite(str(out_dir / image_id), crop)
        np.save(out_dir / f"{image_id}.npy", landmarks)

def check_lm(image_id):
    img = cv2.imread(str(out_dir / image_id))
    lm = np.load(out_dir / f"{image_id}.npy")

    plt.imshow(img[..., ::-1])
    plt.scatter(lm[:,0], lm[:,1], c='r')
    plt.show()

if __name__ == "__main__":
    img_id = "002832.jpg"

    check_lm(img_id)
