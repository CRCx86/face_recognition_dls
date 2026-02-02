import os
import pandas as pd
from pathlib import Path


def select_from_raw():
    # Конфигурация путей
    base_dir = "E:\\Deep Learning School\\FR\\celeba-original-wild-images"
    attr_file = f"{base_dir}\\list_attr_celeba.csv"
    bbox_file = f"{base_dir}\\list_bbox_celeba.csv"

    id_dir = "E:\\Deep Learning School\\FR\\anno"
    id_file   = f"{id_dir}\\identity_CelebA.txt"

    output_dir = "E:\\Deep Learning School\\FR\\result\\selected_images"
    output_file = "selected_files_10k.txt"

    # Создаем директорию для результатов
    Path(output_dir).mkdir(exist_ok=True)

    # Загружаем данные
    df_attr = pd.read_csv(attr_file)
    df_bbox = pd.read_csv(bbox_file)
    df_id   = pd.read_csv(id_file, delimiter=' ')

    # Merge
    df = (
        df_attr
        .merge(df_bbox, on="image_id")
        .merge(df_id, on="image_id")
    )

    # Критерии отбора
    condition = (
            (df_attr['Wearing_Hat'] == -1) &
            (df_attr['Wearing_Earrings'] == -1) &
            (df_attr['Bald'] == -1) &
            (df_attr['Blurry'] == -1) &
            # (df_attr['Heavy_Makeup'] == -1) &
            (df_attr['Oval_Face'] == 1) &
            (df_attr['Eyeglasses'] == -1) &
            (df_attr['No_Beard'] == 1) &
            (df_attr['Smiling'] == 1) &
            (df_bbox['width'] > 0) &
            (df_bbox['height'] > 0)
    )

    df = df[condition]

    # Балансировка по identity
    MAX_PER_ID = 20
    df_balanced = (
        df
        .groupby("identity", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), MAX_PER_ID), random_state=42))
    )

    # Финальный сэмпл
    df_final = df_balanced.sample(n=min(10_000, len(df_balanced)), random_state=42)

    # Поиск файлов
    selected_paths = []
    for f_name in df_final["image_id"]:
        for root, _, files in os.walk(base_dir):
            if f_name in files:
                selected_paths.append(os.path.join(root, f_name))
                break

    # Сохраняем
    out_path = os.path.join(output_dir, output_file)
    with open(out_path, "w") as f:
        for p in selected_paths:
            f.write(p + "\n")

    print(f"Сохранено путей: {len(selected_paths)} → {out_path}")