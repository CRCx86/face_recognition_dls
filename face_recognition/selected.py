import os
import pandas as pd
from pathlib import Path

# новая версия, хорошие результаты при обучении
def load_and_prepare_data():
    """Загрузка и подготовка данных CelebA"""

    # Конфигурация путей
    base_dir = "E:\\Deep Learning School\\FR\\celeba-original-wild-images"
    attr_file = f"{base_dir}\\list_attr_celeba.csv"
    bbox_file = f"{base_dir}\\list_bbox_celeba.csv"

    id_dir = "E:\\Deep Learning School\\FR\\anno"
    id_file = f"{id_dir}\\identity_CelebA.txt"

    # Загружаем данные
    df_attr = pd.read_csv(attr_file)
    df_bbox = pd.read_csv(bbox_file)
    df_id = pd.read_csv(id_file, delimiter=' ', names=['image_id', 'identity'])

    # Merge ВСЕХ данных
    df = (
        df_attr
        .merge(df_bbox, on="image_id")
        .merge(df_id, on="image_id")
    )

    # Применяем фильтры
    condition = (
            (df['Wearing_Hat'] == -1) &
            (df['Wearing_Earrings'] == -1) &
            (df['Bald'] == -1) &
            (df['Blurry'] == -1) &
            # (df['Oval_Face'] == 1) &
            (df['Eyeglasses'] == -1) &
            # (df['No_Beard'] == 1) &
            # (df['Smiling'] == 1) &
            (df['width'] > 0) &
            (df['height'] > 0)
    )

    df_filtered = df[condition].copy()
    print(f"После фильтрации: {len(df_filtered)} изображений")
    print(f"Уникальных identity: {df_filtered['identity'].nunique()}")

    return df_filtered

def calculate_optimal_classes(total_images):
    """Рассчитывает оптимальное количество классов"""
    print(f"\n{'='*50}")
    print("РАСЧЕТ ОПТИМАЛЬНОЙ КОНФИГУРАЦИИ")
    print(f"{'='*50}")
    print(f"Всего доступно изображений: {total_images}")

    configs = [
        ("Консервативно (50 img/class)", 50),
        ("Балансировано (30 img/class)", 30),
        ("Агрессивно (20 img/class)", 20),
        ("Минимум (10 img/class)", 10)
    ]

    for name, per_class in configs:
        max_classes = total_images // per_class
        print(f"{name}: {max_classes} классов")

    return total_images // 30  # рекомендую 30 изображений на класс

def balance_for_resnet(df_filtered, target_classes=None, min_per_class=30):
    """Балансировка данных под ResNet18"""

    # Подсчитываем количество изображений на identity
    id_counts = df_filtered['identity'].value_counts()

    # Определяем target_classes если не задано
    if target_classes is None:
        target_classes = calculate_optimal_classes(len(df_filtered))

    print(f"\n{'='*50}")
    print("БАЛАНСИРОВКА ДАННЫХ")
    print(f"{'='*50}")

    # Берем только identity с достаточным количеством изображений
    valid_ids = id_counts[id_counts >= min_per_class].index.tolist()
    print(f"Identity с ≥ {min_per_class} изображениями: {len(valid_ids)}")

    if len(valid_ids) < target_classes:
        print(f"Внимание! Достаточно данных только для {len(valid_ids)} классов")
        target_classes = len(valid_ids)

    # Выбираем топ-N identity
    selected_ids = valid_ids[:target_classes]
    df_selected = df_filtered[df_filtered['identity'].isin(selected_ids)].copy()

    # Рассчитываем сколько взять с каждого класса
    images_per_class = len(df_selected) // target_classes
    images_per_class = min(images_per_class, 100)  # ограничиваем сверху
    images_per_class = max(images_per_class, min_per_class)  # ограничиваем снизу

    print(f"Цель: {target_classes} классов")
    print(f"Изображений на класс: {images_per_class}")

    # Балансируем
    df_balanced = (
        df_selected
        .groupby("identity", group_keys=False)
        .apply(lambda x: x.sample(
            min(len(x), images_per_class),
            random_state=42
        ))
        .reset_index(drop=True)
    )

    # Статистика
    final_counts = df_balanced['identity'].value_counts()
    print(f"\nРЕЗУЛЬТАТ БАЛАНСИРОВКИ:")
    print(f"Изображений всего: {len(df_balanced)}")
    print(f"Уникальных identity: {df_balanced['identity'].nunique()}")
    print(f"Минимум на класс: {final_counts.min()}")
    print(f"Максимум на класс: {final_counts.max()}")
    print(f"Среднее на класс: {final_counts.mean():.1f}")
    print(f"Медиана на класс: {final_counts.median():.1f}")

    return df_balanced

def save_results(df_balanced, base_dir):
    """Сохранение результатов"""
    output_dir = "E:\\Deep Learning School\\FR\\result\\selected_images"
    output_file = "selected_files_balanced.txt"

    # Создаем директорию для результатов
    Path(output_dir).mkdir(exist_ok=True)

    # Поиск полных путей к файлам
    selected_paths = []
    for f_name in df_balanced["image_id"]:
        # Проверяем в подпапках
        for root, _, files in os.walk(base_dir):
            if f_name in files:
                selected_paths.append(os.path.join(root, f_name))
                break
        else:
            print(f"Предупреждение: файл {f_name} не найден")

    # Сохраняем список путей
    out_path = os.path.join(output_dir, output_file)
    with open(out_path, "w") as f:
        for p in selected_paths:
            f.write(p + "\n")

    # Сохраняем метаданные
    metadata_path = os.path.join(output_dir, "metadata.csv")
    df_balanced.to_csv(metadata_path, index=False)

    print(f"\n{'='*50}")
    print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ")
    print(f"{'='*50}")
    print(f"Список путей: {out_path}")
    print(f"Метаданные: {metadata_path}")
    print(f"Сохранено путей: {len(selected_paths)}")

    return out_path, metadata_path

def main():
    """Основная функция - точка входа"""
    print("Начало обработки CelebA датасета...")

    # 1. Загрузка и подготовка данных
    df_filtered = load_and_prepare_data()

    # 2. Балансировка для ResNet18
    # Можно задать конкретное количество классов или оставить None для автоматического расчета
    df_balanced = balance_for_resnet(
        df_filtered,
        target_classes=250,  # или None для авторасчета
        min_per_class=20     # минимально изображений на класс
    )

    # 3. Сохранение результатов
    base_dir = "E:\\Deep Learning School\\FR\\celeba-original-wild-images"
    save_results(df_balanced, base_dir)

    print("\nОбработка завершена успешно!")

# Запустить всю обработку целиком
if __name__ == "__main__":
    main()
    