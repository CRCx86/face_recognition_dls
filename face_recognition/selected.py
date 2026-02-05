import os
import pandas as pd
from pathlib import Path


# предыдущая версия - плохой отбор
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

# новая версия
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

# ====================
# СПОСОБЫ ВЫЗОВА:
# ====================

# СПОСОБ 1: Запустить всю обработку целиком
if __name__ == "__main__":
    main()

# СПОСОБ 2: Поэтапный вызов (если нужно в Jupyter/Colab)
def run_step_by_step():
    """Поэтапный запуск для отладки"""

    # Шаг 1: Только загрузка
    print("Шаг 1: Загрузка данных...")
    df = load_and_prepare_data()

    # Шаг 2: Просмотр статистики
    print("\nШаг 2: Анализ данных...")
    print(f"Всего изображений: {len(df)}")
    print(f"Уникальных identity: {df['identity'].nunique()}")

    # Шаг 3: Авторасчет оптимального количества классов
    print("\nШаг 3: Расчет конфигурации...")
    optimal_classes = calculate_optimal_classes(len(df))
    print(f"Рекомендуемое количество классов: {optimal_classes}")

    # Шаг 4: Балансировка с конкретными параметрами
    print("\nШаг 4: Балансировка...")
    df_balanced = balance_for_resnet(
        df,
        target_classes=optimal_classes,  # или задайте число
        min_per_class=30
    )

    # Шаг 5: Сохранение
    print("\nШаг 5: Сохранение...")
    save_results(df_balanced, "E:\\Deep Learning School\\FR\\celeba-original-wild-images")

    return df_balanced

# СПОСОБ 3: Гибкая настройка параметров
def custom_selection():
    """Кастомная настройка параметров отбора"""

    df = load_and_prepare_data()

    # Настройте параметры под ваши нужды:
    params = {
        'target_total': 10000,      # сколько всего изображений нужно
        'target_classes': 200,       # сколько классов нужно
        'min_per_class': 30,         # минимум изображений на класс
        'max_per_class': 100,        # максимум изображений на класс
    }

    # Фильтрация identity
    id_counts = df['identity'].value_counts()
    valid_ids = id_counts[
        (id_counts >= params['min_per_class']) &
        (id_counts <= params['max_per_class'])
        ].index.tolist()

    # Выбираем нужное количество классов
    if len(valid_ids) > params['target_classes']:
        valid_ids = valid_ids[:params['target_classes']]

    df_selected = df[df['identity'].isin(valid_ids)]

    # Рассчитываем сколько взять с каждого identity
    images_per_class = params['target_total'] // len(valid_ids)

    # Балансируем
    df_balanced = (
        df_selected
        .groupby("identity", group_keys=False)
        .apply(lambda x: x.sample(
            min(len(x), images_per_class),
            random_state=42
        ))
    )

    print(f"Результат: {len(df_balanced)} изображений")
    print(f"{len(valid_ids)} классов по ~{images_per_class} изображений")

    # Сохраняем
    save_results(df_balanced, "E:\\Deep Learning School\\FR\\celeba-original-wild-images")

    return df_balanced

