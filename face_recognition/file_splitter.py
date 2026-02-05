import os
import shutil
import random
from pathlib import Path
from collections import defaultdict


def split_by_images_per_person(image_dir, identity_file, output_dir,
                               train_per_person=14, val_per_person=3):
    """
    Разделяет изображения каждого человека на train, val, test
    train_per_person: сколько изображений взять в train
    val_per_person: сколько изображений взять в val
    остальные - в test
    """

    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'val'
    test_dir = Path(output_dir) / 'test'

    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Читаем маппинг
    id_to_images = {}
    with open(identity_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_name = parts[0]
                person_id = parts[1]

                if person_id not in id_to_images:
                    id_to_images[person_id] = []
                id_to_images[person_id].append(image_name)

    train_count = val_count = test_count = 0

    for person_id, images in id_to_images.items():
        random.shuffle(images)

        train_images = images[:train_per_person]
        val_images = images[train_per_person:train_per_person + val_per_person]
        test_images = images[train_per_person + val_per_person:]

        # Копируем train
        for img in train_images:
            src = Path(image_dir) / img
            dst = train_dir / img
            if src.exists():
                shutil.copy2(src, dst)
                train_count += 1

        # Копируем val
        for img in val_images:
            src = Path(image_dir) / img
            dst = val_dir / img
            if src.exists():
                shutil.copy2(src, dst)
                val_count += 1

        # Копируем test
        for img in test_images:
            src = Path(image_dir) / img
            dst = test_dir / img
            if src.exists():
                shutil.copy2(src, dst)
                test_count += 1

    print(f"Разделение по изображениям на человека:")
    print(f"Train: {train_count} изображений")
    print(f"Val: {val_count} изображений")
    print(f"Test: {test_count} изображений")


def get_detailed_image_statistics(image_dir, identity_file):
    """
    Получает детальную статистику по вашим 10000 фото
    """
    # Словарь для хранения: ID человека -> список его фото
    person_to_images = defaultdict(list)

    # Сначала читаем все файлы из папки
    image_files = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)

    print(f"Всего фото в папке: {len(image_files)}")

    # Создаем словарь для быстрого поиска
    image_set = set(image_files)

    # Читаем маппинг файла
    with open(identity_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                file_name = parts[0]
                person_id = parts[1]

                # Если файл есть в нашей папке
                if file_name in image_set:
                    person_to_images[person_id].append(file_name)

    # Статистика
    total_people = len(person_to_images)
    total_images = sum(len(images) for images in person_to_images.values())

    # Считаем сколько людей имеет N фото
    images_per_person = [len(images) for images in person_to_images.values()]

    stats = {
        'total_people': total_people,
        'total_images': total_images,
        'avg_images': total_images / total_people if total_people > 0 else 0,
        'max_images': max(images_per_person) if images_per_person else 0,
        'min_images': min(images_per_person) if images_per_person else 0,
        'people_with_20_images': sum(1 for count in images_per_person if count == 20),
        'people_with_less_than_20': sum(1 for count in images_per_person if count < 20),
        'people_with_more_than_20': sum(1 for count in images_per_person if count > 20),
        'missing_mappings': len(image_files) - total_images,
    }

    return stats, person_to_images


if __name__ == "__main__":
    # Конфигурация
    IMAGE_DIR = "E:\\Deep Learning School\\FR\\result\\align"  # Замените на путь к папке с изображениями
    IDENTITY_FILE = "E:\\Deep Learning School\\FR\\anno\\identity_CelebA.txt"  # Путь к файлу с маппингом
    OUTPUT_DIR = "E:\\Deep Learning School\\FR\\celeb_split"  # Папка для результатов


    # Вариант 1: Разделение изображений каждого человека
    print("\n" + "="*50 + "\n")
    print("Вариант 1: Разделение изображений каждого человека")
    split_by_images_per_person(
        image_dir=IMAGE_DIR,
        identity_file=IDENTITY_FILE,
        output_dir=OUTPUT_DIR + "_per_person",
        train_per_person=22,  # 14 из 20 в train
        val_per_person=4      # 3 в val, 3 в test
    )

    # Использование
    stats, person_to_images = get_detailed_image_statistics(IMAGE_DIR, IDENTITY_FILE)

    print("\n" + "="*50)
    print(f"ДЕТАЛЬНАЯ СТАТИСТИКА ДЛЯ {stats['total_images']} ФОТО")
    print("="*50)
    print(f"Уникальных людей: {stats['total_people']}")
    print(f"Изображений с маппингом: {stats['total_images']}")
    print(f"Среднее изображений на человека: {stats['avg_images']:.2f}")
    print(f"Минимум изображений у человека: {stats['min_images']}")
    print(f"Максимум изображений у человека: {stats['max_images']}")
    print(f"\nЛюдей с ровно 20 изображениями: {stats['people_with_20_images']}")
    print(f"Людей с менее чем 20 изображениями: {stats['people_with_less_than_20']}")
    print(f"Людей с более чем 20 изображениями: {stats['people_with_more_than_20']}")
    print(f"\nФото без маппинга в identity_CelebA.txt: {stats['missing_mappings']}")

    # Топ людей по количеству фото
    if person_to_images:
        print("\nТоп-10 людей с наибольшим количеством фото:")
        print("-" * 40)
        sorted_people = sorted(person_to_images.items(),
                               key=lambda x: len(x[1]),
                               reverse=True)[:10]

        for i, (person_id, images) in enumerate(sorted_people, 1):
            print(f"{i:2}. ID {person_id}: {len(images):3} фото")
            print(f"   Примеры файлов: {', '.join(images[:3])}" +
                  ("..." if len(images) > 3 else ""))

