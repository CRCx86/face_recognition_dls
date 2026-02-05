import numpy as np
import cv2
import torch
import os
import math
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Импортируем модель
from face_recognition.stacked_hourglass import StackedHourglass

def load_and_prepare_image(image_path, img_size=256):
    """Загружает и подготавливает изображение"""
    # Загружаем оригинал
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Сохраняем оригинальный размер
    h, w = img_orig.shape[:2]
    print(f"Оригинальный размер: {w}x{h}")

    # Загружаем через PIL для трансформаций
    img_pil = Image.open(image_path).convert('RGB')

    # Трансформации для модели
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img_pil).unsqueeze(0)  # [1, 3, H, W]

    return img_orig, img_tensor, (w, h)

def heatmaps_to_points(heatmaps: torch.Tensor, img_size: int, hm_size: int):
    """Преобразует heatmaps в координаты точек"""
    assert heatmaps.ndim == 3
    assert heatmaps.shape[1] == hm_size

    points = []
    for i in range(heatmaps.shape[0]):
        hm = heatmaps[i]
        idx = torch.argmax(hm)
        y = (idx // hm_size).item()
        x = (idx % hm_size).item()

        # Масштабируем к размеру изображения
        scale = img_size / hm_size
        x_img = x * scale
        y_img = y * scale
        points.append([x_img, y_img])

    return np.array(points, dtype=np.float32)

def visualize_landmarks(image, landmarks, title="Landmarks"):
    """Визуализирует изображение с ключевыми точками"""
    img_display = image.copy()

    # Цвета для разных точек
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    labels = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']

    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img_display, (int(x), int(y)), 5, colors[i], -1)
        cv2.putText(img_display, labels[i], (int(x)+10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

    # Рисуем линию между глазами
    if len(landmarks) >= 2:
        cv2.line(img_display,
                 (int(landmarks[0][0]), int(landmarks[0][1])),
                 (int(landmarks[1][0]), int(landmarks[1][1])),
                 (255, 255, 0), 2)

    # Конвертируем BGR в RGB для matplotlib
    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

    return img_display

def calculate_angle(p1, p2):
    """Вычисляет угол между горизонталью и линией между двумя точками"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def check_alignment_need(landmarks, threshold_angle=10.0):
    """Проверяет, нужно ли выравнивание"""
    if len(landmarks) < 2:
        return False, 0.0

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    angle = calculate_angle(left_eye, right_eye)

    need_align = abs(angle) > threshold_angle
    return need_align, angle

def align_face_debug(image, landmarks, target_size=(256, 256)):
    """Отладочная версия выравнивания с визуализацией промежуточных шагов"""
    print("\n=== Процесс выравнивания ===")

    # Нормализованный шаблон
    dst_pts = np.array([
        [0.35, 0.35],  # левый глаз
        [0.65, 0.35],  # правый глаз
        [0.50, 0.50],  # нос
        [0.35, 0.65],  # левый угол рта
        [0.65, 0.65],  # правый угол рта
    ], dtype=np.float32)

    # Масштабируем шаблон
    dst_pts[:, 0] *= target_size[0]
    dst_pts[:, 1] *= target_size[1]

    print(f"Исходные точки:\n{landmarks}")
    print(f"Целевые точки:\n{dst_pts}")

    # Находим аффинное преобразование
    M, inliers = cv2.estimateAffinePartial2D(landmarks, dst_pts)

    if M is None:
        print("Ошибка: не удалось найти преобразование")
        return image, None

    print(f"\nМатрица преобразования:\n{M}")

    # Вычисляем угол поворота из матрицы
    rotation_angle = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    print(f"Угол поворота: {rotation_angle:.2f} градусов")

    # Применяем преобразование
    aligned = cv2.warpAffine(
        image, M, target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # Вычисляем трансформированные точки для проверки
    ones = np.ones((landmarks.shape[0], 1))
    landmarks_homo = np.hstack([landmarks, ones])
    transformed_pts = (M @ landmarks_homo.T).T

    print(f"\nТрансформированные точки:\n{transformed_pts}")
    print(f"Целевые точки (ожидаемые):\n{dst_pts}")

    return aligned, M

def test_single_image(image_path, model_path="models/hourglass_landmarks.pth", img_size=256):
    """Тестирует одно изображение"""
    print(f"Тестируем изображение: {os.path.basename(image_path)}")
    print("=" * 50)

    # Загружаем модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StackedHourglass().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Загружаем изображение
    img_orig, img_tensor, orig_size = load_and_prepare_image(image_path, img_size)

    # Получаем предсказания
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        pred_hms = model(img_tensor)[-1]  # Берем последний выход

    # Преобразуем heatmaps в точки (256x256 координаты)
    hm_size = pred_hms.shape[-1]
    pts_256 = heatmaps_to_points(pred_hms[0].cpu(), img_size, hm_size)

    print(f"\nТочки на изображении 256x256:")
    for i, (x, y) in enumerate(pts_256):
        print(f"  Точка {i}: ({x:.1f}, {y:.1f})")

    # Масштабируем точки к оригинальному размеру
    scale_x = orig_size[0] / img_size
    scale_y = orig_size[1] / img_size
    pts_orig = pts_256.copy()
    pts_orig[:, 0] *= scale_x
    pts_orig[:, 1] *= scale_y

    print(f"\nТочки на оригинальном изображении {orig_size[0]}x{orig_size[1]}:")
    for i, (x, y) in enumerate(pts_orig):
        print(f"  Точка {i}: ({x:.1f}, {y:.1f})")

    # Проверяем, нужно ли выравнивание
    need_align, angle = check_alignment_need(pts_orig)
    print(f"\nУгол между глазами: {angle:.2f} градусов")
    print(f"Нужно выравнивание: {need_align}")

    # Визуализация 1: оригинал с точками
    print("\n1. Оригинальное изображение с ключевыми точками:")
    img_with_landmarks = visualize_landmarks(img_orig, pts_orig,
                                             f"Original Image (Angle: {angle:.1f}°)")

    if need_align:
        print("\n2. Пытаемся выровнять...")
        aligned, M = align_face_debug(img_orig, pts_orig)

        # Визуализация выровненного
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        plt.title(f"Original (Angle: {angle:.1f}°)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
        plt.title(f"Aligned (Rotated: {math.degrees(math.atan2(M[1, 0], M[0, 0])):.1f}°)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Сохраняем для сравнения
        save_dir = "debug_results"
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        cv2.imwrite(os.path.join(save_dir, f"{base_name}_original.jpg"), img_orig)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_with_landmarks.jpg"), img_with_landmarks)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_aligned.jpg"), aligned)

        print(f"\nРезультаты сохранены в папку: {save_dir}")
    else:
        print("\nВыравнивание не требуется (угол в пределах допустимого)")

        # Просто ресайзим для сравнения
        resized = cv2.resize(img_orig, (256, 256))

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        plt.title(f"Original (Angle: {angle:.1f}°)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        plt.title("Resized (no alignment)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return pts_orig, angle, need_align

if __name__ == "__main__":
    # Тестируем одно изображение
    image_path = "E:\\Deep Learning School\\FR\\result\\cropped\\007840.jpg"  # Замените на свой путь

    # Тестируем несколько изображений по очереди
    # image_dir = "E:\\Deep Learning School\\FR\\result\\cropped"
    # image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Выберите несколько изображений для теста
    # test_images = image_files[:5]  # Первые 5 изображений

    # for img_file in test_images:
    #     img_path = os.path.join(image_dir, img_file)
    #     test_single_image(img_path)

    # Или тестируем одно конкретное
    if os.path.exists(image_path):
        test_single_image(image_path)
    else:
        print(f"Файл не найден: {image_path}")
        print("Доступные файлы в директории:")
        dir_path = os.path.dirname(image_path)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            for f in files[:10]:  # Показать первые 10 файлов
                print(f"  {f}")






# import numpy as np
# import cv2
# import torch
# import os
# import math
# from tqdm import tqdm
# from torch.utils.data import Dataset
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data.dataloader import default_collate
#
# class SimpleImageDataset(Dataset):
#     def __init__(self, image_dir, transform=None, img_size=256, hm_size=64):
#         self.image_dir = image_dir
#         self.transform = transform
#         self.img_size = img_size
#         self.hm_size = hm_size
#
#         # Получаем список всех файлов изображений
#         self.image_paths = []
#         self.original_shapes = []  # Будем хранить оригинальные размеры
#         valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
#
#         for root, dirs, files in os.walk(image_dir):
#             for file in files:
#                 if any(file.lower().endswith(ext) for ext in valid_extensions):
#                     full_path = os.path.join(root, file)
#                     self.image_paths.append(full_path)
#
#         self.image_paths.sort()
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#
#         # Открываем изображение и сохраняем оригинальный размер
#         image = Image.open(img_path).convert('RGB')
#         original_size = image.size  # (width, height)
#
#         # Применяем трансформации для модели
#         if self.transform:
#             transformed_image = self.transform(image)
#
#         return transformed_image, img_path, original_size
#
# def heatmaps_to_points(heatmaps: torch.Tensor, img_size: int, hm_size: int):
#     assert heatmaps.ndim == 3
#     assert heatmaps.shape[1] == hm_size
#
#     points = []
#     for i in range(heatmaps.shape[0]):
#         hm = heatmaps[i]
#         idx = torch.argmax(hm)
#         y = (idx // hm_size).item()
#         x = (idx % hm_size).item()
#
#         # Масштабируем к размеру изображения
#         scale = img_size / hm_size
#         x_img = x * scale
#         y_img = y * scale
#         points.append([x_img, y_img])
#
#     return np.array(points, dtype=np.float32)
#
# def calculate_angle(p1, p2):
#     """Вычисляет угол между горизонталью и линией между двумя точками"""
#     dx = p2[0] - p1[0]
#     dy = p2[1] - p1[1]
#     angle = math.degrees(math.atan2(dy, dx))
#     return angle
#
# def should_align(landmarks, threshold_angle=10.0):
#     """
#     Определяет, нужно ли выравнивать лицо
#     threshold_angle: максимальный допустимый угол наклона (градусы)
#     """
#     if len(landmarks) < 2:
#         return False
#
#     # Используем линию между глазами для определения наклона
#     # Предполагаем, что первые две точки - глаза
#     left_eye = landmarks[0]
#     right_eye = landmarks[1]
#
#     angle = calculate_angle(left_eye, right_eye)
#     return abs(angle) > threshold_angle
#
# def scale_landmarks(landmarks, from_size, to_size):
#     """
#     Масштабирует ключевые точки из одного размера в другой
#     from_size: (width, height) исходного изображения
#     to_size: (width, height) целевого изображения
#     """
#     landmarks_scaled = landmarks.copy()
#     landmarks_scaled[:, 0] = landmarks[:, 0] * to_size[0] / from_size[0]
#     landmarks_scaled[:, 1] = landmarks[:, 1] * to_size[1] / from_size[1]
#     return landmarks_scaled
#
# def align_face(image, landmarks, target_size=(256, 256)):
#     """
#     Выравнивает лицо по ключевым точкам
#     """
#     # Нормализованный шаблон для выравнивания
#     dst_pts = np.array([
#         [0.35, 0.35],  # левый глаз
#         [0.65, 0.35],  # правый глаз
#         [0.50, 0.50],  # нос
#         [0.35, 0.65],  # левый угол рта
#         [0.65, 0.65],  # правый угол рта
#     ], dtype=np.float32)
#
#     # Масштабируем шаблон к целевому размеру
#     dst_pts[:, 0] *= target_size[0]
#     dst_pts[:, 1] *= target_size[1]
#
#     # Находим аффинное преобразование
#     M, _ = cv2.estimateAffinePartial2D(landmarks, dst_pts)
#
#     # Применяем преобразование
#     aligned = cv2.warpAffine(
#         image, M, target_size,
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_REFLECT
#     )
#
#     return aligned, M
#
# def custom_collate_fn(batch):
#     """
#     Кастомная collate функция для обработки батча
#     """
#     images = []
#     img_paths = []
#     original_sizes = []
#
#     for img, path, size in batch:
#         images.append(img)
#         img_paths.append(path)
#         original_sizes.append(size)
#
#     # Собираем тензоры в батч
#     images = torch.stack(images, 0)
#
#     return images, img_paths, original_sizes
#
# def run_alignment_and_save(model, dataset, loader, angle_threshold=10.0):
#     model.load_state_dict(torch.load("models/hourglass_landmarks.pth"))
#     model.eval()
#     model = model.cuda()
#
#     aligned_count = 0
#     skipped_count = 0
#
#     with torch.no_grad():
#         for batch_idx, (imgs, img_paths, original_sizes) in enumerate(tqdm(loader)):
#             imgs = imgs.cuda()
#
#             # Получаем heatmaps от модели
#             pred_hms = model(imgs)[-1]  # [B, N_landmarks, H, W]
#
#             for i in range(imgs.size(0)):
#                 # Преобразуем heatmaps в координаты на изображении 256x256
#                 pts_256 = heatmaps_to_points(
#                     pred_hms[i].cpu(),
#                     img_size=dataset.img_size,
#                     hm_size=dataset.hm_size,
#                 )
#
#                 # Загружаем оригинальное изображение
#                 img_path = str(img_paths[i])
#                 img_orig = cv2.imread(img_path)
#                 if img_orig is None:
#                     print(f"Не удалось загрузить изображение: {img_path}")
#                     continue
#
#                 # Получаем оригинальный размер
#                 orig_size = original_sizes[i]  # (width, height)
#
#                 # Масштабируем точки из 256x256 к оригинальному размеру
#                 pts_original = scale_landmarks(pts_256, (256, 256), orig_size)
#
#                 # Проверяем, нужно ли выравнивать
#                 if should_align(pts_original, threshold_angle=angle_threshold):
#                     # Выравниваем лицо
#                     aligned, M = align_face(img_orig, pts_original, target_size=(256, 256))
#                     action = "выровнено"
#                     aligned_count += 1
#                 else:
#                     # Просто ресайзим до 256x256 БЕЗ выравнивания
#                     aligned = cv2.resize(img_orig, (256, 256))
#                     action = "пропущено (без выравнивания)"
#                     skipped_count += 1
#
#                 # Сохраняем результат
#                 save_path = os.path.join(SAVE_DIR, os.path.basename(img_path))
#                 cv2.imwrite(save_path, aligned)
#
#                 # Выводим информацию о каждом 10-м изображении
#                 if (batch_idx * loader.batch_size + i) % 10 == 0:
#                     print(f"{os.path.basename(img_path)}: {action}")
#
#     print(f"\nСтатистика:")
#     print(f"Выровнено лиц: {aligned_count}")
#     print(f"Пропущено (уже горизонтально): {skipped_count}")
#
# if __name__ == "__main__":
#     # Папка для сохранения выровненных лиц
#     SAVE_DIR = r"E:\Deep Learning School\FR\result\align"
#     os.makedirs(SAVE_DIR, exist_ok=True)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Инициализируем модель
#     from face_recognition.stacked_hourglass import StackedHourglass
#     model = StackedHourglass().to(device)
#     model.eval()
#
#     # Трансформации только для подачи в модель
#     basic_transform = transforms.Compose([
#         transforms.Resize((256, 256)),  # Для модели
#         transforms.ToTensor(),
#     ])
#
#     image_dir = "E:\\Deep Learning School\\FR\\result\\cropped"
#     dataset = SimpleImageDataset(image_dir, transform=basic_transform, img_size=256)
#
#     # Создаем DataLoader с кастомной collate функцией
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=64,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True,
#         collate_fn=custom_collate_fn  # Используем именованную функцию
#     )
#
#     # Запускаем выравнивание с порогом 10 градусов
#     run_alignment_and_save(model, dataset, loader, angle_threshold=10.0)
#








# # import numpy as np
# # import cv2
# # import torch
# # import os
# # import matplotlib.pyplot as plt
# # from tqdm import tqdm  # индикатор прогресса
# # from torch.utils.data import Dataset
# # from PIL import Image
# # from torchvision import transforms
# #
# # from face_recognition.heatmap_dataset import CelebAHeatmapDataset
# # from face_recognition.stacked_hourglass import StackedHourglass
# #
# #
# # import numpy as np
# # import cv2
# # import torch
# # import os
# # import math
# # from tqdm import tqdm
# # from torch.utils.data import Dataset
# # from PIL import Image
# # from torchvision import transforms
# # from scipy.spatial import distance
# #
# #
# # class SimpleImageDataset(Dataset):
# #     """Простой датасет для загрузки изображений"""
# #
# #     def __init__(self, image_dir, transform=None, img_size=256, hm_size=64):
# #         """
# #         Args:
# #             image_dir (str): Путь к папке с изображениями
# #             transform (callable, optional): Трансформации для изображений
# #         """
# #         self.image_dir = image_dir
# #         self.transform = transform
# #         self.img_size = img_size
# #         self.hm_size = hm_size
# #
# #         # Получаем список всех файлов изображений
# #         self.image_paths = []
# #         valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
# #
# #         for root, dirs, files in os.walk(image_dir):
# #             for file in files:
# #                 if any(file.lower().endswith(ext) for ext in valid_extensions):
# #                     self.image_paths.append(os.path.join(root, file))
# #
# #         # Сортируем для воспроизводимости
# #         self.image_paths.sort()
# #
# #         # Храним оригинальные размеры
# #         self.original_shapes = []
# #         for path in self.image_paths:
# #             with Image.open(path) as img:
# #                 self.original_shapes.append(img.size)  # (width, height)
# #
# #     def __len__(self):
# #         """Возвращает количество изображений в датасете"""
# #         return len(self.image_paths)
# #
# #     def __getitem__(self, idx):
# #         """Загружает и возвращает одно изображение по индексу"""
# #         # Загружаем изображение
# #         img_path = self.image_paths[idx]
# #         original_size = self.original_shapes[idx]  # (width, height)
# #
# #         image = Image.open(img_path).convert('RGB')  # Конвертируем в RGB
# #
# #         # Применяем трансформации если они есть
# #         if self.transform:
# #             image = self.transform(image)
# #
# #         return image, img_path, original_size
# #
# # def calculate_angle(p1, p2):
# #     """Вычисляет угол между горизонталью и линией между двумя точками"""
# #     dx = p2[0] - p1[0]
# #     dy = p2[1] - p1[1]
# #     angle = math.degrees(math.atan2(dy, dx))
# #     return angle
# #
# # def should_align(landmarks, threshold_angle=10.0):
# #     """
# #     Определяет, нужно ли выравнивать лицо
# #     threshold_angle: максимальный допустимый угол наклона (градусы)
# #     """
# #     # Используем линию между глазами для определения наклона
# #     left_eye = landmarks[0]  # предполагаем, что это левый глаз
# #     right_eye = landmarks[1] # предполагаем, что это правый глаз
# #
# #     angle = calculate_angle(left_eye, right_eye)
# #
# #     # Если угол близок к 0 (горизонтально), не выравниваем
# #     return abs(angle) > threshold_angle
# #
# #
# # def heatmaps_to_points(
# #         heatmaps: torch.Tensor,
# #         img_size: int,
# #         hm_size: int,
# # ):
# #     assert heatmaps.ndim == 3
# #     assert heatmaps.shape[1] == hm_size
# #
# #     points = []
# #
# #     for i in range(heatmaps.shape[0]):
# #         hm = heatmaps[i]
# #
# #         idx = torch.argmax(hm)
# #         y = (idx // hm_size).item()
# #         x = (idx % hm_size).item()
# #
# #         # hm → img_size
# #         scale = img_size / hm_size
# #         x_img = x * scale
# #         y_img = y * scale
# #
# #         points.append([x_img, y_img])
# #
# #     return np.array(points, dtype=np.float32)
# #
# #
# # def align_face(image, landmarks, target_size=(256, 256), scale_factor=1.0):
# #     """
# #     Выравнивает лицо по ключевым точкам
# #
# #     Args:
# #         image: исходное изображение (H, W, C)
# #         landmarks: ключевые точки лица [N, 2]
# #         target_size: размер выходного изображения
# #         scale_factor: масштаб относительно области лица
# #     """
# #     # Нормализованный шаблон для выравнивания
# #     dst_pts = np.array([
# #         [0.35, 0.35],  # левый глаз
# #         [0.65, 0.35],  # правый глаз
# #         [0.50, 0.50],  # нос
# #         [0.35, 0.65],  # левый угол рта
# #         [0.65, 0.65],  # правый угол рта
# #     ], dtype=np.float32)
# #
# #     # Масштабируем шаблон к целевому размеру
# #     dst_pts[:, 0] *= target_size[0]
# #     dst_pts[:, 1] *= target_size[1]
# #
# #     # Находим аффинное преобразование
# #     M, _ = cv2.estimateAffinePartial2D(landmarks, dst_pts)
# #
# #     # Применяем преобразование
# #     aligned = cv2.warpAffine(
# #         image, M, target_size,
# #         # flags=cv2.INTER_LINEAR,
# #         # borderMode=cv2.BORDER_REFLECT
# #     )
# #
# #     aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
# #
# #     return aligned_rgb, M
# #
# #
# # def run_alignment_and_save(model, dataset, loader):
# #     model.load_state_dict(torch.load("models/hourglass_landmarks.pth"))
# #     model.eval()
# #
# #     model = model.cuda()
# #
# #     with torch.no_grad():
# #         for batch_idx, (img, img_path) in enumerate(tqdm(loader)):
# #             imgs = img.cuda()
# #             img_paths = img_path
# #
# #             pred_hms = model(imgs)[-1]  # [B, N_landmarks, H, W]
# #
# #             for i in range(imgs.size(0)):
# #                 # heatmaps → img_size coords
# #                 pts_img = heatmaps_to_points(
# #                     pred_hms[i].cpu(),
# #                     img_size=dataset.img_size,
# #                     hm_size=dataset.hm_size,
# #                 )
# #
# #                 # img_size → original coords
# #                 # pts_orig = img_points_to_original(
# #                 #     pts_img,
# #                 #     orig_shapes[i],
# #                 #     dataset.img_size,
# #                 # )
# #
# #                 img_orig = cv2.imread(str(img_paths[i]))
# #
# #                 aligned, _ = align_face(img_orig, pts_img)
# #
# #                 save_path = os.path.join(SAVE_DIR, os.path.basename(img_paths[i]))
# #                 cv2.imwrite(save_path, aligned)
# #
# #
# # if __name__ == "__main__":
# #     # Папка для сохранения выровненных лиц
# #     SAVE_DIR = r"E:\Deep Learning School\FR\result\align"
# #     os.makedirs(SAVE_DIR, exist_ok=True)
# #
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #
# #     model = StackedHourglass().to(device)
# #     model.eval()
# #
# #     basic_transform = transforms.Compose([
# #         transforms.Resize((256, 256)),
# #         transforms.ToTensor(),  # Конвертируем в тензор
# #     ])
# #
# #     image_dir = "E:\\Deep Learning School\\FR\\result\\cropped"
# #     dataset = SimpleImageDataset(image_dir, transform=basic_transform, img_size=256)
# #
# #     loader = torch.utils.data.DataLoader(
# #         dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
# #     )
# #
# #     run_alignment_and_save(model, dataset, loader)
# #
# #
# # # def img_points_to_original(points_img, orig_shape, img_size):
# # #     h, w = orig_shape
# # #     sx = w / img_size
# # #     sy = h / img_size
# # #
# # #     pts = points_img.copy()
# # #     pts[:, 0] *= sx
# # #     pts[:, 1] *= sy
# # #
# # #     return pts
# #
# #
# # # def collate_fn(batch):
# # #     return {
# # #         "img": torch.stack([b["img"] for b in batch]),
# # #         "heatmaps": torch.stack([b["heatmaps"] for b in batch]),
# # #         "img_path": [b["img_path"] for b in batch],
# # #         "orig_shape": [b["orig_shape"] for b in batch],  # list[(h,w)]
# # #     }
# #
# #
# # # рабочий
# # # def heatmaps_to_points(hm, img_shape):
# # #     """
# # #     hm: [N, H, W] тензор предсказанных heatmap
# # #     img_shape: (H_img, W_img, C)
# # #     """
# # #     H_hm, W_hm = hm.shape[1:]
# # #     H_img, W_img = img_shape[:2]
# # #
# # #     pts = []
# # #     for i in range(hm.shape[0]):
# # #         idx = torch.argmax(hm[i])
# # #         y = idx // W_hm
# # #         x = idx % W_hm
# # #
# # #         # масштабируем в пиксели изображения
# # #         x = x.item() * W_img / W_hm
# # #         y = y.item() * H_img / H_hm
# # #         pts.append([x, y])
# # #
# # #     return np.array(pts, dtype=np.float32)
# # #
# # # def run_alignment(model, dataset, target_size=(256, 256)):
# # #     model.load_state_dict(torch.load("models/hourglass_landmarks.pth"))
# # #     model.eval().cuda()
# # #
# # #     # берем первую картинку для примера
# # #     img_tensor, _ = dataset[0]
# # #     img_path = dataset.images[0]
# # #     img = cv2.imread(str(img_path))
# # #
# # #     with torch.no_grad():
# # #         pred_hm = model(img_tensor.unsqueeze(0).cuda())[-1][0].cpu()
# # #
# # #     pred_pts = heatmaps_to_points(pred_hm, img.shape)
# # #
# # #     # Нормализованный шаблон для выравнивания
# # #     dst_pts = np.array([
# # #             [0.35, 0.35],  # левый глаз
# # #             [0.65, 0.35],  # правый глаз
# # #             [0.50, 0.50],  # нос
# # #             [0.35, 0.65],  # левый угол рта
# # #             [0.65, 0.65],  # правый угол рта
# # #     ], dtype=np.float32)
# # #
# # #     # Масштабируем шаблон к целевому размеру
# # #     dst_pts[:, 0] *= target_size[0]
# # #     dst_pts[:, 1] *= target_size[1]
# # #
# # #     # similarity transform
# # #     M, _ = cv2.estimateAffinePartial2D(pred_pts, dst_pts)
# # #
# # #     # warpAffine: (width, height)
# # #     aligned = cv2.warpAffine(img, M, target_size)
# # #
# # #     # показать через matplotlib
# # #     aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
# # #     plt.imshow(aligned_rgb)
# # #     plt.axis('off')
# # #     plt.show()
