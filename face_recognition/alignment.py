import numpy as np
import cv2
import torch
import os
import math
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SimpleImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, img_size=256, hm_size=64):
        self.image_dir = image_dir
        self.transform = transform
        self.img_size = img_size
        self.hm_size = hm_size

        # Получаем список всех файлов изображений
        self.image_paths = []
        self.original_shapes = []  # Будем хранить оригинальные размеры
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    full_path = os.path.join(root, file)
                    self.image_paths.append(full_path)

        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Открываем изображение и сохраняем оригинальный размер
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)

        # Применяем трансформации для модели
        if self.transform:
            transformed_image = self.transform(image)

        return transformed_image, img_path, original_size

def heatmaps_to_points(heatmaps: torch.Tensor, img_size: int, hm_size: int):
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

def calculate_angle(p1, p2):
    """Вычисляет угол между горизонталью и линией между двумя точками"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def should_align(landmarks, threshold_angle=10.0):
    """
    Определяет, нужно ли выравнивать лицо
    threshold_angle: максимальный допустимый угол наклона (градусы)
    """
    if len(landmarks) < 2:
        return False

    # Используем линию между глазами для определения наклона
    # Предполагаем, что первые две точки - глаза
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    angle = calculate_angle(left_eye, right_eye)
    return abs(angle) > threshold_angle

def scale_landmarks(landmarks, from_size, to_size):
    """
    Масштабирует ключевые точки из одного размера в другой
    from_size: (width, height) исходного изображения
    to_size: (width, height) целевого изображения
    """
    landmarks_scaled = landmarks.copy()
    landmarks_scaled[:, 0] = landmarks[:, 0] * to_size[0] / from_size[0]
    landmarks_scaled[:, 1] = landmarks[:, 1] * to_size[1] / from_size[1]
    return landmarks_scaled

def align_face(image, landmarks, target_size=(256, 256)):
    """
    Выравнивает лицо по ключевым точкам
    """
    # Нормализованный шаблон для выравнивания
    dst_pts = np.array([
        [0.35, 0.35],  # левый глаз
        [0.65, 0.35],  # правый глаз
        [0.50, 0.50],  # нос
        [0.35, 0.65],  # левый угол рта
        [0.65, 0.65],  # правый угол рта
    ], dtype=np.float32)

    # Масштабируем шаблон к целевому размеру
    dst_pts[:, 0] *= target_size[0]
    dst_pts[:, 1] *= target_size[1]

    # Находим аффинное преобразование
    M, _ = cv2.estimateAffinePartial2D(landmarks, dst_pts)

    # Применяем преобразование
    aligned = cv2.warpAffine(
        image, M, target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return aligned, M

def custom_collate_fn(batch):
    """
    Кастомная collate функция для обработки батча
    """
    images = []
    img_paths = []
    original_sizes = []

    for img, path, size in batch:
        images.append(img)
        img_paths.append(path)
        original_sizes.append(size)

    # Собираем тензоры в батч
    images = torch.stack(images, 0)

    return images, img_paths, original_sizes

def run_alignment_and_save(model, dataset, loader, angle_threshold=10.0):
    model.load_state_dict(torch.load("models/hourglass/hourglass_landmarks_best.pth"))
    model.eval()
    model = model.cuda()

    aligned_count = 0
    skipped_count = 0

    with torch.no_grad():
        for batch_idx, (imgs, img_paths, original_sizes) in enumerate(tqdm(loader)):
            imgs = imgs.cuda()

            # Получаем heatmaps от модели
            pred_hms = model(imgs)[-1]  # [B, N_landmarks, H, W]

            for i in range(imgs.size(0)):
                # Преобразуем heatmaps в координаты на изображении 256x256
                pts_256 = heatmaps_to_points(
                    pred_hms[i].cpu(),
                    img_size=dataset.img_size,
                    hm_size=dataset.hm_size,
                )

                # Загружаем оригинальное изображение
                img_path = str(img_paths[i])
                img_orig = cv2.imread(img_path)
                if img_orig is None:
                    print(f"Не удалось загрузить изображение: {img_path}")
                    continue

                # Получаем оригинальный размер
                orig_size = original_sizes[i]  # (width, height)

                # Масштабируем точки из 256x256 к оригинальному размеру
                pts_original = scale_landmarks(pts_256, (256, 256), orig_size)

                # Проверяем, нужно ли выравнивать
                if should_align(pts_original, threshold_angle=angle_threshold):
                    # Выравниваем лицо
                    aligned, M = align_face(img_orig, pts_original, target_size=(256, 256))
                    action = "выровнено"
                    aligned_count += 1
                else:
                    # Просто ресайзим до 256x256 БЕЗ выравнивания
                    aligned = cv2.resize(img_orig, (256, 256))
                    action = "пропущено (без выравнивания)"
                    skipped_count += 1

                # Сохраняем результат
                save_path = os.path.join(SAVE_DIR, os.path.basename(img_path))
                cv2.imwrite(save_path, aligned)

                # Выводим информацию о каждом 10-м изображении
                if (batch_idx * loader.batch_size + i) % 10 == 0:
                    print(f"{os.path.basename(img_path)}: {action}")

    print(f"\nСтатистика:")
    print(f"Выровнено лиц: {aligned_count}")
    print(f"Пропущено (уже горизонтально): {skipped_count}")

if __name__ == "__main__":
    # Папка для сохранения выровненных лиц
    SAVE_DIR = r"E:\Deep Learning School\FR\result\align"

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Инициализируем модель
    from face_recognition.stacked_hourglass import StackedHourglass
    model = StackedHourglass().to(device)
    model.eval()

    # Трансформации только для подачи в модель
    basic_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Для модели
        transforms.ToTensor(),
    ])

    image_dir = "E:\\Deep Learning School\\FR\\result\\cropped"
    dataset = SimpleImageDataset(image_dir, transform=basic_transform, img_size=256)

    # Создаем DataLoader с кастомной collate функцией
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Используем именованную функцию
    )

    # Запускаем выравнивание с порогом 10 градусов
    run_alignment_and_save(model, dataset, loader, angle_threshold=10.0)
