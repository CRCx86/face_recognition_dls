import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

from tqdm import tqdm  # для прогресс-бара

import numpy as np

import math


train_dir = 'E:\\Deep Learning School\\FR\\celeb_split_per_person\\train'
val_dir = 'E:\\Deep Learning School\\FR\\celeb_split_per_person\\val'
test_dir = 'E:\\Deep Learning School\\FR\\celeb_split_per_person\\test'
label_file = 'E:\\Deep Learning School\\FR\\anno\\identity_CelebA.txt'


class FaceDatasetWithCSV(Dataset):
    """Датасет с метками из CSV файла"""
    def __init__(self, image_dir, csv_file=None, transform=None, split='train'):
        """
        Args:
            image_dir (string): Папка с фотографиями
            csv_file (string): Путь к CSV файлу с метками (опционально)
            transform (callable, optional): Трансформации
        """
        self.image_dir = image_dir
        self.transform = transform

        self.labels_df = pd.read_csv(csv_file, delimiter=' ', names=['image_id', 'identity'])
        existing_files = set(os.listdir(image_dir))
        self.labels_df = self.labels_df[self.labels_df['image_id'].isin(existing_files)]

        print(f"{split.upper()}: {len(self.labels_df)} изображений")

        # СОЗДАЕМ ЕДИНЫЙ СЛОВАРЬ METOK ДЛЯ ВСЕХ НАБОРОВ
        if split == 'train':
            # В тренировочном наборе создаем mapping
            unique_ids = sorted(self.labels_df['identity'].unique())
            self.id_to_idx = {old_id: idx for idx, old_id in enumerate(unique_ids)}
            self.num_classes = len(unique_ids)
            print(f"Уникальных людей в train: {self.num_classes}")

            # Сохраняем mapping для использования в val/test
            torch.save(self.id_to_idx, 'models/hourglass/label_mapping.pth')
        else:
            # Загружаем mapping из train
            self.id_to_idx = torch.load('models/hourglass/label_mapping.pth')
            self.num_classes = len(self.id_to_idx)
            print(f"Используем mapping из train, классов: {self.num_classes}")

            # Оставляем только людей, которые есть в train
            self.labels_df = self.labels_df[self.labels_df['identity'].isin(self.id_to_idx.keys())]
            print(f"После фильтрации по train людям: {len(self.labels_df)} изображений")

        # Преобразуем метки
        self.labels = [self.id_to_idx[old_id] for old_id in self.labels_df['identity']]
        self.image_files = self.labels_df['image_id'].tolist()

        # Проверяем распределение классов
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"Минимальное количество изображений на класс: {counts.min()}")
        print(f"Максимальное количество изображений на класс: {counts.max()}")
        print(f"Среднее количество изображений на класс: {counts.mean():.2f}")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Загружаем изображение
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Если ошибка загрузки, создаем черное изображение
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        # Последняя проверка
        if label >= self.num_classes:
            print(f"ВНИМАНИЕ: метка {label} >= num_classes {self.num_classes}, исправляю...")
            label = 0

        return image, label


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    https://arxiv.org/abs/1801.07698
    """

    def __init__(
            self,
            num_classes: int,
            embedding_dim: int,
            margin: float = 0.5,
            scale: float = 64.0,
            label_smoothing: float = 0.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.label_smoothing = label_smoothing

        # Классовые веса (W)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Предвычисления
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        """
        embeddings: [B, embedding_dim]
        labels:     [B]
        """

        # L2-нормализация
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)

        # cos(theta)
        cosine = F.linear(embeddings, weight)  # [B, num_classes]
        cosine = cosine.clamp(-1.0, 1.0)

        # sin(theta)
        sine = torch.sqrt(1.0 - cosine ** 2)

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply margin only to target class
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        # CrossEntropy с label smoothing
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing
        )

        return loss


class FaceModel(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        # Берем предобученный ResNet
        self.backbone = models.resnet34(weights='IMAGENET1K_V1')

        # Замораживаем первые слои (опционально)
        for param in self.backbone.parameters():
            param.requires_grad = True  # Размораживаем все для тонкой настройки

        # Заменяем последний слой
        in_features = self.backbone.fc.in_features

        # Убираем классификатор ResNet
        self.backbone.fc = nn.Identity()

        # Эмбеддинг-голова
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=False)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x


# ==================== 2. ФУНКЦИИ ДЛЯ ОЦЕНКИ ====================
def test_model(model, dataloader, criterion, device='cuda'):
    """Вычисляет accuracy модели на данных из dataloader"""
    model.eval()  # Переводим модель в режим оценки
    correct = 0
    total = 0

    with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
        for images, labels in tqdm(dataloader, desc="Testing", leave=False):
            # Переносим данные на устройство (GPU/CPU)
            images = images.to(device)
            labels = labels.to(device)

            # Прямой проход
            embeddings = model(images)

            # 2. cosine similarity (без margin!)
            emb = F.normalize(embeddings, dim=1)
            W = F.normalize(criterion.weight, dim=1)
            logits = F.linear(emb, W)

            preds = logits.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = correct / total

    model.train()  # Возвращаем модель в режим обучения
    return accuracy


def validate_model(model, dataloader, criterion, device='cuda'):
    """Вычисляет loss и accuracy на валидационном наборе"""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            embeddings = model(images)
            loss = criterion(embeddings, labels)

            total_loss += loss.item() * images.size(0)

            # 3. Accuracy БЕЗ margin
            emb = F.normalize(embeddings, dim=1)
            W = F.normalize(criterion.weight, dim=1)
            logits = F.linear(emb, W)  # cos(theta)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    model.train()
    return avg_loss, accuracy


def quick_test(model, dataloader, criterion, device='cuda'):
    """Быстрая проверка accuracy без прогресс-бара"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            embeddings = model(images)
            # 2. cosine similarity (без margin!)
            emb = F.normalize(embeddings, dim=1)
            W = F.normalize(criterion.weight, dim=1)
            logits = F.linear(emb, W)

            preds = logits.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    model.train()
    return accuracy


def train_model(model, train_loader, val_loader, criterion, num_epochs=10, lr=0.001):
    """Полный цикл обучения с валидацией"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # История обучения
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'train_acc': []}

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")

        # Обучение
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Прямой проход
            embeddings = model(images)
            loss = criterion(embeddings, labels)

            # Обратный проход

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Статистика
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(embeddings.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Обновляем прогресс-бар
            current_acc = train_correct / train_total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        # Средние значения за эпоху
        avg_train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total

        # Валидация
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        # Сохраняем историю
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)

        # Вывод статистики
        print(f"\nTraining Loss: {avg_train_loss:.4f}, Training Acc: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}")

        # Сохранение лучшей модели
        if val_accuracy > 0.7:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, f'models\\resnet_detection_arcface\\best_model_epoch{epoch+1}_acc{val_accuracy:.4f}.pth')
            print(f"✅ Model saved with accuracy: {val_accuracy:.4f}")

    return model, history


if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Вариант 1: С CSV файлом
    train_dataset = FaceDatasetWithCSV(image_dir=train_dir, csv_file=label_file, transform=train_transform, split='train')
    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = FaceDatasetWithCSV(image_dir=val_dir, csv_file=label_file, transform=val_transform, split='val')
    val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False, num_workers=4, pin_memory=True)

    # 1. Загружаем предобученную на ImageNet модель
    model = FaceModel(embedding_dim=256)  # предобученная

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # 2. Узнаем сколько у вас уникальных людей
    num_classes = train_dataset.num_classes
    print(f"\n✅ Количество классов (уникальных людей): {num_classes}")

    # Loss функция и оптимизатор
    criterion = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=256,
        margin=0.5,
        scale=64.0,
        label_smoothing=0.1
    ).to(device)

    # Быстрая проверка
    print("Быстрая проверка accuracy на валидации:")
    acc = quick_test(model, val_loader, criterion)


    # Полное обучение
    print("\nНачинаем обучение...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=30,
        lr=0.0005
    )

    test_dataset = FaceDatasetWithCSV(image_dir=test_dir, csv_file=label_file, transform=val_transform, split='test')
    test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)

    final_acc = test_model(trained_model, test_loader, criterion)

    print(f"Финальная accuracy на тесте: {final_acc:.4f}")

