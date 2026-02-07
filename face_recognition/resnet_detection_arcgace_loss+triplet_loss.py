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
            csv_file (string): Путь к CSV файлу с идентификаторами (обязательно)
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


class TripletFaceDataset(FaceDatasetWithCSV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # label -> список индексов
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            self.label_to_indices.setdefault(label, []).append(idx)

        self.labels_set = list(self.label_to_indices.keys())

    def __getitem__(self, idx):
        anchor_img, anchor_label = super().__getitem__(idx)

        # POSITIVE
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = np.random.choice(self.label_to_indices[anchor_label])

        positive_img, _ = super().__getitem__(pos_idx)

        # NEGATIVE
        neg_label = np.random.choice(
            [l for l in self.labels_set if l != anchor_label]
        )
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        negative_img, _ = super().__getitem__(neg_idx)

        return anchor_img, positive_img, negative_img, anchor_label


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
        x = F.normalize(x, dim=1)
        return x


# ==================== 2. ФУНКЦИИ ДЛЯ ОЦЕНКИ ====================
def test_model(model, dataloader, device='cuda'):
    """Вычисляет accuracy модели на данных из dataloader"""
    model.eval()

    total_acc = 0.0
    total = 0

    for anchor, positive, negative, _ in tqdm(dataloader, desc="Testing", leave=False):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        acc = triplet_accuracy(emb_a, emb_p, emb_n)

        batch_size = anchor.size(0)
        total_acc += acc.item() * batch_size
        total += batch_size

    accuracy = total_acc / total
    print(f"Accuracy: {accuracy:.4f} ({total_acc}/{total})")

    model.train()

    return accuracy


def validate_model(model, loader, arcface_loss, triplet_loss, lambda_arc, lambda_triplet, device):
    """Вычисляет loss и accuracy на валидационном наборе"""
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for anchor, positive, negative, labels in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = (
                lambda_arc * arcface_loss(emb_a, labels) +
                lambda_triplet * triplet_loss(emb_a, emb_p, emb_n)
        )

        acc = triplet_accuracy(emb_a, emb_p, emb_n)

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples


def quick_test(model, dataloader, device='cuda'):
    """Быстрая проверка accuracy без прогресс-бара"""
    model.eval()

    total_acc = 0
    total = 0

    with torch.no_grad():
        for anchor, positive, negative, _ in tqdm(dataloader, desc="Quick testing", leave=False):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            acc = triplet_accuracy(emb_a, emb_p, emb_n)

            batch_size = anchor.size(0)
            total_acc += acc.item() * batch_size
            total += batch_size

    accuracy = total_acc / total
    print(f"Accuracy: {accuracy:.4f} ({total_acc}/{total})")

    model.train()
    return accuracy


def train_model(model, train_loader, val_loader, arcface_loss, triplet_loss, num_epochs=10, lr=0.001, lambda_arc=1.0, lambda_triplet=0.5):
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
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")

        # Обучение
        model.train()
        total_loss = 0.0
        total_triplet_acc = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for anchor, positive, negative, labels in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Прямой проход
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss_arc = arcface_loss(emb_a, labels)
            loss_triplet = triplet_loss(emb_a, emb_p, emb_n)

            loss = (
                    lambda_arc * loss_arc +
                    lambda_triplet * loss_triplet
            )

            # Обратный проход

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                triplet_acc = triplet_accuracy(emb_a, emb_p, emb_n)

            # Статистика
            batch_size = anchor.size(0)
            total_loss  += loss.item() * batch_size
            total_triplet_acc += triplet_acc.item() * batch_size
            total_samples += batch_size

            # Обновляем прогресс-бар
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'T-Acc': f'{triplet_acc.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        # Средние значения за эпоху
        avg_train_loss = total_loss / total_samples
        avg_train_triplet_acc = total_triplet_acc / total_samples

        # Валидация
        val_loss, val_triplet_acc = validate_model(
            model,
            val_loader,
            arcface_loss,
            triplet_loss,
            lambda_arc,
            lambda_triplet,
            device
        )

        # Сохраняем историю
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_triplet_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_triplet_acc)

        # Вывод статистики
        print(
            f"\nTrain Loss: {avg_train_loss:.4f} | "
            f"Train Triplet Acc: {avg_train_triplet_acc:.4f}"
        )
        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Triplet Acc:   {val_triplet_acc:.4f}"
        )

        # Сохранение лучшей модели
        if val_triplet_acc > 0.7:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_triplet_acc': val_triplet_acc,
            }, f'models\\resnet_detection_triplet\\mixed_loss_epoch{epoch+1}_acc{val_triplet_acc:.4f}.pth')
            print(f"✅ Model saved (Triplet Acc = {val_triplet_acc:.4f})")

    return model, history


@torch.no_grad()
def triplet_accuracy(emb_a, emb_p, emb_n):
    d_ap = (emb_a - emb_p).pow(2).sum(1)
    d_an = (emb_a - emb_n).pow(2).sum(1)
    return (d_ap < d_an).float().mean()


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
    train_dataset = TripletFaceDataset(image_dir=train_dir, csv_file=label_file, transform=train_transform, split='train')
    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = TripletFaceDataset(image_dir=val_dir, csv_file=label_file, transform=val_transform, split='val')
    val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False, num_workers=4, pin_memory=True)

    # 1. Загружаем предобученную на ImageNet модель
    model = FaceModel(embedding_dim=256)  # предобученная

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # 2. Узнаем сколько у вас уникальных людей
    num_classes = train_dataset.num_classes
    print(f"\n✅ Количество классов (уникальных людей): {num_classes}")

    arcface_loss = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=256,
        margin=0.5,
        scale=64.0,
        label_smoothing=0.1
    ).to(device)

    # Loss функция и оптимизатор
    triplet_criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=0.2
    ).to(device)
    # Эта штука работает так же как и выше
    # triplet_criterion = nn.TripletMarginLoss(
    #     margin=0.2,
    #     p=2
    # )

    # Быстрая проверка
    print("Быстрая проверка accuracy на валидации:")
    acc = quick_test(model, val_loader)


    # Полное обучение
    print("\nНачинаем обучение...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        arcface_loss=arcface_loss,
        triplet_loss=triplet_criterion,
        num_epochs=30,
        lr=0.001
    )

    test_dataset = TripletFaceDataset(image_dir=test_dir, csv_file=label_file, transform=val_transform, split='test')
    test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)

    final_acc = test_model(trained_model, test_loader)

    print(f"Финальная accuracy на тесте: {final_acc:.4f}")

