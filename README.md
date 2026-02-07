# Проект по распознаванию лиц - Deep Learning School (Первый семестр)

## Обзор проекта
Данный проект реализует пайплайн распознавания лиц, разработанный в рамках первого семестра Deep Learning School. Система обрабатывает изображения лиц через несколько этапов для достижения надежного распознавания.

## Структура проекта

Пайплайн выполняется в следующей последовательности:

### 1. **selected**
- Начальный этап отбора и предобработки данных
- Фильтрация и подготовка исходных наборов данных с изображениями лиц

### 2. **crop**
- Модуль обрезки лиц
- Извлекает области лиц из полных изображений
- Обеспечивает согласованные размеры входных данных для последующих этапов

### 3. **stacked_hourglass**
- Детекция лицевых ориентиров с использованием Stacked Hourglass
- Обнаружение ключевых точек лица (глаза, нос, углы рта)

### 4. **alignment**
- Выравнивание и нормализация лиц
- Использует обнаруженные ориентиры для согласованного выравнивания лиц
- Стандартизирует ориентацию и масштаб лиц

### 5. **file_splitter**
- Утилита для организации данных
- Разделяет набор данных на обучающую, валидационную и тестовую выборки

### 6. **resnet_detection_arcface**
- Распознавание лиц на основе ResNet с использованием ArcFace loss
- Реализация Additive Angular Margin Loss
- Фокусируется на улучшении различимости признаков

### 7. **resnet_detection_triplet_loss**
- Распознавание лиц на основе ResNet с использованием Triplet Loss
- Реализация подхода metric learning
- Изучает эмбеддинги путем сравнения позитивных и негативных пар

### 8. **resnet_detection_arcface+triplet_loss**
- Гибридный подход, объединяющий ArcFace и Triplet Loss
- Использует преимущества обеих функций потерь
- Потенциально улучшенная точность распознавания

## Требования

### Зависимости
- Python 3.7+
- PyTorch 1.7+
- torchvision
- OpenCV
- NumPy
- PIL/Pillow
- tqdm

## Установка

```bash
# Клонировать репозиторий
git clone <url-репозитория>
cd face-recognition-dls

# Создать виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

## Использование

### Базовое выполнение пайплайна
Запуск этапов последовательно:
```bash
python selected.py [опции]
python crop.py [опции]
python stacked_hourglass.py [опции]
python alignment.py [опции]
python file_splitter.py [опции]
python resnet_detection_arcface.py [опции]
python resnet_detection_triplet_loss.py [опции]
python resnet_detection_arcface+triplet_loss.py [опции]
```

## Подготовка данных

1. Организуйте набор данных в следующей структуре:
```
dataset/
│   ├── part 1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── part 2/
│       └── image1.jpg
└── list_attr_celeba.csv/
└── .....
```

2. Запустите пайплайн предобработки:
```bash
python selected.py
python crop.py
# Продолжите через все этапы изменив пути к каталогам
```

## Результаты

Метрики производительности:
- Точность: >= 0.7 для всех моделей


## Структура файлов

```
project/
├── selected.py
├── crop.py
├── stacked_hourglass.py
├── alignment.py
├── file_splitter.py
├── resnet_detection_arcface.py
├── resnet_detection_triplet_loss.py
├── resnet_detection_arcface+triplet_loss.py
├── models/
│   ├── hourglass
│   │   ├── hourglass_landmarks_best.pth
│   │   ├── label_mapping.pth
│   ├── resnet_detection
│   │   ├── best_model_epoch30_acc0.7911.pth
│   ├── resnet_detection_arcface
│   │   ├── best_model_epoch30_acc0.8108.pth
│   ├── resnet_detection_triplet
└── README.md
```

[Укажите вашу лицензию здесь]

## Благодарности

- Deep Learning School за основу проекта
- Авторам оригинальных исследовательских статей
- Сообществу open-source за различные библиотеки и инструменты

## Контакты

По вопросам или для получения поддержки обращайтесь:
- @alexandrzinov

---

*Этот проект был разработан в рамках учебной программы первого семестра Deep Learning School.*
