### 0. **архив с результатми**
https://drive.google.com/drive/folders/1zKB-nM3D1h_hkXYBC4x17UWJ-Vqnc58n?usp=drive_link

### 1. **selected**
- Основная идея работы утилиты в том, чтобы отсечь все фотки, которые будут мешать обучению.
```bash
    # Фильтры вышли следующие: все, кроме шляп, серьги (это не принципиально) 
    # не лысые, не размытые, без очков
    condition = (
            (df['Wearing_Hat'] == -1) &
            (df['Wearing_Earrings'] == -1) &
            (df['Bald'] == -1) &
            (df['Blurry'] == -1) &
            (df['Eyeglasses'] == -1) &
    )
```
таким образом получилось выбрать больше 200+ классов (людей) с более чем 20 фотографиями.
Это в целом основной поинт отбора для последующего обучения на ImageNet.
Получилось чуть более 7000+ фотографий.
Результат: файл - selectd_files_balanced.txt

### 2. **crop**
 - Тут простая обрезка по bbox
 - Результат: взять из архива images.zip каталог result/cropped:

### 3. **stacked_hourglass**
- Стандартная реализация на основе код из ноутбука
- Результат: из архива - models/hourglass/hourglass_landmarks_best.pth

### 4. **alignment**
- Здесь выравниваем на основании работы stacked hourglass. Выравнивание не всегда получается хорошим.
Некоторые изображение "разрываются". 
- Результат: взять из архива images.zip каталога - result/aling

### 5. **file_splitter**
- Здесь резделение на train, val и test выборки
- Результат: взять их архива images.zip каталог celeb_split_per_person

### 6. **resnet_detectione**
- Распознавание на основании разделенного датасета.
- Результат: из архива - models/resnet_detection/best_model_epoch30_acc0.7911.pth

### 7. **resnet_detection_arcface**
- Распознавание на основании разделенного датасета.
- Результат: из архива - models/resnet_detection_acrface/best_model_epoch30_acc0.8108.pth

### 8. **resnet_detection_triplet_loss**
- Эта штука в итоге получается очень тяжелой.
- Результат: если коллаб "родит" результат, то добавлю в архив

### 9. **resnet_detection_arcface+triplet_loss**
- И эта.
- Результат: если коллаб "родит" результат, то добавлю в архив
