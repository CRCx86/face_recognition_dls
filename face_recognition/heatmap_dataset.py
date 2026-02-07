import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path

def gaussian_heatmap(size, center, sigma=2):
    x = np.arange(0, size, 1, float)
    y = x[:, None]
    x0, y0 = center
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

class CelebAHeatmapDataset(Dataset):
    def __init__(self, img_dir, img_size=256, hm_size=64):
        self.img_dir = Path(img_dir)
        self.images = list(self.img_dir.glob("*.jpg"))
        self.img_size = img_size
        self.hm_size = hm_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lm_path = img_path.with_suffix(".jpg.npy")

        img_orig = cv2.imread(str(img_path))
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        h, w = img_orig.shape[:2]

        landmarks = np.load(lm_path).astype(np.float32)  # оригинальные координаты

        # ресайз
        img_resized = cv2.resize(img_orig, (self.img_size, self.img_size))

        # оригинал в img_size
        sx = self.img_size / w
        sy = self.img_size / h
        landmarks_img = landmarks.copy()
        landmarks_img[:, 0] *= sx
        landmarks_img[:, 1] *= sy

        # оригинал в hm_size
        hm_scale = self.hm_size / self.img_size

        heatmaps = np.zeros((landmarks.shape[0], self.hm_size, self.hm_size), dtype=np.float32)
        for i, (x, y) in enumerate(landmarks_img):
            hx = x * hm_scale
            hy = y * hm_scale
            heatmaps[i] = gaussian_heatmap(self.hm_size, (hx, hy))

        img_tensor = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1).float()
        heatmaps = torch.from_numpy(heatmaps)

        return {
            "img": img_tensor,
            "heatmaps": heatmaps,
            "img_path": str(img_path),
            "orig_shape": (h, w),
        }
