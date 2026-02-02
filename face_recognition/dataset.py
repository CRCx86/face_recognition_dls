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

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        landmarks = np.load(lm_path).astype(np.float32)

        # resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        scale_x = self.hm_size / w
        scale_y = self.hm_size / h

        heatmaps = np.zeros((5, self.hm_size, self.hm_size), dtype=np.float32)

        for i, (x, y) in enumerate(landmarks):
            hx = x * scale_x
            hy = y * scale_y
            heatmaps[i] = gaussian_heatmap(self.hm_size, (hx, hy))

        img = torch.tensor(img / 255.0).permute(2, 0, 1).float()
        heatmaps = torch.tensor(heatmaps)

        return img, heatmaps
