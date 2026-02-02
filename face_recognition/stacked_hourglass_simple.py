import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.optim as optim

from face_recognition.dataset import CelebAHeatmapDataset


class Residual(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class Hourglass(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.down = nn.Sequential(
            Residual(ch),
            nn.MaxPool2d(2)
        )
        self.mid = Residual(ch)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        d = self.down(x)
        m = self.mid(d)
        u = self.up(m)
        return u + x


class StackedHourglassSimple(nn.Module):
    def __init__(self, n_points=5, stacks=3):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Residual(64),
            Residual(64)
        )

        self.hgs = nn.ModuleList([Hourglass(64) for _ in range(stacks)])
        self.heads = nn.ModuleList([
            nn.Conv2d(64, n_points, 1) for _ in range(stacks)
        ])

    def forward(self, x):
        x = self.pre(x)
        outputs = []
        for hg, head in zip(self.hgs, self.heads):
            x = hg(x)
            outputs.append(head(x))
        return outputs  # intermediate supervision


dataset = CelebAHeatmapDataset(
    "E:\\Deep Learning School\\FR\\result\\cropped"
)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=4
)

def run_train(model, epoch):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epoch):
        model.train()
        total_loss = 0

        for imgs, hms in loader:
            imgs = imgs.cuda()
            hms = hms.cuda()

            predictions = model(imgs)
            loss = sum(criterion(p, hms) for p in predictions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "hourglass_landmarks_simple.pth")









