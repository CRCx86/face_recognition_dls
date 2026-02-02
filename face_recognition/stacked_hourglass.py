import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.optim as optim

from face_recognition.dataset import CelebAHeatmapDataset


dataset = CelebAHeatmapDataset(
    "E:\\Deep Learning School\\FR\\result\\cropped"
)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=4
)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return self.relu(x + residual)


class Hourglass(nn.Module):
    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth

        self.res1 = ResidualBlock(channels, channels)
        self.pool = nn.MaxPool2d(2)

        self.res2 = ResidualBlock(channels, channels)

        if depth > 1:
            self.hg = Hourglass(depth - 1, channels)
        else:
            self.res_mid = ResidualBlock(channels, channels)

        self.res3 = ResidualBlock(channels, channels)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.res1(x)

        low = self.pool(x)
        low = self.res2(low)

        if self.depth > 1:
            low = self.hg(low)
        else:
            low = self.res_mid(low)

        low = self.res3(low)
        up2 = self.up(low)

        return up1 + up2


class StackedHourglass(nn.Module):
    def __init__(self, n_points=5, stacks=3, channels=64):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(channels, channels),
            ResidualBlock(channels, channels),
        )

        self.hgs = nn.ModuleList([
            Hourglass(depth=2, channels=channels)
            for _ in range(stacks)
        ])

        self.features = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(channels, channels),
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
            )
            for _ in range(stacks)
        ])

        self.heads = nn.ModuleList([
            nn.Conv2d(channels, n_points, 1)
            for _ in range(stacks)
        ])

        self.merge_predications = nn.ModuleList([
            nn.Conv2d(n_points, channels, 1)
            for _ in range(stacks - 1)
        ])

        self.merge_features = nn.ModuleList([
            nn.Conv2d(channels, channels, 1)
            for _ in range(stacks - 1)
        ])

    def forward(self, x):
        x = self.pre(x)
        outputs = []

        for i in range(len(self.hgs)):
            hg = self.hgs[i](x)
            feat = self.features[i](hg)
            out = self.heads[i](feat)

            outputs.append(out)

            if i < len(self.hgs) - 1:
                x = x + self.merge_features[i](feat) + self.merge_predications[i](out)

        return outputs


def run_train(model, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, hms in loader:
            imgs = imgs.cuda()
            hms = hms.cuda()

            predications = model(imgs)
            loss = sum(criterion(p, hms) for p in predications)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "hourglass_landmarks.pth")
