import torch.nn as nn
from tqdm import tqdm  # индикатор прогресса

import torch
import torch.optim as optim

from face_recognition.heatmap_dataset import CelebAHeatmapDataset

from torch.utils.data import DataLoader, random_split


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


def run_train(model, train_loader, val_loader, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            imgs = batch["img"].cuda()
            hms = batch["heatmaps"].cuda()

            predications = model(imgs)
            loss = sum(criterion(p, hms) for p in predications)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["img"].cuda()
                hms = batch["heatmaps"].cuda()

                predictions = model(imgs)
                loss = sum(criterion(p, hms) for p in predictions)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "models/hourglass/hourglass_landmarks_best.pth")
                print(f"Saved best model with val loss: {best_val_loss:.6f}")

    torch.save(model.state_dict(), "models/hourglass/hourglass_landmarks.pth")
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    criterion = nn.MSELoss()

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["img"].cuda()
            hms = batch["heatmaps"].cuda()

            predictions = model(imgs)
            # Use only the final prediction for evaluation
            loss = criterion(predictions[-1], hms)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.6f}")
    return avg_test_loss


if __name__ == "__main__":
    model = StackedHourglass().cuda()

    dataset = CelebAHeatmapDataset(
        "E:\\Deep Learning School\\FR\\result\\cropped"
    )

    # Split dataset: 70% train, 15% validation, 15% test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    print(f"Dataset sizes: Train={train_size}, Val={val_size}, Test={test_size}")

    # Initialize model
    model = StackedHourglass().cuda()

    # Train model
    epochs = 25
    run_train(model, train_loader, val_loader, epochs)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss = evaluate_model(model, test_loader)
    # run_test(model)
    # run_tests(model)