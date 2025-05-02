# train_model.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import segmentation_models_pytorch as smp
from dataset import TiledDataset  # Assuming you saved the dataset code in dataset.py

# Paths
RGB_FILE = '../datasets/ortho_blessem_20210718_rgb.tif'
MASK_FILE = '../datasets/ortho_blessem_20210718_mask.tif'
TILE_COORDS_PATH = '../datasets/tile_coords.npy'
SAVE_PATH = '../models/deeplabv3plus_best.pth'

# Parameters
TILE_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 2025
TEST_RATIO = 0.1
VAL_RATIO = 0.1

def get_dataloaders():
    tile_coords = np.load(TILE_COORDS_PATH) if os.path.exists(TILE_COORDS_PATH) else None

    train_dataset = TiledDataset(
        rgb_file=RGB_FILE,
        mask_file=MASK_FILE,
        tile_size=TILE_SIZE,
        transform=None,
        subset='train',
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        tile_coords=tile_coords,
    )

    val_dataset = TiledDataset(
        rgb_file=RGB_FILE,
        mask_file=MASK_FILE,
        tile_size=TILE_SIZE,
        transform=None,
        subset='val',
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        tile_coords=tile_coords
    )

    test_dataset = TiledDataset(
        rgb_file=RGB_FILE,
        mask_file=MASK_FILE,
        tile_size=TILE_SIZE,
        transform=None,
        subset='test',
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        tile_coords=tile_coords
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader

def train():
    train_loader, val_loader, test_loader = get_dataloaders()

    model = smp.DeepLabV3Plus(
        encoder_name="se_resnet50",
        encoder_weights="imagenet",
        classes=1,
        activation=None  # Use with BCEWithLogitsLoss
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        for batch in loop:
            inputs = batch['tensor'].to(DEVICE)
            masks = batch['mask'].float().to(DEVICE)  # Shape: (B, 1, H, W)
            masks = masks.squeeze(1) if masks.shape[1] == 1 else masks  # Handle single-channel

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)  # Output shape: (B, H, W)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['tensor'].to(DEVICE)
                masks = batch['mask'].float().to(DEVICE)
                masks = masks.squeeze(1)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, masks)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Average Test Loss: {avg_test_loss:.4f}")

        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved new best model at epoch {epoch+1} with loss {best_loss:.4f}")

if __name__ == '__main__':
    train()
