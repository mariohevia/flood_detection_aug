# test_model.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score, accuracy
from dataset import TiledDataset
from tqdm import tqdm

# Paths
RGB_FILE = '../datasets/ortho_blessem_20210718_rgb.tif'
MASK_FILE = '../datasets/ortho_blessem_20210718_mask.tif'
TILE_COORDS_PATH = '../datasets/tile_coords.npy'
MODEL_PATH = '../models/deeplabv3plus_best.pth'

# Parameters
TILE_SIZE = 512
BATCH_SIZE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 2025
TEST_RATIO = 0.1
VAL_RATIO = 0.1

model = smp.DeepLabV3Plus(
    encoder_name="se_resnet50",
    encoder_weights=None,
    classes=1,
    activation=None
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def get_test_loader():
    tile_coords = np.load(TILE_COORDS_PATH) if os.path.exists(TILE_COORDS_PATH) else None

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

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return test_loader

def test():
    test_loader = get_test_loader()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch['tensor'].to(DEVICE)
            masks = batch['mask'].to(torch.uint8).to(DEVICE)
            masks = masks.squeeze(1) if masks.shape[1] == 1 else masks

            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)

            tp, fp, fn, tn = get_stats(probs, masks, mode='binary', threshold=0.5)

            total_tp += tp.sum()
            total_fp += fp.sum()
            total_fn += fn.sum()
            total_tn += tn.sum()

    # Compute metrics
    iou = iou_score(total_tp, total_fp, total_fn, total_tn, reduction='micro')
    dice = f1_score(total_tp, total_fp, total_fn, total_tn, reduction='micro')
    acc = accuracy(total_tp, total_fp, total_fn, total_tn, reduction='micro')

    print(f"Average IoU: {100 * iou:.2f}%")
    print(f"Average Dice: {100 * dice:.2f}%")
    print(f"Average Accuracy: {100 * acc:.2f}%")

def predict(image, device="cuda"):
    # Preprocess
    x = image.unsqueeze(0)  # Add batch dimension
    x = x.to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model(x)
    
    return prediction  # Values between 0-1 for binary segmentation

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    # Runs the test in all the testing dataset
    # test()

    # Opens a random tile in the dataset
    def read_random_tile():
        # Paths to the rgb and mask images
        rgb_file_path = '../datasets/ortho_blessem_20210718_rgb.tif'
        mask_file_path = '../datasets/ortho_blessem_20210718_mask.tif'

        with rasterio.open(rgb_file_path) as src, \
            rasterio.open(mask_file_path) as src_mask:
            # Obtaining percentiles with low resolution image
            out_height = 512
            out_width = out_height * src.width // src.height  # Maintains the aspect ratio
            out_shape = (3, out_height, out_width)
            scaled_img = src.read(out_shape=out_shape)  # Reads the image in low resolution
            percentiles_2 = np.percentile(scaled_img, 2, axis=(1, 2)).reshape(-1, 1, 1)  # Shape: (3, 1, 1)
            percentiles_98 = np.percentile(scaled_img, 98, axis=(1, 2)).reshape(-1, 1, 1)  # Shape: (3, 1, 1)

            # Loading a random tile of size 512x512
            tile_size = 512
            x_start = np.random.randint(0, src.width - tile_size + 1)
            y_start = np.random.randint(0, src.height - tile_size + 1)
            print(x_start, y_start)
            window = Window(x_start, y_start, tile_size, tile_size)

            # Read the RGB tile - will be in (C,H,W) format
            image = src.read(window=window)  # Shape: (3, 512, 512)

            # Normalise and clip (applying to each channel)
            final_img = (image - percentiles_2) / (percentiles_98 - percentiles_2)
            final_img = np.clip(final_img, 0, 1)

            # Read the mask tile - will be in (C,H,W) format
            mask_image = src_mask.read(window=window)

            # Normalise and transform to tensor for DL model
            mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
            preprocessed_img = (final_img - mean) / std
            tensor_img = torch.tensor(preprocessed_img).float()

            sample = {'image':final_img, 'tensor':tensor_img, 'mask':mask_image}
        return sample
    
    # Predicts the mask for one random tile
    sample = read_random_tile()
    predictions = predict(sample['tensor'])
    threshold = 0.5
    predictions = (predictions > threshold).float()
    prediction_plt = np.transpose(predictions[0].cpu().numpy(), (1, 2, 0))

    # Visualize the tile
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(np.transpose(sample['image'], (1, 2, 0)))
    axes[0].set_title("Dataset Tile")
    axes[0].axis('off')
    axes[1].imshow(prediction_plt, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    axes[2].imshow(np.transpose(sample['mask'], (1, 2, 0)), cmap='gray')
    axes[2].set_title("Real Mask")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('../img/trained_prediction.png',transparent=True)
    plt.close()
