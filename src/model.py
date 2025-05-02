# model.py

import torch
import segmentation_models_pytorch as smp
from torchvision import transforms

# Define model
model = smp.DeepLabV3Plus(
    encoder_name="se_resnet50",
    encoder_weights="imagenet",
    classes=1,  # Binary segmentation
    activation="sigmoid"
)

def predict(image, device="cuda"):
    model.eval()
    model.to(device)
    
    # Preprocess
    x = image.unsqueeze(0)  # Add batch dimension
    x = x.to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model(x)
    
    return prediction  # Values between 0-1 for binary segmentation

def predict_batch(images, device="cuda"):
    model.eval()
    model.to(device)

    x = images.to(device)  # Create a batch tensor

    # Inference
    with torch.no_grad():
        predictions = model(x)

    return predictions  # Values between 0-1 for binary segmentation

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from rasterio.windows import Window

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
    
    sample = read_random_tile()
    predictions = predict(sample['tensor'])
    # predictions = predict_batch(sample['tensor'].unsqueeze(0))
    threshold = 0.5
    predictions = (predictions > threshold).float()
    prediction_plt = np.transpose(predictions[0].cpu().numpy(), (1, 2, 0))

    # Visualize a sample from the batch
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.transpose(sample['image'], (1, 2, 0)))
    axes[0].set_title("Dataset Tile")
    axes[0].axis('off')
    axes[1].imshow(prediction_plt, cmap='gray')
    axes[1].set_title("Mask Tile")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('../img/untrained_prediction.png',transparent=True)
    plt.close()