# dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import rasterio
from rasterio.windows import Window

class TiledDataset(Dataset):
    """Dataset that loads tiles from large TIFF files using Rasterio."""

    def __init__(
        self, 
        rgb_file, 
        mask_file, 
        tile_size=512, 
        transform=None, 
        subset='train', 
        test_ratio=0.1, 
        val_ratio=0.1, 
        seed=None,
        tile_coords=None
        ):
        """
        Args:
            rgb_file (string): Path to the large RGB .tif file.
            mask_file (string): Path to the large mask .tif file.
            tile_size (int, optional): Size of the square tiles to extract (e.g., 512).
            transform (callable, optional): Optional transform to be applied on a sample.
            subset (string, optional): What type of dataset is required ('train', 'val', 'test')
            test_ratio (float, optional): Ratio of the dataset to use for testing (between 0 and 1).
            val_ratio (float, optional): Ratio of the dataset to use for validation (between 0 and 1).
            seed (int, optional): Seed for sampling the train/test subsets.
            tile_coords (numpy, optional): Optional array of tile coordinates.
        """
        self.rgb_file = rgb_file
        self.mask_file = mask_file
        self.tile_size = tile_size
        self.transform = transform

        with rasterio.open(self.rgb_file) as src, \
            rasterio.open(self.mask_file) as src_mask:

            assert src_mask.width == src.width and src_mask.height == src.height

            # Load a lower resolution version to compute the 2nd and 98th percentiles.
            out_height = 512
            out_width = out_height * src.width // src.height # Maintains the aspect ratio
            out_shape = (3, out_height, out_width)
            scaled_img = src.read(out_shape=out_shape) # Reads the image in low resolution

            self.percentiles_2 = np.percentile(scaled_img, 2, axis=(1, 2)).reshape(-1, 1, 1) 
            self.percentiles_98 = np.percentile(scaled_img, 98, axis=(1, 2)).reshape(-1, 1, 1) 
            self.rgb_width = src.width
            self.rgb_height = src.height
            self.mask_width = src_mask.width
            self.mask_height = src_mask.height

        self.tile_coords = tile_coords
        self._create_tile_coords()
        self.water_tile_coords = self.tile_coords[self.tile_coords[:, 2] == 1][:, :2]
        self.non_water_tile_coords = self.tile_coords[self.tile_coords[:, 2] == 0][:, :2]

        # Create controlled splits
        if seed is not None:
            torch.manual_seed(seed)
        ratios = [1-test_ratio-val_ratio, val_ratio, test_ratio]
        water_train, water_val, water_test = random_split(self.water_tile_coords, ratios)
        non_water_train, non_water_val, non_water_test = random_split(self.non_water_tile_coords, ratios)
        if seed is not None:
            torch.manual_seed(torch.initial_seed())
        match subset:
            case 'train':
                self.tile_coords = water_train + non_water_train
            case 'test':
                self.tile_coords = water_test + non_water_test
            case 'val':
                self.tile_coords = water_val + non_water_val
            case _ :
                raise ValueError(f"Unknown subset {subset}")

    def _create_tile_coords(self):
        if self.tile_coords is None:
            non_water_tile_coords = []
            water_tile_coords = []
            with rasterio.open(self.rgb_file) as src, \
                rasterio.open(self.mask_file) as src_mask:
                for x_start in range(0, src.height, self.tile_size):
                    for y_start in range(0, src.width, self.tile_size):
                        # Define the window (tile)
                        window = Window(x_start, y_start, self.tile_size, self.tile_size)
                        image = src.read(window=window)
                        mask = src_mask.read(window=window)
                        match (bool(np.all(image == 0)), bool(np.all(mask == 0))):
                            case (False, True): # The mask is all black but not the tile
                                non_water_tile_coords.append((x_start, y_start))
                            case (False, False): # Both the tile and the mask are not all black
                                water_tile_coords.append((x_start, y_start))
                            case (False, True):
                                raise ValueError("The tile is all black but the mask indicates water")
                            case (True, True): # Both the tile and the mask are all black
                                pass
                        # if len(water_tile_coords)>1 and len(non_water_tile_coords)>1:
                        #     break
                tile_coords = (
                    [(x, y, 0) for x, y in non_water_tile_coords] + 
                    [(x, y, 1) for x, y in water_tile_coords]
                )
                self.tile_coords = np.array(tile_coords, dtype=np.int32)
                np.save('../datasets/tile_coords.npy', self.tile_coords)

        
    def normalise_tile(self, tile):
        # Normalise and clip
        tile = (tile - self.percentiles_2) / (self.percentiles_98 - self.percentiles_2)
        image = np.clip(tile, 0, 1)
        # Normalise for ImageNet and turn into tensor
        mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
        preprocessed_img = (image - mean) / std
        tensor_img = torch.tensor(preprocessed_img).float()
        return image, tensor_img

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, idx):
        # Get the top-left coordinates of the idx tile
        x_start, y_start = (self.tile_coords)[idx]

        # Define the window (tile) to read
        window = Window(x_start, y_start, self.tile_size, self.tile_size)

        with rasterio.open(self.rgb_file) as src, \
            rasterio.open(self.mask_file) as src_mask:
            # Read the RGB tile
            rgb_image = src.read(window=window)
            image, tensor = self.normalise_tile(rgb_image)

            # Read the mask tile
            mask_image = src_mask.read(window=window)
            sample = {'image':image, 'tensor':tensor, 'mask':mask_image}

        if self.transform:
            transformed = self.transform(**sample)
            sample['image'] = transformed['image']
            sample['tensor'] = transformed['tensor']
            sample['mask'] = transformed['mask']

        return sample

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import os

    rgb_file_path = '../datasets/ortho_blessem_20210718_rgb.tif'
    mask_file_path = '../datasets/ortho_blessem_20210718_mask.tif'
    tile_coords_path = '../datasets/tile_coords.npy'
    if os.path.exists(tile_coords_path):
        tile_coords = np.load(tile_coords_path)
    else:
        tile_coords = None
    tile_size = 512 # Used in the original paper
    batch_size = 20 # How many tiles read at once from the large image
    seed = 2025

    # Create the tiled dataset
    flood_dataset = TiledDataset(
        rgb_file=rgb_file_path,
        mask_file=mask_file_path,
        tile_size=tile_size,
        transform=None,
        test_ratio=0.1,
        seed=seed,
        tile_coords=tile_coords
    )

    # Create the dataloader
    dataloader = DataLoader(flood_dataset, batch_size=batch_size, shuffle=False)

    # Iterate through a batch
    for i, batch in enumerate(dataloader):
        images = batch['image']
        tensors = batch['tensor']
        masks = batch['mask']

        print(f"Batch {i+1}")
        print("Image batch shape:", images.shape)
        print("Tensor batch shape:", tensors.shape)
        print("Mask batch shape:", masks.shape)
        print("Image dtype:", images.dtype)
        print("Tensor dtype:", tensors.dtype)
        print("Mask dtype:", masks.dtype)

        # Visualize a sample from the batch
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(np.transpose(images[0], (1, 2, 0)))
        axes[0].set_title("Tile")
        axes[1].imshow(np.transpose(masks[0], (1, 2, 0)), cmap='gray')
        axes[1].set_title("Mask Tile")
        plt.show()

        if i == 0:  # Just visualize one batch for example
            break