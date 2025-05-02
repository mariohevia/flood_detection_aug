import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Paths to the rgb and mask images
rgb_file_path = '../datasets/ortho_blessem_20210718_rgb.tif'
mask_file_path = '../datasets/ortho_blessem_20210718_mask.tif'

with rasterio.open(rgb_file_path) as src, \
    rasterio.open(mask_file_path) as src_mask:
    # Check that the image and the mask are both the same size
    assert src_mask.width == src.width and src_mask.height == src.height

    # Define the shape of the output to be read. It will use a lower resolution.
    out_height = 512
    out_width = out_height * src.width // src.height # Maintains the aspect ratio
    out_shape = (3, out_height, out_width)
    scaled_img = src.read(out_shape=out_shape)
    mask_out_shape = (1, out_height, out_width)
    scaled_mask = src_mask.read(out_shape=mask_out_shape)

    print(f"Original image shape: {src.shape}")
    print(f"Scaled image shape: {scaled_img.shape}")

    # Transpose to HWC for matplotlib
    scaled_img = np.transpose(scaled_img, (1, 2, 0))
    scaled_mask = np.transpose(scaled_mask, (1, 2, 0))

    # Display the image
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(scaled_img)
    plt.title('Scaled image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(scaled_mask, cmap='gray')
    plt.title('Scaled mask')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(scaled_img)
    plt.imshow(scaled_mask, alpha=0.6, cmap='gray')
    plt.title('Overlaid image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../img/initial_load.png',transparent=True)
    plt.close()

    ### Scaling pixel values from 0-65535 to 0-1
    scaled_pixels_img = (scaled_img / 65535)
    # Display the window
    plt.imshow(scaled_pixels_img)
    plt.axis('off')
    plt.savefig('../img/scaling_bands.png',transparent=True)
    plt.close()

    ### Getting statistics of the pixel values

    # Read all the image bands (band 1=R, 2=G, 3=B)
    r = src.read(1)
    g = src.read(2)
    b = src.read(3)

    # Calculate 2nd and 98th percentiles for each band/channel
    p2_r, p98_r = np.percentile(r, [2, 98])
    p2_g, p98_g = np.percentile(g, [2, 98])
    p2_b, p98_b = np.percentile(b, [2, 98])

    print(f"Red 2nd and 98th percentiles: {p2_r}, {p98_r}")
    print(f"Green 2nd and 98th percentiles: {p2_g}, {p98_g}")
    print(f"Blue 2nd and 98th percentiles: {p2_b}, {p98_b}")

    # Using the scaled image with lower resolution
    percentiles_2 = np.percentile(scaled_img, 2, axis=(0, 1))
    percentiles_98 = np.percentile(scaled_img, 98, axis=(0, 1))
    print(f"2nd percentiles: {percentiles_2}")
    print(f"98th percentiles: {percentiles_98}")

    # Normalize and clip for display
    final_img = (scaled_img - percentiles_2) / (percentiles_98 - percentiles_2)
    final_img = np.clip(final_img, 0, 1)

    # Display the window
    plt.imshow(final_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../img/final_load.png',transparent=True)
    plt.close()
