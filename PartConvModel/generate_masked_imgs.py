import os
from PIL import Image
import numpy as np
import argparser

# Parameter initialization
img_dir = argparser.out_dir_imgs
mask_dir = argparser.out_dir_masks
masked_img_dir = argparser.out_dir_masked_imgs
size = argparser.size

# Create directory to save generated masked images
if not os.path.exists(masked_img_dir):
    os.makedirs(masked_img_dir)

# Get image and mask paths
img_names = sorted(os.listdir(img_dir))
mask_names = sorted(os.listdir(mask_dir))

print(mask_names)


# Iterate through each image and mask
for img_number, name in enumerate(zip(img_names, mask_names)):
    print(img_number)
    img_name = name[0]
    mask_name = name[1]

    # Open image/mask as numpy arrays
    img_pth = os.path.join(img_dir, img_name)
    img = Image.open(img_pth).convert("RGB")
    img = np.asarray(img, dtype="uint8")

    mask_pth = os.path.join(mask_dir, mask_name)
    #mask = Image.open(mask_pth).convert("L")
    mask = Image.open(mask_pth)
    mask = np.asarray(mask, dtype="uint8")
    mask = mask / 255

    # Create new array to hold masked image
    img_masked = np.zeros((size, size, 3), dtype="uint8")

    # Pixel-wise multiplication
    for i in range(size):
        for j in range(size):
            img_masked[i,j,0] = int(img[i,j,0]) * int(mask[i,j])  # R
            img_masked[i,j,1] = int(img[i,j,1]) * int(mask[i,j])  # G
            img_masked[i,j,2] = int(img[i,j,2]) * int(mask[i,j])  # B
    
    # Save masked img
    img_masked = Image.fromarray(img_masked)
    img_masked.save(os.path.join(masked_img_dir, "{}.jpg".format(img_name)))


