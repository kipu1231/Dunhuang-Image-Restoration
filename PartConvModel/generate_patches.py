import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import sys
import os
from math import floor
import pickle as pk
import argparser


# ---------------------------------------------------------------
#           Function to draw patches on original image
# ---------------------------------------------------------------

def drawOutline(img, x, y, size):
    
    new_img = img.copy()

    # Columns
    dim = (size, 2*thickness, 3)
    column = np.zeros(dim, dtype="uint8")
    column[:,:,0] = 255
    
    # Left column
    new_img[(y - half_size):(y + half_size),
            (x - half_size - 0):(x - half_size + 2*thickness),
            :] = column
    
    # Right column
    new_img[(y - half_size):(y + half_size),
            (x + half_size - 2*thickness):(x + half_size + 0),
            :] = column

    # Rows
    dim = (2*thickness, size, 3)
    row = np.zeros(dim, dtype="uint8")
    row[:,:,0] = 255

    # Bottom row
    new_img[(y - half_size - 0):(y - half_size + 2*thickness),
            (x - half_size):(x + half_size),
            :] = row

    # Top row
    new_img[(y + half_size - 2*thickness):(y + half_size + 0),
            (x - half_size):(x + half_size),
            :] = row

    return new_img


# ---------------------------------------------------------------
#                         Patch generator
# ---------------------------------------------------------------

# Parameter initialization
N = argparser.N                         # number of patches
size = argparser.size                   # patch size = (H, W) = (size, size)
half_size = int(size/2)                 # half the patch size
thresh = argparser.thresh               # distance threshold between patch centers
thickness = argparser.thickness         # border thickness for patch visualization
img_dir = argparser.img_dir             # directory of images to generate patches for
out_dir_imgs = argparser.out_dir_imgs   # directory to save generated patches
out_dir_viz = argparser.out_dir_viz     # directory to save generated visualizations
out_dir_info = argparser.out_dir_info   # directory to save patch info
mode = argparser.mode                   # generate patches for training or testing images

# Check for positive number of patches
if N < 1:
    print("Need to generate >0 patches")
    sys.exit()

# Check for even value of side dimension
if size % 2 != 0:
    print("Dimension of patches must be even")
    sys.exit()

# Create output directories if they do not exist
if not os.path.exists(out_dir_imgs):
    os.makedirs(out_dir_imgs)

if not os.path.exists(out_dir_viz):
    os.makedirs(out_dir_viz)

# Get paths to each of the images
img_names = sorted(os.listdir(img_dir))


# ---------------------------------------------------------------
#                        Train Images
# ---------------------------------------------------------------

if mode == "train":

    # Iterate through each image in directory
    for img_number, name in enumerate(img_names):

        # Open image
        img_pth = os.path.join(img_dir, name)
        img_orig = Image.open(img_pth).convert("RGB")
        img_orig = np.asarray(img_orig, dtype="uint8")
        img_draw = img_orig.copy()

        # Check if the patch size exceeds image size
        if (size >= img_orig.shape[0]) or (size >= img_orig.shape[1]):
            print("Patch size larger than current image")
            continue
        
        # Initialization
        x_centers = []      # x-coordinates of all patch centers for current image
        y_centers = []      # y-coordinates of all patch centers for current image

        # Generate N patches for current image
        for i in range(N):
            
            # Randomly pick a pixel in the image to be the patch center
            x_center = random.randint(0+half_size, img_orig.shape[1]-half_size-1)
            y_center = random.randint(0+half_size, img_orig.shape[0]-half_size-1)
            
            # Keep generating a patch center until is far enough away from all previous patch centers
            j=0
            too_close = True
            while (too_close):

                # Assume the current pixel is far away from all other patch centers
                too_close = False

                # Calculate distance to all patch centers
                for x, y in zip(x_centers, y_centers):
                    dist = np.sqrt((x - x_center)**2 + (y - y_center)**2)

                    # Current pixel is too close to another patch center. Generate another random pixel and try again
                    if (dist < thresh):
                        x_center = random.randint(0+half_size, img_orig.shape[1]-half_size-1)
                        y_center = random.randint(0+half_size, img_orig.shape[0]-half_size-1)
                        too_close = True
                        break         
                
                j += 1

            print("[Img: %d, Patch: %d, Iters to find center: %d]" % (img_number+1, i+1, j))

            # Append new patch center to lists
            x_centers.append(x_center)
            y_centers.append(y_center)

            # Add and subtract (size/2) from the patch center to generate the patch
            patch = img_orig[(y_center-half_size):(y_center+half_size),
                             (x_center-half_size):(x_center+half_size),
                             :]

            # Update viualization
            img_draw = drawOutline(img_draw, x_center, y_center, size)

            # Save patch
            patch_img = Image.fromarray(patch)
            patch_img.save(os.path.join(out_dir_imgs, "{}_{}.jpg".format(name[0:3], i+1)))

        # Save patch visualization for current image
        viz = Image.fromarray(img_draw)
        viz.save(os.path.join(out_dir_viz, "{}.jpg".format(name[0:3])))



# ---------------------------------------------------------------
#                         Test Images
# ---------------------------------------------------------------

elif mode == "test":

    # Create save directory if it does not exist
    if not os.path.exists(out_dir_info):
        os.makedirs(out_dir_info)

    # Iterate through each image in directory
    for img_number, name in enumerate(img_names[0:2]):

        # Open image
        img_pth = os.path.join(img_dir, name)
        img_orig = Image.open(img_pth).convert("RGB")
        img_orig = np.asarray(img_orig, dtype="uint8")
        img_draw = img_orig.copy()

        # Check if the patch size exceeds image size
        if (size >= img_orig.shape[0]) or (size >= img_orig.shape[1]):
            print("Patch size larger than current image")
            continue
        
        # Get rows and cols of current image
        n_rows = img_orig.shape[0]
        n_cols = img_orig.shape[1]

        # Integer division to get number of patches
        q1 = floor(n_rows/size)
        q2 = floor(n_cols/size)
        rows = q1 + 1           # number of patches going vertically
        cols = q2 + 1           # number of patches going horizontally across the image
        N = rows * cols         # number of patches to split current image into
        
        # Generate patch centers
        x_centers = np.linspace((0 + half_size), (n_cols - half_size), cols, dtype="uint16")
        y_centers = np.linspace((0 + half_size), (n_rows - half_size), rows, dtype="uint16")

        # Generate N patches for current image, going left->right and top->bottom
        iters = 0
        for y_center in y_centers:
            for x_center in x_centers:
                
                # Add and subtract (size/2) from the current patch center to generate the patch
                patch = img_orig[(y_center-half_size):(y_center+half_size),
                                 (x_center-half_size):(x_center+half_size),
                                 :]
                
                print("[Img: %d, Patch: %d]" % (img_number+1, iters+1))

                # Update viualization
                img_draw = drawOutline(img_draw, x_center, y_center, size)

                # Save patch
                patch_img = Image.fromarray(patch)
                patch_img.save(os.path.join(out_dir_imgs, "{}_{}.jpg".format(name[0:3], iters+1)))

                # Save patch center
                with open(os.path.join(out_dir_info, "{}_{}.pk".format(name[0:3], iters+1)), "wb") as f:
                    pk.dump([x_center, y_center, rows, cols], f)
                
                iters += 1
            
        # Save patch visualization for current image
        viz = Image.fromarray(img_draw)
        viz.save(os.path.join(out_dir_viz, "{}.jpg".format(name[0:3])))