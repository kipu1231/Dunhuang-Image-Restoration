import os
import pickle as pk
import numpy as np
from PIL import Image
import sys
import argparser
import glob
import torch
import torchvision.transforms as transforms


# Reconstructs the patches for 1 image
# patches = list of patches (numpy arrays)
# H = height of original image
# W = width of original image
# img_num = which image we are reconstructing (start from 0)
def reconstruct(patches, H, W, img_num, center_info, args):
    out_dir_info = args.out_dir_info
    size = args.size
    half_size = int(size / 2)
    N = len(patches)

    #reshape tensor and change to numpy array
    #patches = patches.permute(0, 2, 3, 1).numpy()
    patches = patches.numpy()

    #get path to filenames
    filenames = sorted(glob.glob(os.path.join(out_dir_info, '{}_masked.jpg*'.format(img_num))))

    #reconstructed_img = np.zeros((H, W, 3))
    #divisor = np.zeros((H, W, 3))
    reconstructed_img = np.zeros((3, H, W))
    divisor = np.zeros((3, H, W))
    x_centers = []
    y_centers = []

    # Cycle through all patches for one image
    for idx in range(len(center_info)):
        x_centers.append(int(center_info[idx][0]))
        y_centers.append(int(center_info[idx][1]))

    # for fname in filenames:
    #     with open(fname, "rb") as f:
    #         patch_info = pk.load(f) # [x_center of current patch, y_center of current patch, rows that current image was divided into, cols that current image was divided into]
    #         x_centers.append(patch_info[0])
    #         y_centers.append(patch_info[1])

    iters = 0
    for x_center, y_center in zip(x_centers, y_centers):
        # print(x_center)
        # print(y_center)
        # Add and subtract (size/2) from the current patch center to generate the patch. Add patch values to whatever is currently there
        # reconstructed_img[(y_center-half_size):(y_center+half_size),
        #                     (x_center-half_size):(x_center+half_size), :] = np.maximum.reduce(reconstructed_img[(y_center-half_size):(y_center+half_size),
        #                                                                                       (x_center-half_size):(x_center+half_size), :], patches[iters])

        #reconstructed_img[(y_center - half_size):(y_center + half_size), (x_center - half_size):(x_center + half_size), :] += patches[iters]
        #divisor[(y_center - half_size):(y_center + half_size), (x_center - half_size):(x_center + half_size),:] += 1

        reconstructed_img[:, (y_center - half_size):(y_center + half_size), (x_center - half_size):(x_center + half_size)] += patches[iters]
        divisor[:,(y_center - half_size):(y_center + half_size), (x_center - half_size):(x_center + half_size)] += 1

        iters += 1

    # Save reconstructed image
    reconstructed_img = reconstructed_img/divisor
    #rec_img = torch.from_numpy(reconstructed_img)
    #reconstructed_img = rec_img.permute(0, 2, 3, 1).numpy()

    reconstructed_img = Image.fromarray((np.transpose(reconstructed_img, (1, 2, 0))*255).astype('uint8'))

    return reconstructed_img



# out_dir_imgs = argparser.out_dir_imgs
# if not os.path.exists(out_dir_imgs):
#     print("Couldn't find saved patches")
#     sys.exit()
#
# # Get paths to each of the images
# patch_names = sorted(os.listdir(out_dir_imgs))
# patches_list = []
#
# # Iterate through each image in directory
# for patch_name in patch_names[0:9]:
#
#     # Open image
#     pth = os.path.join(out_dir_imgs, patch_name)
#     patch = Image.open(pth).convert("RGB")
#     patch = np.asarray(patch, dtype="uint8")
#
#     patches_list.append(patch)
#
# img_num = 0
# img_num = reconstruct(patches_list, 686, 693, img_num)
#
#
# patches_list = []
# for patch_name in patch_names[9:21]:
#
#     # Open image
#     pth = os.path.join(out_dir_imgs, patch_name)
#     patch = Image.open(pth).convert("RGB")
#     patch = np.asarray(patch, dtype="uint8")
#
#     patches_list.append(patch)
#
# img_num = reconstruct(patches_list, 719, 797, img_num)