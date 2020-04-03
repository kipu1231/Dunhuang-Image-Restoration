import os
import glob
import sys
import random
from PIL import Image
import numpy as np
from math import floor
import pickle as pk
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import argparser


class DATA(Dataset):
    def __init__(self, mode, train_status):
        
        # Load parsed arguments
        args = argparser.arg_parse()
        mean = argparser.mean
        std = argparser.std

        # Set directories
        self.mode = mode
        self.train_status = train_status
        self.test_info_dir = args.out_dir_info
        self.size = args.size

        #distinguish wether model is pre-trained or fine tuned
        if train_status == 'pretrain':
            self.data_dir = args.data_dir_pre_train
        else:
            self.data_dir = args.data_dir_fine_tune

        #set directories
        if self.mode == "train":
            gt_dir = os.path.join(self.data_dir, "imgs")
            masked_imgs_dir = os.path.join(self.data_dir, "masked_imgs")
            masks_dir = os.path.join(self.data_dir, "mask")
        elif self.mode == "test" and self.train_status == 'TA':
            masked_imgs_dir = os.path.join(args.data_dir_test)
        elif self.mode == "test":
            masked_imgs_dir = os.path.join(args.data_dir_test, "test")
            gt_dir = os.path.join(args.data_dir_test, "test_gt")
        else:
            print("Invalid mode in dataloader")
            sys.exit()

        # Get paths to each image. Length = number of images in directory (e.g. 400 for training)
        if self.mode ==  "train":
            self.masks = sorted(glob.glob(os.path.join(masks_dir, '*.png')))
            self.masked_imgs = sorted(glob.glob(os.path.join(masked_imgs_dir, '*.jpg')))
        elif self.mode == "test":
            self.masked_imgs = sorted(glob.glob(os.path.join(masked_imgs_dir, '*masked.jpg')))
            self.masks = sorted(glob.glob(os.path.join(masked_imgs_dir, '*mask.jpg')))

        if not train_status == 'TA':
            self.gt_imgs = sorted(glob.glob(os.path.join(gt_dir, '*.jpg')))

        # Define transform        
        self.transformT = transforms.Compose([transforms.ToTensor()])
        self.transformN = transforms.Compose([transforms.Normalize(mean, std)])

    # Perform data augmentation on images
    def augment(self, masked, mask, gt):

        # 0.5 probability of performing one transformation
        if random.random() > 0.5:

            p = random.random()

            # Rotate
            if p < 0.5:
                if p < 0.25:
                    angle = -90
                else:
                    angle = 90
                masked = TF.rotate(masked, angle)
                mask = TF.rotate(mask, angle)
                gt = TF.rotate(gt, angle)

            # Horizontal flip
            elif p >= 0.5:
                TF.hflip(masked)
                TF.hflip(mask)
                TF.hflip(gt)

            # Colours
            # else:
            #     TF.adjust_brightness(masked, 1.4)
            #     TF.adjust_hue(masked, 0.3)
            #     TF.adjust_contrast(masked, 1.3)
            #     TF.adjust_saturation(masked, 0.7)

            #     # TF.adjust_brightness(mask, 1.4)
            #     # TF.adjust_hue(mask, 0.3)
            #     # TF.adjust_contrast(mask, 1.3)
            #     # TF.adjust_saturation(mask, 0.7)

            #     TF.adjust_brightness(gt, 1.4)
            #     TF.adjust_hue(gt, 0.3)
            #     TF.adjust_contrast(gt, 1.3)
            #     TF.adjust_saturation(gt, 0.7)

        return masked, mask, gt

    def patchTestData(self,  masked, masked_name, type):
        # Create save directory if it does not exist
        #if not os.path.exists(self.test_info_dir):
        #   os.makedirs(self.test_info_dir)

        name = os.path.split(masked_name)

        img_orig = np.asarray(masked, dtype="uint8")
        img_draw = img_orig.copy()

        # Check if the patch size exceeds image size
        if (self.size >= img_orig.shape[0]) or (self.size >= img_orig.shape[1]):
            print("Patch size larger than current image")

        # Get rows and cols of current image
        n_rows = img_orig.shape[0]
        n_cols = img_orig.shape[1]

        # Integer division to get number of patches
        q1 = floor(n_rows / self.size)
        q2 = floor(n_cols /self.size)
        rows = q1 + 1  # number of patches going vertically
        cols = q2 + 1  # number of patches going horizontally across the image
        N = rows * cols  # number of patches to split current image into

        # Generate patch centers
        half_size = int(self.size / 2)
        x_centers = np.linspace((0 + half_size), (n_cols - half_size), cols, dtype="uint16")
        y_centers = np.linspace((0 + half_size), (n_rows - half_size), rows, dtype="uint16")

        # Generate N patches for current image, going left->right and top->bottom
        iters = 0
        patches = []
        center = []
        for y_center in y_centers:
            for x_center in x_centers:
                # Add and subtract (size/2) from the current patch center to generate the patch
                patch = img_orig[(y_center - half_size):(y_center + half_size),
                        (x_center - half_size):(x_center + half_size), :]
                patches.append(self.transformT(Image.fromarray(np.uint8(patch))).unsqueeze(0))

                center.append([int(x_center), int(y_center)])

                iters += 1

        patches = torch.cat(patches)
        img_info = {"name": name[1], "Width": n_cols, "Heigth": n_rows, "center_info": center}

        return patches, img_info

    def __getitem__(self, idx):
        # Get paths
        mask = self.masks[idx]
        masked = self.masked_imgs[idx]

        if not self.train_status == 'TA':
            gt = self.gt_imgs[idx]
            gt = Image.open(gt).convert("RGB")

        # Load paths as PIL images
        mask = np.asarray(Image.open(mask), dtype="uint8")

        #preprocess of masks since format is JPEG
        if self.mode == 'test':
            mask = mask / 255
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j] != 0 and mask[i][j] != 1:
                        if mask[i][j] > 0.5:
                            mask[i][j] = 1
                        else:
                            mask[i][j] = 0
            mask = mask * 255

        mask = Image.fromarray(mask).convert("RGB")
        masked = Image.open(masked).convert("RGB")

        if self.mode == "train":
            #data augmentation
            masked, mask, gt = self.augment(masked, mask, gt)

            # transformations to tensor
            masked = self.transformT(masked)
            mask = self.transformT(mask)
            gt = self.transformT(gt)

            return masked, mask, gt

        elif self.mode == "test":
            masked, img_info = self.patchTestData(masked, self.masked_imgs[idx], 'RGB')
            mask, _ = self.patchTestData(mask, self.masks[idx], 'RGB')

            if not self.train_status == 'TA':
                gt = self.transformT(gt)

                return masked, mask, gt, img_info

            else:
                return masked, mask, img_info

    def __len__(self):
        return len(self.masked_imgs)