
import os
import random
import sys
import csv

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import data
import argparser
import model
import evaluate
import loss
import test
import reconstruct_img
from PIL import Image
import torch.nn as nn


# Function to save model
def save_model(model, save_pth):
    checkpoint = model.state_dict()
    torch.save(checkpoint, save_pth)


# Function to load model
def load_model(model, load_pth):
    if torch.cuda.is_available():
        checkpoint = torch.load(load_pth)   
    else:
        checkpoint = torch.load(load_pth,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)


# Function to remove deprecated models
def remove_prev_model(remove_path):
    if os.path.exists(remove_path):
        os.remove(remove_path)


# ------------------------------------------------
#  PRE-TRAINING
# ------------------------------------------------

def pre_train(device, model, model_pre_train_pth):

    print()
    print("***** PRE-TRAINING *****")
    print()

    # ------------
    #  Load Places2 data
    # ------------

    print("---> preparing dataloader...")

    # Training dataloader. Length = dataset size / batch size
    train_dataset = data.DATA(mode="train", train_status="pretrain")
    dataloader_train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=argparser.n_cpu
    )
    print("---> length of training dataset: ", len(train_dataset))

    # Load test images
    test_dataset = data.DATA(mode="test", train_status="test")
    dataloader_test = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=argparser.n_cpu
    )
    print("---> length of test dataset: ", len(test_dataset))

    # -------
    # Model
    # -------

    # load model from checkpoint if available
    if os.path.exists(model_pre_train_pth):
        print("---> Found previously saved {}, loading checkpoint and CONTINUE pre-training"
              .format(args.saved_pre_train_name))
        load_model(model, model_pre_train_pth)
    else:
        print("---> Start pre-training from scratch: no checkpoint found")

    # ----------------
    #  Optimizer
    # ----------------

    # Optimizer
    print("---> preparing optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=argparser.LR)
    criterion = nn.MSELoss()

    # Move model to device
    model.to(device)

    # ----------
    #  Training
    # ----------

    print("---> start training cycle ...")
    with open(os.path.join(args.output_dir, "pretrain_losses.csv"), "w", newline="") as csv_losses:
        with open(os.path.join(args.output_dir, "pretrain_scores.csv"), "w", newline="") as csv_scores:
            writer_losses = csv.writer(csv_losses)
            writer_losses.writerow(["Epoch", "Iteration", "Loss"])

            writer_scores = csv.writer(csv_scores)
            writer_scores.writerow(["Epoch", "Total Loss", "MSE", "SSIM", "Final Score"])

            highest_final_score = 0.0   # the higher the better, combines mse and ssim
            iteration = 0

            for epoch in range(args.pretrain_epochs):

                model.train()

                loss_sum = 0  # store accumulated loss for one epoch

                for idx, (imgs_masked, masks, gts) in enumerate(dataloader_train):
                    # Move to device
                    imgs_masked = imgs_masked.to(device)  # (N, 3, H, W)
                    masks = masks.to(device)  # (N, 3, H, W)
                    gts = gts.to(device)  # (N, 3, H, W)

                    #print("masked images shape: ",imgs_masked.shape) #torch.Size([32, 3, 256, 256])
                    #print("masks shape: ",masks.shape) #torch.Size([32, 1, 256, 256])
                    #print("target images shape: ",gts.shape) #torch.Size([32, 3, 256, 256])

                    # Model forward path => predicted images
                    preds = model(imgs_masked, masks)

                    original_pixels = torch.mul(masks, imgs_masked)
                    ones = torch.ones(masks.size()).cuda()
                    reversed_masks = torch.sub(ones,masks)
                    predicted_pixels = torch.mul(reversed_masks, preds)
                    preds = torch.add(original_pixels, predicted_pixels)

                    # Calculate total loss
                    #train_loss = loss.total_loss(preds, gts)
                    train_loss = criterion(preds, gts)

                    # Execute Back-Propagation
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f]" %
                          (epoch + 1, args.pretrain_epochs, (idx + 1), len(dataloader_train), train_loss), end="")

                    loss_sum += train_loss.item()
                    writer_losses.writerow([epoch+1, iteration+1, train_loss.item()])
                    iteration += 1

                # ------------------
                #  Evaluate & Save Model
                # ------------------

                if (epoch + 1) % args.val_epoch == 0:
                    mse, ssim = test.test(args, model, device, dataloader_test, mode="validate")
                    final_score = 1 - mse / 100 + ssim
                    print("\nMetrics on test set @ epoch {}:".format(epoch+1))
                    print("-> Average MSE:  {:.5f}".format(mse))
                    print("-> Average SSIM: {:.5f}".format(ssim))
                    print("-> Final Score:  {:.5f}".format(final_score))

                    if final_score > highest_final_score:
                        save_model(model, model_pre_train_pth)
                        highest_final_score = final_score

                    writer_scores.writerow([epoch+1, loss_sum, mse, ssim, final_score])

                save_model(model, os.path.join(args.model_dir_pre_train, "Net_pretrain_epoch{}.pth.tar".format(epoch + 1)))
                if epoch > 0:
                    remove_prev_model(os.path.join(args.model_dir_pre_train, "Net_pretrain_epoch{}.pth.tar".format(epoch)))

    print("\n***** Pre-Training FINISHED *****")


# ------------------------------------------------
#  FINE-TUNING
# ------------------------------------------------

def fine_tune(device, model, model_pre_train_pth, model_fine_tune_pth):

    print()
    print("***** Start FINE-TUNING *****")
    print()

    # ------------
    #  Load Dunhuang Grottoes data
    # ------------

    print("---> preparing dataloader...")

    # Training dataloader. Length = dataset size / batch size
    train_dataset = data.DATA(mode="train", train_status="finetune")
    dataloader_train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=argparser.n_cpu
    )
    print("---> length of training dataset: ", len(train_dataset))

    # Load test images
    test_dataset = data.DATA(mode="test", train_status="test")
    dataloader_test = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=argparser.n_cpu
    )
    print("---> length of test dataset: ", len(test_dataset))

    # -------
    # Test reconstruct
    # -------
    # for idx, (imgs_masked, masks, gts, info) in enumerate(dataloader_test):
    #     print("masked images shape: ", gts.shape)
    #     print("masked images shape: ", imgs_masked.shape)
    #     print("masked images shape: ", masks.shape)
    #     name = str.split(info['name'][0], '_')
    #     reconstruct = reconstruct_img.reconstruct(imgs_masked.squeeze(), int(info['Heigth']), int(info['Width']), name[0], args)
    #     reconstruct.save('test.jpg')
    #     te = np.asarray(reconstruct)
    #     print(te.shape)
    #     print(name[0])
    #
    #     gts = gts.squeeze()
    #     gts = gts.permute(1, 2, 0).numpy()
    #     gts = (gts * 255).astype('uint8')

    # -------
    # Model
    # -------

    # load model from fine-tune checkpoint if available
    if os.path.exists(model_fine_tune_pth):
        print("---> found previously saved {}, loading checkpoint and CONTINUE fine-tuning"
              .format(args.saved_fine_tune_name))
        load_model(model, model_fine_tune_pth)
    # load best pre-train model and start fine-tuning
    elif os.path.exists(model_pre_train_pth) and args.train_mode == "w_pretrain":
        print("---> found previously saved {}, loading checkpoint and START fine-tuning"
              .format(args.saved_pre_train_name))
        load_model(model, model_pre_train_pth)

    # freeze batch-norm params in fine-tuning
    if args.train_mode == "w_pretrain" and args.pretrain_epochs > 10:
        model.freeze()

    # ----------------
    #  Optimizer
    # ----------------

    # Optimizer
    print("---> preparing optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=argparser.LR_FT)
    criterion = nn.MSELoss()
    # Move model to device
    model.to(device)

    # ----------
    #  Training
    # ----------

    print("---> start training cycle ...")
    with open(os.path.join(args.output_dir, "finetune_losses.csv"), "w", newline="") as csv_losses:
        with open(os.path.join(args.output_dir, "finetune_scores.csv"), "w", newline="") as csv_scores:
            writer_losses = csv.writer(csv_losses)
            writer_losses.writerow(["Epoch", "Iteration", "Loss"])

            writer_scores = csv.writer(csv_scores)
            writer_scores.writerow(["Epoch", "Total Loss", "MSE", "SSIM", "Final Score"])

            iteration = 0
            highest_final_score = 0.0   # the higher the better, combines mse and ssim

            for epoch in range(args.finetune_epochs):

                model.train()

                loss_sum = 0    # store accumulated loss for one epoch

                for idx, (imgs_masked, masks, gts) in enumerate(dataloader_train):

                    # Move to device
                    imgs_masked = imgs_masked.to(device)    # (N, 3, H, W)
                    masks = masks.to(device)                # (N, 1, H, W)
                    gts = gts.to(device)                    # (N, 3, H, W)

                    #print("masked images shape: ",imgs_masked.shape)
                    #print("masks shape: ",masks.shape)
                    #print("target images shape: ",gts.shape)

                    # Model forward path => predicted images
                    preds = model(imgs_masked, masks)

                    original_pixels = torch.mul(masks, imgs_masked)
                    ones = torch.ones(masks.size()).cuda()
                    reversed_masks = torch.sub(ones, masks)
                    predicted_pixels = torch.mul(reversed_masks, preds)
                    preds = torch.add(original_pixels, predicted_pixels)

                    # Calculate total loss
                    #train_loss = loss.total_loss(preds, gts)
                    train_loss = criterion(preds, gts)
                    # Execute Back-Propagation
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f]" %
                          (epoch + 1, args.finetune_epochs, (idx + 1), len(dataloader_train), train_loss), end="")

                    loss_sum += train_loss.item()
                    writer_losses.writerow([epoch+1, iteration+1, train_loss.item()])
                    iteration += 1

                # ------------------
                #  Evaluate & Save Model
                # ------------------

                if (epoch+1) % args.val_epoch == 0:
                    mse, ssim = test.test(args, model, device, dataloader_test, mode="validate")
                    final_score = 1 - mse / 100 + ssim
                    print("\nMetrics on test set @ epoch {}:".format(epoch+1))
                    print("-> Average MSE:  {:.5f}".format(mse))
                    print("-> Average SSIM: {:.5f}".format(ssim))
                    print("-> Final Score:  {:.5f}".format(final_score))

                    if final_score > highest_final_score:
                        save_model(model, model_fine_tune_pth)
                        highest_final_score = final_score

                    writer_scores.writerow([epoch+1, loss_sum, mse, ssim, final_score])

                save_model(model, os.path.join(args.model_dir_fine_tune, "Net_finetune_epoch{}.pth.tar".format(epoch+1)))
                if epoch > 0:
                    remove_prev_model(os.path.join(args.model_dir_fine_tune, "Net_finetune_epoch{}.pth.tar".format(epoch)))

    print("\n***** Fine-tuning FINISHED *****")


if __name__ == "__main__":

    # -------
    #  General Setup
    # -------

    # Read input arguments
    args = argparser.arg_parse()

    # Directory for miscellaneous outputs
    if not os.path.exists(args.output_dir):
        print("Created directory for outputs: {}".format(args.output_dir))
        os.makedirs(args.output_dir)

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set seeds for batch shuffling
    random.seed(1)
    torch.manual_seed(1)

    # -------
    # Model
    # -------

    # initialize model
    print("\n---> initialize model...")
    model = model.Net()

    # print model information
    # print(model)
    # print(list(model.parameters()))

    # SET paths to best models
    model_pre_train_pth = os.path.join(args.model_dir_pre_train, args.saved_pre_train_name)
    model_fine_tune_pth = os.path.join(args.model_dir_fine_tune, args.saved_fine_tune_name)

    # -------
    #  Test Evaluate
    # ------

    # checkpoint = torch.load('Net_best_fine_tune.pth.tar', map_location='cpu')
    # model.load_state_dict(checkpoint)
    #
    # Load test images
    test_dataset = data.DATA(mode="test", train_status="TA")
    dataloader_test = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=argparser.n_cpu)

    train_dataset = data.DATA(mode="train", train_status="x")
    dataloader_train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=argparser.n_cpu)
    # #
    # # for idx, (imgs_masked, masks, gts, _) in enumerate(dataloader_test):
    # #     print(imgs_masked.size())
    # #     out = model(imgs_masked.squeeze())
    #
    data_iter = iter(dataloader_test)
    imgs_masked, masks, gts = data_iter.next()
    # unique, counts = np.unique(masks.numpy(), return_counts=True)
    print(masks.size())
    # print(dict(zip(unique, counts)))
    #


    # #
    # #
    data_iter2 = iter(dataloader_train)
    imgs_masked, masks, gts = data_iter2.next()
    print(masks.size())
    # unique, counts = np.unique(masks.numpy(), return_counts=True)
    # print(dict(zip(unique, counts)))

    # # print(imgs_masked.size())
    # # print(masks.size())
    # # print(gts.size())
    # # masks = masks.squeeze()
    # imgs_masked = imgs_masked.squeeze(0)
    # # print(imgs_masked[1].permute(1, 2, 0).numpy())
    # #
    # # i = Image.fromarray((masks[1].numpy() * 255).astype('uint8'))
    # p = Image.fromarray((imgs_masked[1].permute(1, 2, 0).numpy() * 255).astype('uint8')).convert("RGB")
    # # i.save('mask.png')
    # p.save('masked.jpg')
    #
    # mse, ssim = test.test(args, model, device, dataloader_test, mode="validate")
    #
    # print(mse)
    # print(ssim)
    # print('-------')

    # -------
    #  Pre-training
    # -------
    if args.train_mode == "w_pretrain":
        # Check if pre-train data directory exists
        if not os.path.exists(args.data_dir_pre_train):
            print("Could not find pre-train data directory: {}".format(args.data_dir_pre_train))
            sys.exit()

        # Check if directory for saved pre-train models exist
        if not os.path.exists(args.model_dir_pre_train):
            print("Created directory for pretrained models: {}".format(args.model_dir_pre_train))
            os.makedirs(args.model_dir_pre_train)

        ''' pre-train model on Place2 dataset '''
        pre_train(device, model, model_pre_train_pth)

    # -------
    #  Fine-tuning
    # -------

    # Check if fine-tune data directory exists
    if not os.path.exists(args.data_dir_fine_tune):
        print("Could not find fine-tune data directory: {}".format(args.data_dir_fine_tune))
        sys.exit()

    # Check if directory for saved pre-train models exist
    if not os.path.exists(args.model_dir_fine_tune):
        print("Created directory for fine tuned models: {}".format(args.model_dir_fine_tune))
        os.makedirs(args.model_dir_fine_tune)

    ''' fine-tune model on Dunhuang Grottoes dataset '''
    fine_tune(device, model, model_pre_train_pth, model_fine_tune_pth)
