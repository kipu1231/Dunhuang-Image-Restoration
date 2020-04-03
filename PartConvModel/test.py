import os
import torch
import data
import argparser
import model
import evaluate
import reconstruct_img
import numpy as np
from PIL import Image


def TAtest(args, model, device, dataloader_test, mode):
    model.eval()

    outputs_list = []
    img_name_list = []

    with torch.no_grad():
        for idx, (imgs_masked, masks, img_info) in enumerate(dataloader_test):
            imgs_masked = imgs_masked.to(device)
            masks = masks.to(device)

            imgs_masked = imgs_masked.squeeze(0)
            masks = masks.squeeze(0)

            outputs = model(imgs_masked, masks)

            if torch.cuda.is_available():
                imgs_masked = imgs_masked.cpu()  # (N, 3, H, W)
                masks = masks.cpu()  # (N, 1, H, W)
                outputs = outputs.cpu()

            original_pixels = torch.mul(masks, imgs_masked)
            reversed_masks = torch.sub(torch.ones(masks.size()), masks)
            predicted_pixels = torch.mul(reversed_masks, outputs)
            prediction = torch.add(original_pixels, predicted_pixels)

            #reconstruct image from patches
            name = str.split(img_info['name'][0], '_')
            reconstructed = reconstruct_img.reconstruct(prediction, int(img_info['Heigth']), int(img_info['Width']),
                                                      name[0], img_info['center_info'], args)

            ''' Reshape and transform gts and reconstructed image'''
            reconstructed.save('test.jpg')
            #reconstructed image = numpy array of shape (H, W. 3)
            img_reconstruct = np.asarray(reconstructed)


            ''' Append gts and reconstructed image to list'''
            outputs_list.append(img_reconstruct)
            img_name_list.append(name[0])

    if mode == "validate":
        return 'not needed'

    elif mode == "test":
        if not os.path.exists(args.predictions):
            os.makedirs(args.predictions)
        for idx in range(len(outputs_list)):
            img = Image.fromarray(outputs_list[idx])
            img_path = os.path.join(args.predictions, img_name_list[idx] + '.jpg')
            img.save(img_path)
        return 'Images saved'


def test(args, model, device, dataloader_test, mode):
    model.eval()

    gts_list = []
    outputs_list = []
    img_name_list = []

    with torch.no_grad():
        for idx, (imgs_masked, masks, gts, img_info) in enumerate(dataloader_test):
            # print(idx)
            imgs_masked = imgs_masked.to(device)
            masks = masks.to(device)
            gts = gts.to(device)

            imgs_masked = imgs_masked.squeeze(0)
            masks = masks.squeeze(0)

            outputs = model(imgs_masked, masks)

            if torch.cuda.is_available():
                imgs_masked = imgs_masked.cpu()  # (N, 3, H, W)
                masks = masks.cpu()  # (N, 1, H, W)
                outputs = outputs.cpu()
                gts = gts.cpu()

            original_pixels = torch.mul(masks, imgs_masked)
            reversed_masks = torch.sub(torch.ones(masks.size()), masks)
            predicted_pixels = torch.mul(reversed_masks, outputs)
            prediction = torch.add(original_pixels, predicted_pixels)

            #reconstruct image from patches
            name = str.split(img_info['name'][0], '_')
            reconstructed = reconstruct_img.reconstruct(prediction, int(img_info['Heigth']), int(img_info['Width']),
                                                      name[0], img_info['center_info'], args)

            ''' Reshape and transform gts and reconstructed image'''
            reconstructed.save('test.jpg')
            #reconstructed image = numpy array of shape (H, W. 3)
            img_reconstruct = np.asarray(reconstructed)

            #gts = numpy array of shape (H, W. 3)
            gts = gts.squeeze()
            gts = gts.permute(1, 2, 0).numpy()
            gts = (gts * 255).astype('uint8')

            ''' Append gts and reconstructed image to list'''
            gts_list.append(gts)
            outputs_list.append(img_reconstruct)
            img_name_list.append(name[0])

    if mode == "validate":
        mse, ssim = evaluate.get_average_mse_ssim_train(gts_list, outputs_list)
        return mse, ssim

    elif mode == "test":
        if not os.path.exists(args.predictions):
            os.makedirs(args.predictions)
        for idx in range(len(outputs_list)):
            img = Image.fromarray(outputs_list[idx])
            img_path = os.path.join(args.predictions, img_name_list[idx] + '.jpg')
            img.save(img_path)
        return 'Images saved'


if __name__ == "__main__":
    args = argparser.arg_parse()

    # Load test images
    test_dataset = data.DATA(mode="test", train_status="TA")
    dataloader_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.batch_size_test,
                                                  shuffle=False,
                                                  num_workers=argparser.n_cpu)

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.Net()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    # start the testing process
    _ = TAtest(args, model, device, dataloader_test, mode="test")

