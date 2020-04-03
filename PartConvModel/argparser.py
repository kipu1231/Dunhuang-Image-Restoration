import argparse

# -----------------
# Input Arguments
# -----------------


def arg_parse():
    parser = argparse.ArgumentParser()

    # Miscellaneous
    parser.add_argument("--pretrain_epochs", type=int, default=15)         # number of epochs to train
    parser.add_argument("--finetune_epochs", type=int, default=10)         # number of epochs to train
    parser.add_argument("--val_epoch", type=int, default=1)         # on which epoch to evaluate and save model
    parser.add_argument("--batch_size", type=int, default=32)       # size of batch for training
    parser.add_argument("--batch_size_test", type=int, default=1)   # size of batch for testing
    parser.add_argument("--gpu", type=int, default=0)               # GPU number

    # Directories
    parser.add_argument("--data_dir_pre_train", type=str, default="../Data_Place2/")      # Path to Data_Challenge2
    parser.add_argument("--data_dir_fine_tune", type=str, default="../Data_Challenge_new/")      # Path to Data_Challenge2
    parser.add_argument("--data_dir_test", type=str, default="../Data_Challenge2/")  # Path to Data_Challenge2
    parser.add_argument("--model_dir_pre_train", type=str, default="../pre_train_models/")  # Directory to save/load models
    parser.add_argument("--model_dir_fine_tune", type=str, default="../fine_tune_models/")  # Directory to save/load models
    parser.add_argument("--saved_pre_train_name", type=str, default="Net_best_pre_train.pth.tar")
    parser.add_argument("--saved_fine_tune_name", type=str, default="Net_best_fine_tune.pth.tar")
    parser.add_argument("--output_dir", type=str, default="../outputs")             # Directory to save output figures and stats
    parser.add_argument("--out_dir_info", type=str, default="../output/test")       # Directory to save test image splits
    parser.add_argument("--predictions", type=str, default="../predictions")
    parser.add_argument("--resume", type=str, default="../Net_best_fine_tune.pth-6.tar")

    # other parameters
    parser.add_argument("--size", type=int, default=256)  # Image size
    parser.add_argument("--train_mode", type=str, default="w_pretrain")  # if pretrain or not

    args = parser.parse_args()
    return args


# ------------------
# Other Parameters
# ------------------

# Data pre-processing
N = 25                                                      # number of patches
size = 256                                                  # dimension of patches
thresh = 70                                                 # distance threshold between patch centers
thickness = 2                                               # thickness of borders for patch visualization
img_dir = "../Data_Challenge2/test_gt/"                     # directory of ground truth images to extract patches from
out_dir_imgs = "../output/imgs/"                            # directory to save generated patches
out_dir_viz = "../output/viz/"                              # directory to save patch visualizations
out_dir_info = "../output/info/"                            # directory to save patch info (patch centers, etc)
out_dir_masks = "../output/masks/"                          # directory to save generated masks
out_dir_masked_imgs = "../output/masked_imgs"               # directory to save generated masked images
out_dir_reconstructed_imgs = "../output/reconstructed_imgs" # directory to save reconstructed test images
mode = "test"                                               # generate patches for train or test images

# Data loading
mean = [0.485, 0.456, 0.406]      # normalization mean
std = [0.229, 0.224, 0.22]       # normalization standard dev

# Training
LR = 2e-4                   # learning rate pretraining
LR_FT = 5e-5
n_cpu = 4                   # number of cpu threads during batch generation
