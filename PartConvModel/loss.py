import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable, Function

import argparser
import data
import model

# All pretrained torchvision models have the same preprocessing, 
# which is to normalize using the following mean/std values: 
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Balancing coefficients for individual losses
CONTENT = 0.05
STYLE = 1000
TV = 0.1

# tutorial on content & style loss
# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#loss-functions

def gram_matrix(input): # used for style loss calculation
    a, b, c, d = input.size()  
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def content_loss(img_vector, target_vector):
    return F.mse_loss(img_vector,target_vector)

def style_loss(img_vector, target_vector):
    i_gram = gram_matrix(img_vector)
    t_gram = gram_matrix(target_vector)
    return F.mse_loss(i_gram, t_gram) 

def total_variational_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h =  (x.size()[2]-1) * x.size()[3]
    count_w = x.size()[2] * (x.size()[3] - 1)
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size

def total_loss(img,target):
    # create loss network (for calculation of content and style loss)
    feature_extractor = models.vgg16(pretrained=True)
    loss_network = nn.Sequential(
        *list(feature_extractor.children())[0][:12],
    )
    for params in loss_network.parameters():
        params.requires_grad= False
    loss_network.eval()
    if torch.cuda.is_available():
        loss_network.cuda()
    # print("Loss Network: \n",loss_network)

    # preprocess images:
    transform = transforms.Normalize(MEAN,STD)

    # transform the output images and target images to feed into loss network
    tf_img = []
    tf_target = []
    for i in range(len(img)):
        tf_img.append(transform(img[i]))
        tf_target.append(transform(target[i]))
    tf_img = torch.stack(tf_img)
    tf_target = torch.stack(tf_target)

    # feed transformed image into loss network
    img_vector = loss_network(tf_img)
    target_vector = loss_network(tf_target)

    # calculate individual losses
    contentLoss = content_loss(img_vector, target_vector)
    styleLoss = style_loss(img_vector, target_vector)
    tvLoss = total_variational_loss(img)

    # get linear combination of individual losses
    total_loss = CONTENT*contentLoss + STYLE*styleLoss + TV*tvLoss
    
    return total_loss



