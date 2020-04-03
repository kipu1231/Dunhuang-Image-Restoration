###############################################################################
# U-like end-to-end network with skip connections and partial convolutions
###############################################################################

import torch
import torch.nn as nn
import copy

from partialconv2d import PartialConv2d
import data


class Net(nn.Module):

    # initialize and create weights
    def __init__(self):
        super(Net, self).__init__()

        # TODO: restructure into model layers => print model will look nicer
        # Multi_Channel: concatenate masks for every channel: R, G, B
        # Upsampling check: upsample masks?

        # ----------
        #  Encoder components
        # ----------

        # partial convolutions: increase #channels
        self.PartialConv2d_3_64 = PartialConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False,
                                                return_mask=True, multi_channel=True)
        # Height/Width: 256 -> 128
        self.PartialConv2d_64_128 = PartialConv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False,
                                                  return_mask=True, multi_channel=True)
        # Height/Width: 128 -> 64
        self.PartialConv2d_128_256 = PartialConv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False,
                                                   return_mask=True, multi_channel=True)
        # Height/Width:  64-> 32
        self.PartialConv2d_256_512 = PartialConv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False,
                                                   return_mask=True, multi_channel=True)
        # Height/Width: 32 -> 16
        self.PartialConv2d_512_512_5 = PartialConv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False,
                                                     return_mask=True, multi_channel=True)
        # Height/Width: 16 -> 8
        self.PartialConv2d_512_512_6 = PartialConv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False,
                                                     return_mask=True, multi_channel=True)
        # Height/Width: 8 -> 4
        self.PartialConv2d_512_512_7 = PartialConv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False,
                                                     return_mask=True, multi_channel=True)
        # Height/Width: 4 -> 2
        self.PartialConv2d_512_512_8 = PartialConv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False,
                                                     return_mask=True, multi_channel=True)
        # Height/Width: 2 -> 1

        # batch normalization
        self.BatchNorm2d_64_e = nn.BatchNorm2d(64)
        self.BatchNorm2d_128_e = nn.BatchNorm2d(128)
        self.BatchNorm2d_256_e = nn.BatchNorm2d(256)
        self.BatchNorm2d_512_e_4 = nn.BatchNorm2d(512)
        self.BatchNorm2d_512_e_5 = nn.BatchNorm2d(512)
        self.BatchNorm2d_512_e_6 = nn.BatchNorm2d(512)
        self.BatchNorm2d_512_e_7 = nn.BatchNorm2d(512)
        self.BatchNorm2d_512_e_8 = nn.BatchNorm2d(512)

        # non-linearity
        self.ReLU = nn.ReLU()

        # ----------
        # Decoder components
        # ----------

        # batch normalization
        self.BatchNorm2d_64_d = nn.BatchNorm2d(64)
        self.BatchNorm2d_128_d = nn.BatchNorm2d(128)
        self.BatchNorm2d_256_d = nn.BatchNorm2d(256)
        self.BatchNorm2d_512_d_9 = nn.BatchNorm2d(512)
        self.BatchNorm2d_512_d_10 = nn.BatchNorm2d(512)
        self.BatchNorm2d_512_d_11 = nn.BatchNorm2d(512)
        self.BatchNorm2d_512_d_12 = nn.BatchNorm2d(512)

        # upsampling
        self.UpsamplingNearest2d_double_size = nn.UpsamplingNearest2d(size=None, scale_factor=2)

        # partial convolutions: reduce #channels
        self.PartialConv2d_1024_512_9 = PartialConv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False,
                                                      return_mask=True, multi_channel=True)
        self.PartialConv2d_1024_512_10 = PartialConv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False,
                                                       return_mask=True, multi_channel=True)
        self.PartialConv2d_1024_512_11 = PartialConv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False,
                                                       return_mask=True, multi_channel=True)
        self.PartialConv2d_1024_512_12 = PartialConv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False,
                                                       return_mask=True, multi_channel=True)
        self.PartialConv2d_768_256_13 = PartialConv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False,
                                                      return_mask=True, multi_channel=True)
        self.PartialConv2d_384_128_14 = PartialConv2d(384, 128, kernel_size=3, stride=1, padding=1, bias=False,
                                                      return_mask=True, multi_channel=True)
        self.PartialConv2d_192_64_15 = PartialConv2d(192, 64, kernel_size=3, stride=1, padding=1, bias=False,
                                                     return_mask=True, multi_channel=True)
        self.PartialConv2d_67_3_16 = PartialConv2d(67, 3, kernel_size=3, stride=1, padding=1, bias=False,
                                                   return_mask=True, multi_channel=True)

        # non-linearity
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        # weight initialization
        # SOURCE: https://github.com/NVIDIA/partialconv/blob/master/models/pd_resnet.py
        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # network’s forward pass
    def forward(self, x, mask):

        # ----------
        #  Encoder layers
        # ----------

        # (256 x 256 x 3)

        mask0 = mask
        y0 = x.clone()

        # Layer 1 (no ReLU)
        x, mask = self.PartialConv2d_3_64(input=x, mask_in=mask)
        x = self.BatchNorm2d_64_e(x)
        mask1 = mask
        y1 = x.clone()

        # => (128 x 128 x 64)

        # Layer 2
        x, mask = self.PartialConv2d_64_128(input=x, mask_in=mask)
        x = self.BatchNorm2d_128_e(x)
        x = self.ReLU(x)
        y2 = x.clone()
        mask2 = mask

        # => (64 x 64 x 128)

        # Layer 3
        x, mask = self.PartialConv2d_128_256(input=x, mask_in=mask)
        x = self.BatchNorm2d_256_e(x)
        x = self.ReLU(x)
        y3 = x.clone()
        mask3 = mask

        # => (32 x 32 x 256)

        # Layer 4
        x, mask = self.PartialConv2d_256_512(input=x, mask_in=mask)
        x = self.BatchNorm2d_512_e_4(x)
        x = self.ReLU(x)
        y4 = x.clone()
        mask4 = mask

        # => (16 x 16 x 512)

        # Layer 5
        x, mask = self.PartialConv2d_512_512_5(input=x, mask_in=mask)
        x = self.BatchNorm2d_512_e_5(x)
        x = self.ReLU(x)
        y5 = x.clone()
        mask5 = mask

        # => (8 x 8 x 512)

        # Layer 6
        x, mask = self.PartialConv2d_512_512_6(input=x, mask_in=mask)
        x = self.BatchNorm2d_512_e_6(x)
        x = self.ReLU(x)
        y6 = x.clone()
        mask6 = mask

        # => (4 x 4 x 512)

        # Layer 7
        x, mask = self.PartialConv2d_512_512_7(input=x, mask_in=mask)
        x = self.BatchNorm2d_512_e_7(x)
        x = self.ReLU(x)
        y7 = x.clone()
        mask7 = mask

        # => (2 x 2 x 512)

        # Layer 8
        x, mask = self.PartialConv2d_512_512_8(input=x, mask_in=mask)
        # print(mask)
        # print(mask.shape)
        # batch norm doesnt work since each channel only has 1 value
        x = self.BatchNorm2d_512_e_8(x)
        x = self.ReLU(x)

        # => (1 x 1 x 512)
        # mask should consist only of 1s at this stage

        # ----------
        # Decoder layers
        # TODO: Should one return masks and concatenate them to encoder masks of skip connections?
        # TODO: 512 (NVIDIA) vs 1024 (Grotto paper) #channels in layer 9 to 12
        # ----------

        # (1 x 1 x 512)
        # Layer 9:
        x = self.UpsamplingNearest2d_double_size(x)
        mask = self.UpsamplingNearest2d_double_size(mask)

        # (2 x 2 x 512)
        x = torch.cat([x, y7], 1)
        mask = torch.cat([mask, mask7], 1)
        # (2 x 2 x 1024)
        x, mask = self.PartialConv2d_1024_512_9(input=x, mask_in=mask)
        x = self.BatchNorm2d_512_d_9(x)
        x = self.leakyReLU(x)

        # => (2 x 2 x 512)

        # Layer 10:
        x = self.UpsamplingNearest2d_double_size(x)
        mask = self.UpsamplingNearest2d_double_size(mask)
        # (4 x 4 x 512)
        x = torch.cat([x, y6], 1)
        mask = torch.cat([mask, mask6], 1)
        # (4 x 4 x 1024)
        x, mask = self.PartialConv2d_1024_512_10(input=x, mask_in=mask)
        x = self.BatchNorm2d_512_d_10(x)
        x = self.leakyReLU(x)

        # => (4 x 4 x 512)

        # Layer 11:
        x = self.UpsamplingNearest2d_double_size(x)
        mask = self.UpsamplingNearest2d_double_size(mask)
        # (8 x 8 x 512)
        x = torch.cat([x, y5], 1)
        mask = torch.cat([mask, mask5], 1)
        # (8 x 8 x 1024)
        x, mask = self.PartialConv2d_1024_512_11(input=x, mask_in=mask)
        x = self.BatchNorm2d_512_d_11(x)
        x = self.leakyReLU(x)

        # => (8 x 8 x 512)

        # Layer 12:
        x = self.UpsamplingNearest2d_double_size(x)
        mask = self.UpsamplingNearest2d_double_size(mask)
        # (16 x 16 x 512)
        x = torch.cat([x, y4], 1)
        mask = torch.cat([mask, mask4], 1)
        # (16 x 16 x 1024)
        x, mask = self.PartialConv2d_1024_512_12(input=x, mask_in=mask)
        x = self.BatchNorm2d_512_d_12(x)
        x = self.leakyReLU(x)

        # => (16 x 16 x 512)

        # Layer 13:
        x = self.UpsamplingNearest2d_double_size(x)
        mask = self.UpsamplingNearest2d_double_size(mask)
        # (32 x 32 x 512)
        x = torch.cat([x, y3], 1)
        mask = torch.cat([mask, mask3], 1)
        # (32 x 32 x 256+512)
        
        x, mask = self.PartialConv2d_768_256_13(input=x, mask_in=mask)
        x = self.BatchNorm2d_256_d(x)
        x = self.leakyReLU(x)

        # => (32 x 32 x 256)

        # Layer 14:
        x = self.UpsamplingNearest2d_double_size(x)
        mask = self.UpsamplingNearest2d_double_size(mask)
        # (64 x 64 x 256)
        x = torch.cat([x, y2], 1)
        mask = torch.cat([mask, mask2], 1)
        # (64 x 64 x 256+128)
        x, mask = self.PartialConv2d_384_128_14(input=x, mask_in=mask)
        x = self.BatchNorm2d_128_d(x)
        x = self.leakyReLU(x)

        # => (64 x 64 x 128)

        # Layer 15:
        x = self.UpsamplingNearest2d_double_size(x)
        mask = self.UpsamplingNearest2d_double_size(mask)
        # (128 x 128 x 128)
        x = torch.cat([x, y1], 1)
        mask = torch.cat([mask, mask1], 1)
        # (128 x 128 x 128+64)
        x, mask = self.PartialConv2d_192_64_15(input=x, mask_in=mask)
        x = self.BatchNorm2d_64_d(x)
        x = self.leakyReLU(x)

        # => (128 x 128 x 64)

        # Layer 16:
        x = self.UpsamplingNearest2d_double_size(x)
        mask = self.UpsamplingNearest2d_double_size(mask)
        # (256 x 256 x 64)
        x = torch.cat([x, y0], 1)
        mask = torch.cat([mask, mask0], 1)
        # => (256 x 256 x 64+3)
        x, mask = self.PartialConv2d_67_3_16(input=x, mask_in=mask)
        # => (256 x 256 x 3)

        return x

    # freeze batchnorm parameters
    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if self.freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

