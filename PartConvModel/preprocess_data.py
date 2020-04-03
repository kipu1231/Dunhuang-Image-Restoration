import random
import os
import numpy as np
from PIL import Image
import scipy.misc
import glob


def createRandomMask():
    ''' defining the number of steps'''
    n = 50000

    ''' creating two arrays '''
    x = np.zeros(n)
    y = np.zeros(n)

    ''' filling with random numbers '''
    for i in range(1, n):
        val = random.randint(1, 4)
        if val == 1:
            x[i] = x[i - 1] + 1
            y[i] = y[i - 1]
        elif val == 2:
            x[i] = x[i - 1] - 1
            y[i] = y[i - 1]
        elif val == 3:
            x[i] = x[i - 1]
            y[i] = y[i - 1] + 1
        else:
            x[i] = x[i - 1]
            y[i] = y[i - 1] - 1
    return x,y


def generateMask():
    x, y = createRandomMask()

    ''' Initialize random starting point P0'''
    p_0_x = random.randint(50, 205)
    p_0_y = random.randint(50, 205)

    img = np.ones((256, 256), dtype='uint8')
    image = np.ones((256, 256), dtype='uint8')

    img = Image.fromarray(img, 'P')

    ''' set P0 to 0'''
    img.putpixel((p_0_x, p_0_y), 0)
    image[p_0_x, p_0_y] = 0

    ''' rescale mask and put it on the image'''
    for i in range(len(x)):
        x_val = int(p_0_x + x[i])
        y_val = int(p_0_y + y[i])

        if 0 <= x_val <= 255 and 0 <= y_val <= 255:
            img.putpixel((x_val, y_val), 0)
            image[x_val, y_val] = 0



    ''' Plot mask '''
    image = img.point(lambda i: i * 255.0)
    #image.show()

    return image


if __name__ == "__main__":
    'define directories'
    img_dir = "./../Data_Challenge2/test/"
    out_dir_mask = "../out/mask"

    if not os.path.exists(out_dir_mask):
        os.makedirs(out_dir_mask)

    'Get paths to each of the images'
    img_names = sorted(os.listdir(img_dir))

    'generate masks'
    for img_number, name in enumerate(img_names):
        print(img_number)
        for i in range(25):
            mask = generateMask()
            mask.save(os.path.join(out_dir_mask, "{}_{}_mask.png".format(name[0:3], i + 1)))



