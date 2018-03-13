import cv2
import numpy as np
import os
from PIL import Image



def padwithones(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 255
    vector[-pad_width[1]:] = 255
    return vector

path = './train-data-ID/X/'
img_list = os.listdir(path)
margin1 = 2
margin2 = 4
for imname in img_list:
    cnt = 0
    im = Image.open(os.path.join(path, imname))
    # # ### padding
    # im1 = np.lib.pad(np.array(im.convert('L')),  margin1, padwithones)
    # im1 = cv2.resize(im1, (40, 40))
    # cv2.imwrite('./tmp/aug_%s_%d.png' % (imname.split('.')[0], cnt), im1)
    # cnt += 1
    # im2 = np.lib.pad(np.array(im.convert('L')),  margin2, padwithones)
    # im2 = cv2.resize(im2, (40, 40))
    # cv2.imwrite('./tmp/aug_%s_%d.png' % (imname.split('.')[0], cnt), im2)
    # cnt += 1
    # translation
    for i in range(10):
        left = np.random.randint(low=0, high=6)
        upper = np.random.randint(low=0, high=6)
        right = np.random.randint(34,40)
        lower = np.random.randint(34, 40)
        new = im.crop((left, upper, right, lower))
        new = new.resize((40,40))
        new.save('./tmp/aug_%s_%d.png' % (imname.split('.')[0], cnt))
        cnt += 1
    #  ### rotate

    # new = im.rotate(-8, resample=Image.BICUBIC,expand=0.9)
    # new.save('./tmp/aug_%s_%d.png' % (imname.split('.')[0], cnt))
    # cnt += 1
    # new = im.rotate(8, resample=Image.NEAREST,expand=0.8)
    # new.save('./tmp/aug_%s_%d.png' % (imname.split('.')[0], cnt))
    # cnt += 1

    rows,cols = np.array(im.convert('L')).shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),3,1.2)
    img = cv2.warpAffine(np.array(im.convert('L')),M,(cols,rows))
    cv2.imwrite('./tmp/aug_%s_%d.png' % (imname.split('.')[0], cnt),img)
    cnt += 1

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-3,1.2)
    img = cv2.warpAffine(np.array(im.convert('L')),M,(cols,rows))
    cv2.imwrite('./tmp/aug_%s_%d.png' % (imname.split('.')[0], cnt),img)
    cnt += 1


