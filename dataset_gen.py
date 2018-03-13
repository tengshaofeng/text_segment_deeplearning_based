# -*- coding:utf-8 -*-
import pickle
import numpy as np
f = open('word_dict_3825.pickle', 'r')
word_dict_3825 = pickle.load(f)
f.close()

# #####-----generate hanzi dataset ----------------------
# hanzi = []
# for i in word_dict_3825:
#     hanzi.append(i.rstrip())
# f = open('corpus.txt', 'w')
# for j in hanzi:
#     f.write(j.encode('gbk'))
#     f.write('\n')

# ########## generate the text line images ------------------

#!/usr/bin/env python

import os
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import random
#import importlib
import sys
import pickle

"""""""""""""""""""""""""""""""""""""""
            可设置参数
"""""""""""""""""""""""""""""""""""""""
Image_size=40  #输出每个字图片尺寸
# Word_size=28   #每幅图中字体尺寸

En_Noise=1     #1：噪声使能;0：噪声关闭
Gaussian_size=1 #高斯滤波窗口大小
Gaussian_e=2    #高斯滤波标准差

Min_random_size=5 #字体随机位置下限值
Max_random_size=8 #字体随机位置上限值

Run_times=5  #运行次数，一个字的样本个数=运行次数*字体数量
"""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""


def rotate(im_cv, angle=3, label=None):
    rows, cols = np.array(im_cv).shape[0:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    img = cv2.warpAffine(im_cv, M, (cols, rows), borderValue=[255, 255, 255])
    # get rotated split point
    c = np.c_[np.reshape(label, (len(label), 1)), np.zeros((len(label), 1))]
    landmark = np.asarray([(M[0][0] * x + M[0][1] * y + M[0][2],
                             M[1][0] * x + M[1][1] * y + M[1][2]) for (x, y) in c])
    return img, landmark[:, 0]


def random_mask(im_cv):
    im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)
    img = 255 - im_cv
    mask = 0.5 * np.ones((img.shape[0], img.shape[1]))
    mask = mask > np.random.rand(img.shape[0], img.shape[1])
    mask = np.asarray(mask, dtype=int)
    res = np.multiply(img, mask)
    res = 255 - res
    # cv2.imwrite("./res.jpg", res)
    return res


def erode(img_cv):
    arr = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel=np.uint8(np.zeros((2,2)))
    for x in range(2):
        kernel[x,1]=1;
        kernel[1,x]=1;

    #腐蚀图像
    eroded=cv2.erode(arr,kernel);

    return eroded


def dilate(img_cv):
    arr = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel=np.uint8(np.zeros((2,2)))
    for x in range(2):
        kernel[x,1]=1;
        kernel[1,x]=1;

    #膨胀图像
    dilated = cv2.dilate(arr,kernel)
    return dilated


#GetFileList遍历所有的字体文件，输出文件路径及文件名
def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            #if s == "xxx":
                #continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList


def Synthesizing_from_seg_data(data_dir=u'./result_word-segmented-from-IDcards-without-expandto500samples/'):
    image_names = []
    dataset = []
    label = []  # [1 2 00 500]
    subfolds = os.listdir(data_dir)
    for subfold in subfolds:
        filenames = os.listdir(os.path.join(data_dir, subfold))
        image_names += [os.path.join(data_dir, subfold, fname) for fname in filenames]

    im = Image.new("RGB", (1024, 48), (255, 255, 255))
    pos_x = 0
    cnt_img = 0
    Times = 2
    for tt in range(Times):
        random.shuffle(image_names)
        for fname in image_names:
            print fname
            offset_x = random.randint(0, 20)
            pos_x = pos_x + offset_x  # 字和字之间采用不定间隔
            im_a = Image.open(fname)
            im_a = im_a.crop((7, 7, 33, 33)) # ((left, upper, right, lower))
            Word_size = np.random.randint(36, 41)
            im_a = im_a.resize((Word_size, Word_size))
            if pos_x + Word_size >= 1024:  # save the text line image
                im.save("./gen_imgs/" + str(cnt_img) + ".png")
                print 'save:', str(cnt_img)+'.png'
                dataset.append([str(cnt_img) + '.png'] + label)
                cnt_img += 1

                # re initial
                im = Image.new("RGB", (1024, 48), (255, 255, 255))
                pos_x = 0
                offset_x = random.randint(0, 20)
                pos_x = pos_x + offset_x  # 字和字之间采用不定间隔
                label = []

            im.paste(im_a, (pos_x, (48 - Word_size) / 2))
            label += [pos_x, pos_x + Word_size]
            pos_x += Word_size

    return dataset, cnt_img

def Synthesizing_from_gen_data(dataset, cnt_img):
    result=[]
    file_object = open("./corpus.txt", "r" )
    all_the_text=file_object.read().decode('gbk')
    print all_the_text
    str_data=[]
    for i in range(0,len(all_the_text)):
        temp=ord(all_the_text[i])
        if temp>=0x4E00 and temp<=0x9FA5:
            str_data.append(all_the_text[i])

    #生成英文字母和数字
    start,end = (0x30, 0x7B)
    for codepoint in range(int(start),int(end)):
        #对每个字都新建一个文件夹 codepoint==0x58:X
        if (codepoint>=0x30 and codepoint<0x3A) or (codepoint>=0x41 and codepoint<0x5B) or (codepoint>=0x61 and codepoint<0x7B):
            word=unichr(codepoint)
            str_data.append(word)
    # 生成符号样本
    str_fuhao=[u'.,:。-']
    for codepoint in range(0,len(str_fuhao[0])):
        word=str_fuhao[0][codepoint]
        str_data.append(word)


    print len(str_data),"个字"


    #读取所用到的字体的文件名，获取用来输出的字符
    str_Font='./get-images/Fonts'
    list = GetFileList(str_Font, [])
    print len(list),"种字体",':'

    # Word_P=[]
    # cnt_img = 0
    # dataset = []  # [0.png 1 200 500]
    label = []  # [1 2 00 500]
    start,end = (0, len(str_data))
    for e in list:  # 11 kinds of fonts
        # draw text on the blank image wiht size of 1024*48
        (filepath, tempfilename) = os.path.split(e)
        (shotname, extension) = os.path.splitext(tempfilename)

        im = Image.new("RGB", (1024, 48), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        pos_x = 0
        for codepoint in range(int(start),int(end)):
            #word=unichr(codepoint)
            word=str_data[codepoint]
            # Word_P.append(word)
            print codepoint,':',word

            Word_size = np.random.randint(26, 40)
            font = ImageFont.truetype(os.path.join(str_Font, str(tempfilename)), Word_size, encoding="Unicode")
            # x_rand=random.randint(Min_random_size,Max_random_size)
            # y_rand=random.randint(Min_random_size,Max_random_size)
            offset_x = random.randint(0, 20)
            pos_x = pos_x + offset_x  # 字和字之间采用不定间隔

            if pos_x + Word_size >= 1024:  # save the text line image
                im = np.array(im)
                if En_Noise:
                    im = cv2.GaussianBlur(im, (Gaussian_size, Gaussian_size), Gaussian_e)
                cv2.imwrite("./gen_imgs/" + str(cnt_img) + ".png", im, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
                dataset.append([str(cnt_img) + '.png'] + label)
                cnt_img += 1

                # erode
                e_im = erode(im)
                cv2.imwrite("./gen_imgs/" + str(cnt_img) + ".png", e_im, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
                dataset.append([str(cnt_img) + '.png'] + label)
                cnt_img += 1

                # dilate
                d_im = dilate(im)
                cv2.imwrite("./gen_imgs/" + str(cnt_img) + ".png", d_im, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
                dataset.append([str(cnt_img) + '.png'] + label)
                cnt_img += 1

                # mask
                m_im = random_mask(im)
                cv2.imwrite("./gen_imgs/" + str(cnt_img) + ".png", m_im, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
                dataset.append([str(cnt_img) + '.png'] + label)
                cnt_img += 1

                # rotate
                r_im, label_rot = rotate(im, -1, label)
                label_rot = [int(round(t)) for t in label_rot]
                cv2.imwrite("./gen_imgs/" + str(cnt_img) + ".png", r_im, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
                dataset.append([str(cnt_img) + '.png'] + label_rot)
                cnt_img += 1

                r_im, label_rot = rotate(im, 1, label)
                label_rot = [int(round(t)) for t in label_rot]
                cv2.imwrite("./gen_imgs/" + str(cnt_img) + ".png", r_im, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
                dataset.append([str(cnt_img) + '.png'] + label_rot)
                cnt_img += 1


                # re initial
                im = Image.new("RGB", (1024, 48), (255, 255, 255))
                dr = ImageDraw.Draw(im)
                pos_x = 0
                offset_x = random.randint(0, 20)
                pos_x = pos_x + offset_x  # 字和字之间采用不定间隔
                label = []
            dr.text((pos_x, (48-Word_size)/2), word, font=font, fill="#000000")
            label += [pos_x, pos_x+Word_size]
            pos_x += Word_size
    return dataset
    # np.savetxt('dataset.txt', dataset, fmt='%s')

###  main####
#data, cnt_img = Synthesizing_from_seg_data()

dataset = Synthesizing_from_gen_data(data, cnt_img)
np.savetxt('dataset.txt', dataset, fmt='%s')