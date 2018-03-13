#!/usr/bin/env python
#-*- coding:utf-8 -*-
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

result=[]
file_object = open( "./hanzi.txt", "r" )
all_the_text=file_object.read().decode('gbk')
print all_the_text
flag=[0]*(0x9FA5-0x4E00+1)
str_data=[]
for i in range(0,len(all_the_text)):
    temp=ord(all_the_text[i])
    if temp>=0x4E00 and temp<=0x9FA5:
        str_data.append(all_the_text[i])
        flag[temp-0x4E00]=1
        """
        if flag[temp-0x4E00]==0:
            str_data.append(all_the_text[i])
            flag[temp-0x4E00]=1
        """

#读取所用到的字体的文件名，获取用来输出的字符
str_Font='./Fonts'
list = GetFileList(str_Font, [])
print len(list),"种字体",':'
num_font=0;
for e in list:
    (filepath,tempfilename) = os.path.split(e);#提取字体文件路径、字体文件名
    (shotname,extension) = os.path.splitext(tempfilename);#提取字体文件名、及文件后缀
    if extension=='.ttf' or extension=='.ttc' or extension=='.TTF' or extension=='.otf':
        num_font+=1 #计算字体个数
print len(str_data),"个字"


Word_P=[]
#生成字体样本
j=0

#start,end = (0x4E00, 0x9FA5)#unicode编码中汉字编码范围
start,end = (0, len(str_data))
isExists=os.path.exists("./WordFontPicture")
if not isExists:
    os.makedirs("./WordFontPicture")

isExists=os.path.exists("./NumFontPicture")
if not isExists:
    os.makedirs("./NumFontPicture")
for codepoint in range(int(start),int(end)):
    #对每个字都新建一个文件夹
    isExists=os.path.exists("./WordFontPicture/"+str(j))
    if not isExists:
        os.makedirs("./WordFontPicture/"+str(j))

    #word=unichr(codepoint)
    word=str_data[j]
    Word_P.append(word)
    print codepoint,':',word
    i=0
    for k in range(0,Run_times):
        for e in list:
            (filepath,tempfilename) = os.path.split(e)
            (shotname,extension) = os.path.splitext(tempfilename)
            if extension=='.ttf' or extension=='.ttc'or extension=='.TTF' or extension=='.otf':# or extension=='.fon':
                im = Image.new("RGB", (Image_size, Image_size), (255, 255, 255))
                dr = ImageDraw.Draw(im)
                Word_size = np.random.randint(26, 28)
                font = ImageFont.truetype(os.path.join(str_Font, str(tempfilename)), Word_size, encoding="Unicode")
                x_rand=random.randint(Min_random_size,Max_random_size)
                y_rand=random.randint(Min_random_size,Max_random_size)
                dr.text((x_rand, y_rand), word, font=font, fill="#000000")
                im=np.array(im)
                if En_Noise:
                    im = cv2.GaussianBlur(im,(Gaussian_size,Gaussian_size),Gaussian_e)
                cv2.imwrite("./WordFontPicture/"+str(j)+"/"+str(i)+".png",im,[int(cv2.IMWRITE_JPEG_QUALITY),5])
                i+=1
    j+=1

#j=0
#生成英文字母和数字
start,end = (0x30, 0x7B)
#start,end = (0x30, 0x3A)
for codepoint in range(int(start),int(end)):
    #对每个字都新建一个文件夹
    if (codepoint>=0x30 and codepoint<0x3A) or codepoint==0x58:#(codepoint>=0x41 and codepoint<0x5B) or (codepoint>=0x61 and codepoint<0x7B):
    #if (codepoint>=0x30 and codepoint<0x3A) or (codepoint==0x58) :
        isExists=os.path.exists("./WordFontPicture/"+str(j))
        if not isExists:
            os.makedirs("./WordFontPicture/"+str(j))
        word=unichr(codepoint)
        Word_P.append(word)
        print j,':',word
        i=0
        for k in range(0,Run_times):
            for e in list:
                (filepath,tempfilename) = os.path.split(e)
                (shotname,extension) = os.path.splitext(tempfilename);        
                if extension=='.ttf' or extension=='.ttc'or extension=='.TTF' or extension=='.otf':# or extension=='.fon':
                    im = Image.new("RGB", (Image_size, Image_size), (255, 255, 255))
                    dr = ImageDraw.Draw(im)
                    Word_size = np.random.randint(26, 28)
                    font = ImageFont.truetype(os.path.join(str_Font, str(tempfilename)), Word_size, encoding="Unicode")
                    x_rand=random.randint(Min_random_size,Max_random_size)
                    y_rand=random.randint(Min_random_size,Max_random_size)
                    dr.text((x_rand+5, y_rand), word, font=font, fill="#000000")         
                    im=np.array(im)
                    if En_Noise:
                        im = cv2.GaussianBlur(im,(Gaussian_size,Gaussian_size),Gaussian_e)
                    cv2.imwrite("./WordFontPicture/"+str(j)+"/"+str(i)+".png",im,[int(cv2.IMWRITE_JPEG_QUALITY),5])
                    i+=1
        j+=1
# str_fuhao=[u'%.-:']
# for codepoint in range(0,len(str_fuhao[0])):
#     #对每个字都新建一个文件夹
#     isExists=os.path.exists("./WordFontPicture/"+str(j))
#     if not isExists:
#         os.makedirs("./WordFontPicture/"+str(j))
#     word=str_fuhao[0][codepoint]
#     Word_P.append(word)
#     print j,':',word
#     i=0
#     for k in range(0,Run_times):
#         for e in list:
#             (filepath,tempfilename) = os.path.split(e)
#             (shotname,extension) = os.path.splitext(tempfilename);
#             if extension=='.ttf' or extension=='.ttc'or extension=='.TTF' or extension=='.otf':# or extension=='.fon':
#                 im = Image.new("RGB", (Image_size, Image_size), (255, 255, 255))
#                 dr = ImageDraw.Draw(im)
#                 Word_size = np.random.randint(19, 34)
#                 font = ImageFont.truetype(os.path.join(str_Font, str(tempfilename)), Word_size, encoding="Unicode")
#                 x_rand=random.randint(Min_random_size,Max_random_size)
#                 y_rand=random.randint(Min_random_size,Max_random_size)
#                 dr.text((x_rand+5, y_rand), word, font=font, fill="#000000")
#                 im=np.array(im)
#                 if En_Noise:
#                     im = cv2.GaussianBlur(im,(Gaussian_size,Gaussian_size),Gaussian_e)
#                 cv2.imwrite("./WordFontPicture/"+str(j)+"/"+str(i)+".png",im,[int(cv2.IMWRITE_JPEG_QUALITY),5])
#                 i+=1
#     j+=1
f1 = file('word_dict_3766.pickle', 'wb')
pickle.dump(Word_P, f1, True)
f1.close()
"""
start,end = (0x21, 0x41)
for codepoint in range(int(start),int(end)):
    #对每个字都新建一个文件夹
    if (codepoint>=0x21 and codepoint<0x30) or (codepoint>=0x3A and codepoint<0x41):
        isExists=os.path.exists("./FontPicture/"+str(j))
        if not isExists:
            os.makedirs("./FontPicture/"+str(j))
        word=unichr(codepoint)
        print j,':',word
        i=0
        for k in range(0,Run_times):
            for e in list:
                (filepath,tempfilename) = os.path.split(e)
                (shotname,extension) = os.path.splitext(tempfilename);        
                if extension=='.ttf' or extension=='.ttc'or extension=='.TTF' or extension=='.otf':# or extension=='.fon':
                    im = Image.new("RGB", (Image_size, Image_size), (255, 255, 255))
                    dr = ImageDraw.Draw(im)
                    font = ImageFont.truetype(os.path.join(str_Font, str(tempfilename)), Word_size, encoding="Unicode")
                    x_rand=random.randint(Min_random_size,Max_random_size)
                    y_rand=random.randint(Min_random_size,Max_random_size)
                    dr.text((x_rand, y_rand), word, font=font, fill="#000000")         
                    im=np.array(im)
                    if En_Noise:
                        im = cv2.GaussianBlur(im,(Gaussian_size,Gaussian_size),Gaussian_e)
                    cv2.imwrite("./FontPicture/"+str(j)+"/"+str(i)+".png",im,[int(cv2.IMWRITE_JPEG_QUALITY),5])
                    i+=1   
        j+=1
"""
