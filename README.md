# text_segment_deeplearning_based
# purpose  
This code is written by tengshaofeng in order to segment character in a text line image.  
# reference paper
Chinese/English mixed Character Segmentation as Semantic Segmentation. Huabin Zheng
# Requirements  
Anaconda2.x tensorflow 0.x 
# prepare dataset
you can generate the dataset from several types of fonts by run:
python dataset_gen.py
the gen_imgs folder store the images, and corpus.txt store the characters.
also you can manually label the 1024*48 text line image.
dataset.txt save the label like：
['0.png', 0, 36, 45, 82, 87, 126, 129, 167, 172, 209, 215, 252, 253, 290, 305, 341, 354, 392, 393, 433, 435, 473]
['1.png', 11, 47, 56, 94, 99, 135, 141, 177, 179, 218, 232, 268, 273, 311, 326, 365, 368, 408, 421, 461, 461, 5]
0 , 36, 45 and so on is the split points(cols) in the image.

# train  
python CharacterSegmentTrain.py

# experiment result
https://github.com/tengshaofeng/text_segment_deeplearning_based/tree/master/result/IFSK_CC@%S`SCTD2YY%PZ5B.jpg
https://github.com/tengshaofeng/text_segment_deeplearning_based/tree/master/result/P81LF(I0444O@(GRGXL19@3.png
https://github.com/tengshaofeng/text_segment_deeplearning_based/tree/master/result/微信图片_20170524142730.jpg
