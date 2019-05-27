# text_segment_deeplearning_based
# purpose  
This code is written by tengshaofeng in order to segment character in a text line image.  

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

![result1](https://github.com/tengshaofeng/text_segment_deeplearning_based/blob/master/result/IFSK_CC%40%25S%60SCTD2YY%25PZ5B.jpg)
![result](https://github.com/tengshaofeng/text_segment_deeplearning_based/blob/master/result/微信图片_20170524142730.jpg)
