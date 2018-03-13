#encoding=utf-8
import cv2
import numpy as np
#定义了一个5×5的十字形结构元素,
#用结构元素与其覆盖的二值图像做“与”操作
#如果都为1，结果图像的该像素为1。否则为0
#腐蚀处理的结果是使原来的二值图像减小一圈。
#00100
#00100
#11111
#00100
#00100

kernel=np.uint8(np.zeros((2,2)))
for x in range(2):
    kernel[x,1]=1;
    kernel[1,x]=1;
#读入图片
img = cv2.imread('./gen_imgs/0.png',0)
# mask
# img = 255 - img
# mask = 0.5*np.ones((img.shape[0],img.shape[1]))
# mask = mask > np.random.rand(img.shape[0],img.shape[1])
# mask = np.asarray(mask, dtype=int)
# res = np.multiply(img, mask)
# res = 255- res
# cv2.imwrite("./res.jpg", res)

#腐蚀图像
eroded=cv2.erode(img,kernel);
#膨胀图像
dilated = cv2.dilate(img,kernel)



#将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
result = cv2.absdiff(dilated,eroded);
#取反
x=0;
y=0;
width=result.shape[0]
height=result.shape[1]
while x<width:
    y=0
    while y<height:
        result[x][y]=255-result[x][y]
        y=y+1;
    x=x+1
cv2.imwrite("./eroded.jpg", eroded)
cv2.imwrite("./dilated.jpg", dilated)
cv2.imwrite("./result.jpg", result)


cv2.waitKey(0)
cv2.destroyAllWindows()