import cv2
import numpy as np
import math
import copy
import os

def show(image, name,x,y):
    cv2.namedWindow(name + '_' + "img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name + '_' + "img", x, y)
    cv2.imshow(name + '_' + "img", image)

def split_pic(image, contours,save_path, image_name):
    for i in range(len(contours)):
        print(i)
        # result_1 = np.zeros((w, h, 3), dtype='uint8')
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)  # 计算点集最外面的矩形边界
        result_1 = image[y:y+h, x:x+w]
        show(result_1, 'xx', 800, 600)
        cv2.imwrite(save_path + '/' + image_name + '_' + str(i) + '.jpg', result_1)

src = cv2.imread("result.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
print(gray.shape)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x = 0
maxarea = 1000
for i in range(len(contours)):
    print(i)
    area = cv2.contourArea(contours[i])
    if area > maxarea:
        maxarea = area
        print('轮廓面积', area)
        x = i
cnt = contours[x]
x, y, w, h = cv2.boundingRect(cnt)  # 计算点集最外面的矩形边界
print('外接矩形中心点：', x + w / 2, y + h / 2)
# 画出边界
cv2.rectangle(src, (x,y), (x+w,y+h), (0,255,0), 10)
print(src.shape)
split_pic(src,x, y, w, h )
show(src, 'src', 800, 600)

cv2.waitKey(0)
cv2.destroyAllWindows()
