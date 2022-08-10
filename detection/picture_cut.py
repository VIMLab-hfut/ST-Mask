import cv2
import numpy as np
import os
import math
import shutil
from PIL import Image

#image = 'D:/test_pictures/13.jpg'  # 分割的图片的位置
pic_target = 'E:/TorchVision_Maskrcnn/Maskrcnn/PennFudanPed/xz_0227_1326_png_split/'  # 分割后的图片保存的文件夹
mask_target = 'E:/TorchVision_Maskrcnn/Maskrcnn/PennFudanPed/xz_0227_1326_mask_split/' #分割后mask图片保存的文件夹
mask_dir_input = 'E:/TorchVision_Maskrcnn/Maskrcnn/PennFudanPed/xz_mask_0227_1326'#mask图片原图
file_dir_input = 'E:/TorchVision_Maskrcnn/Maskrcnn/PennFudanPed/xz_pic_0227_1326'#原图未切割前
'''
# [
[mask_name 0  range_y[i]:range_y[i]+patch_size[0], range_x[j]:range_x[j] + patch_size[1]:]
 mask_name 0  range_y[i]:range_y[i]+patch_size[0], range_x[j]:range_x[j] + patch_size[1]:
]
'''

# ]
# # 要分割后的尺寸
# cut_width = 1024
# cut_length = 1024
# # 读取要分割的图片，以及其尺寸等数据
# picture = cv2.imread(pic_path)
# (width, length, depth) = picture.shape
# # 预处理生成0矩阵
# pic = np.zeros((cut_width, cut_length, depth))
# # 计算可以划分的横纵的个数
# num_width = int(width / cut_width)
# print(num_width)
# num_length = int(length / cut_length)
# print(num_length)
# # for循环迭代生成
# for i in range(0, num_width):
#     for j in range(0, num_length):
#         pic = picture[i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, :]
#         result_path = pic_target + '{}_{}.jpg'.format(i + 1, j + 1)
#         cv2.imwrite(result_path, pic)
#
# print("done!!!")
# image = cv2.imread("D:/test_pictures/0.jpg")
patch_size = (1024,1024)
stride = 1000
#print(1)
def cut_image(image,image2,name,name2,target,target2):
    if len(image.shape) == 2:  # 灰度图像
        imhigh, imwidth = image.shape
    if len(image.shape) == 3:  # RGB图像
        imhigh, imwidth, imch = image.shape
    range_y = np.arange(0, imhigh - patch_size[0], stride)  # 使用给定间隔内的均匀间隔的值来创建数组
    range_x = np.arange(0, imwidth - patch_size[1], stride)
    # print(range_y)
    if range_y[-1] != imhigh - patch_size[0]:  # 无法整除滑动窗口大小
        range_y = np.append(range_y, imhigh - patch_size[0])  # 为原始的数组添加一些value值
    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x, imwidth - patch_size[1])
    # print(range_y)
    num_width = len(range_y)
    num_length = len(range_x)
    #对图像进行切割，将切割后的图片有掩膜的图片保存
    for i in range(0,num_width):
        for j in range(0,num_length):
            pic = image[range_y[i]:range_y[i]+patch_size[0], range_x[j]:range_x[j] + patch_size[1]:]
            pic2 = image2[range_y[i]:range_y[i]+patch_size[0], range_x[j]:range_x[j] + patch_size[1]:]
            print (len(pic[pic==0]))
            num_0 = len(pic[pic == 0])
            if num_0 < 3145728:
                # data_split = (range_y[i]: range_y[i] + patch_size[0], range_x[j] : range_x[j] + patch_size[1])
                result_path = target + '{}_{}_{}.png'.format(name, i+1, j+1)
                cv2.imwrite(result_path, pic)
                result_path = target2 + '{}_{}_{}.png'.format(name2, i + 1, j + 1)
                cv2.imwrite(result_path, pic2)


if __name__ == '__main__':
    patch_size = (1024, 1024)
    stride = 1000
    index = 0
    index_mask = 0
    index_mv = 0
    index_chu = 0
    pic_form = 'png'
    mask_form = 'png'
    # print(2)
    pic_name = []
    mask_name = []
    move_name = []
    pic_files = os.listdir(file_dir_input)
    mask_files = os.listdir(mask_dir_input)
    move_files = os.listdir(mask_target)#切割掩膜后的图片文件
    # print(mask_files)
    # print(pic_files)
    for file in mask_files:#获取掩膜图片名称
        m_name1,ext = file.split('.', 1)
        for name in m_name1:
            m_name, yu = m_name1.split('_gt', 1)
            # print(m_name)
        for file2 in pic_files:  # 将原图切割成小图片
            m_name2,ext =file2.split('.', 1)
            # print(m_name2)
            if m_name2 == m_name:
                index_mask += 1
                mask_name.append(m_name)
                mask_image = cv2.imread(mask_dir_input + '/' + file)
                pic_image = cv2.imread(file_dir_input + '/' + file2)
                # print(1)
                cut_image(mask_image,pic_image, m_name,m_name2, mask_target,pic_target)
                # print(2)


    # for file in move_files:
    #     mv_name,ext = file.split('.',1)
    #     print(mv_name)
    #     if ext == mask_form:
    #         index_mv += 1
    #         move_name.append(mv_name)
    #
    # for file in pic_files:
    #     name, ext = file.split('.', 1)
    #     if ext == pic_form:
    #         index += 1
    #         pic_name.append(name)
    #         image = cv2.imread(file_dir_input + '/' + file)
    #         cut_image(image, pic_name, pic_target, pic_form)
    #         if name in move_name:
    #             shutil.copy(file, pic_dir_output)
    #             index_chu += 1
    #             print('已复制{}到指定文件夹'.format(file))


