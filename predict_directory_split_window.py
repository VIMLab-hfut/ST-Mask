import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import sys
sys.path.append("./detection")
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import cv2
import cv2_util

import random
import time
import datetime



def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
 
    return (b,g,r)


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)  # 255也可以改为256


def cut_image(image,name,target):
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
    for i in range(0, num_width):
        for j in range(0, num_length):
            pic = image[range_y[i]:range_y[i]+patch_size[0], range_x[j]:range_x[j] + patch_size[1]:]
            result_path = target + '{}_{}_{}_gt.png'.format(name, i+1, j+1)
            cv2.imwrite(result_path, pic)


def PredictImg( img, model, device, textnum, image_name, save_path_split):
    #img, _ = dataset_test[0] 
    # img = cv2.imread(image)

    result = img.copy()
    result_split = img.copy()

    dst = img.copy()
    dst1 = img
    img = toTensor(img)

    names = {'0': 'background', '1': 'FD','2':'YD','3':'WD'}
    # names = {'0': 'background','1':'person'}
    # put the model in evaluati
    # on mode
    if len(dst1.shape) == 2:  # 灰度图像
        imhigh, imwidth = dst1.shape
    if len(dst1.shape) == 3:  # RGB图像
        imhigh, imwidth, imch = dst1.shape
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
    for i in range(0, num_width):
        for j in range(0, num_length):
            pic_dst1 = dst1[range_y[i]:range_y[i]+patch_size[0], range_x[j]:range_x[j] + patch_size[1]:]
            pic_dst = dst[range_y[i]:range_y[i]+patch_size[0], range_x[j]:range_x[j] + patch_size[1]:]
            pic = dst[range_y[i]:range_y[i]+patch_size[0], range_x[j]:range_x[j] + patch_size[1]:]
            pic_re = result[range_y[i]:range_y[i]+patch_size[0], range_x[j]:range_x[j] + patch_size[1]:]

            pic = toTensor(pic)

            # result_path = target + '{}_{}_{}_gt.png'.format(name, i+1, j+1)

            prediction = model([pic.to(device)])

            boxes = prediction[0]['boxes']
            labels = prediction[0]['labels']
            scores = prediction[0]['scores']
            masks = prediction[0]['masks']
            flag = 0

            m_bOK = False
            for idx in range(boxes.shape[0]):
                if scores[idx] >= 0.7:#阈值
                    m_bOK = True
                    flag += 1

                    color = random_color()
                    mask = masks[idx, 0].mul(255).byte().cpu().numpy()
                    thresh = mask
                    contours, hierarchy = cv2_util.findContours(
                        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )

                    # split_pic(result_split, contours, save_path_split, image_name)
                    cv2.drawContours(pic_dst, contours, -1, color, -1)
                    result_path = save_path_split + '/' + '{}_{}_{}_src.png'.format(image_name, i + 1, j + 1)
                    cv2.imwrite(result_path, pic_re)
                    result_path = save_path_split + '/' + '{}_{}_{}_bl.png'.format(image_name, i + 1, j + 1)
                    cv2.imwrite(result_path, thresh)
                    x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
                    name = names.get(str(labels[idx].item()))
                    cv2.rectangle(pic_re, (x1, y1), (x2, y2), color, thickness=2)
                    cv2.putText(pic_re, text=name, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)
                    pic_dst1 = cv2.addWeighted(pic_re, 0.7, pic_dst, 0.5, 0)
                    result_path = save_path_split + '/' + '{}_{}_{}.png'.format(image_name, i + 1, j + 1)
                    cv2.imwrite(result_path, pic_dst1)

    if flag > 0:
        textnum += 1
    # if m_bOK:
    # #     cv2.imshow('result', dst1)
    # #     cv2.waitKey()
    # #     cv2.destroyAllWindows()
    #       cv2.namedWindow('result1', 0)
    #       cv2.resizeWindow('result1', 2500, 3000)
    #       cv2.imshow('result1', dst1)
    return dst1, textnum

def split_pic(image, contours, save_path, image_name):
    for i in range(len(contours)):
        # result_1 = np.zeros((w, h, 3), dtype='uint8')
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)  # 计算点集最外面的矩形边界
        result_1 = image[y:y+h, x:x+w]
        # show(result_1, 'xx', 800, 600)
        cv2.imwrite(save_path + '/' + image_name, result_1)



if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #检测设备
    num_classes = 4 #检测种类
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes) #沿用的检测模型
    model.to(device)
    model.eval()
    save = torch.load('E:\TorchVision_Maskrcnn\Maskrcnn\pth_xz_all/model_xz_200_split_DCN_RPN.pth')
    model.load_state_dict(save['model'])
    # 车缝线疵点
    test_path = "E:/TorchVision_Maskrcnn/Maskrcnn/PennFudanPed/xz_test_0318_pic"

    # 车缝线保存路径
    # save_path = "D:/maskrcnn pytorch/TorchVision_Maskrcnn/Maskrcnn/detect_cfx/output_split_0221_complicate"

    save_path_split = "E:\TorchVision_Maskrcnn\Maskrcnn\detect_xz/output_test_xz_split_DCN_RPN_200_0318"
    patch_size = (1024, 1024)
    stride = 1000


    path = []
    for root, dirs, files in os.walk(test_path):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        for f in files:
            path.append(f)
            os.path.join(root, f)
    detect_number = len(path)
    print('总检测张数：', detect_number)
    textnum, num, totoltime = 0, 0, 0
    for i in path:
        m_name, ext = i.split('.', 1)

        selectpath = os.path.join(test_path, i)
        start = time.clock()
        try:
            image = cv2.imread(selectpath)
        except:
            print('Open Error! Try again!')
            continue
        # image, textnum = mask_rcnn.detect_image(image, textnum)
        image, textnum = PredictImg(image, model, device, textnum, m_name, save_path_split)

        # 保存图像
        # image.save(save_path + '/' + i)
        # cv2.imwrite(save_path + '/' + i, image)
        end = time.clock()
        print(str(end - start))
        totoltime += end - start
        num = num + 1
        print("检测完" + str(num) + "张\n")
    # textnum = detect_number - textnum
    percent = textnum / detect_number
    print("总共抽样" + str(detect_number) + "张图片，检测出来了" + str(textnum) + "张图片,检测率为" + str(percent) + ",总耗时" + str(
        totoltime) + "秒")
    # print(2)
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))
    # print(total_time)
    # cv2.waitKey()
    # cv2.destroyAllWindows()