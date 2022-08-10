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




def PredictImg( img, model, device, textnum, image_name, save_path_split):
    #img, _ = dataset_test[0] 
    # img = cv2.imread(image)

    result = img.copy()
    result_split = img.copy()

    dst = img.copy()
    dst1 = img
    img = toTensor(img)

    # names = {'0': 'background', '1': 'FD','2':'YD','3':'WD'}
    names = {'0': 'background','1':'cfx'}
    # put the model in evaluati
    # on mode

    prediction = model([img.to(device)])

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks=prediction[0]['masks']
    flag = 0
    flag_single = 0

    m_bOK = False
    color = random_color()
    thresh = np.zeros((dst1.shape[0], dst1.shape[1]), dtype='uint8')
    for idx in range(boxes.shape[0]):
        if scores[idx] >= 0.7:  # 阈值
            m_bOK = True
            flag += 1
            flag_single += 1
            mask = masks[idx, 0].mul(255).byte().cpu().numpy()
            # thresh = mask
            # else:
            thresh = cv2.bitwise_or(thresh, mask)#或运算
    if (flag_single != 0):
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # split_pic(result_split, contours, save_path_split, image_name)
        cv2.drawContours(dst, contours, -1, color, -1)
        result_path = save_path_split + '/' + '{}_src.png'.format(image_name)
        cv2.imwrite(result_path, result)
        result_path = save_path_split + '/' + '{}_bl.png'.format(image_name)
        cv2.imwrite(result_path, thresh)
        pic_dst1 = cv2.addWeighted(result, 0.7, dst, 0.5, 0)#图像混合加权
        result_path = save_path_split + '/' + '{}_re.png'.format(image_name)
        cv2.imwrite(result_path, pic_dst1)



    # m_bOK = False
    # for idx in range(boxes.shape[0]):
    #     if scores[idx] >= 0:#阈值
    #         m_bOK = True
    #         flag += 1
    #
    #         color = random_color()
    #         mask = masks[idx, 0].mul(255).byte().cpu().numpy()
    #         thresh = mask
    #         contours, hierarchy = cv2_util.findContours(
    #             thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    #         )
    #
    #         # split_pic(result_split, contours, save_path_split, image_name)
    #         cv2.drawContours(dst, contours, -1, color, -1)
    #
    #         x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
    #         name = names.get(str(labels[idx].item()))
    #         cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=-1)
    #         cv2.putText(result, text=name, org=(int(x1), int(y1+10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)
    #         dst1 = cv2.addWeighted(result, 0.7, dst, 0.5, 0)

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
    num_classes = 2 #检测种类
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes) #沿用的检测模型
    model.to(device)
    model.eval()
    save = torch.load('D:/maskrcnn pytorch/TorchVision_Maskrcnn/Maskrcnn/pth_03_cfx_best/model_cfx_100_update.pth')
    model.load_state_dict(save['model'])
    # 车缝线疵点
    test_path = "E:/picture_cfx/新建文件夹 (2)"

    # 车缝线保存路径
    save_path = "E:/TorchVision_Maskrcnn/Maskrcnn/detect_cfx/output_manual_150"

    #save_path_split = "E:/maskrcnn pytorch/TorchVision_Maskrcnn/Maskrcnn/detect_cfx/output_split"
    # patch_size = (1024, 1024)
    # stride = 1000


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
        selectpath = os.path.join(test_path, i)
        start = time.clock()
        try:
            image = cv2.imread(selectpath)
        except:
            print('Open Error! Try again!')
            continue
        # image, textnum = mask_rcnn.detect_image(image, textnum)
        image, textnum = PredictImg(image, model, device, textnum, i, save_path)

        # 保存图像
        # image.save(save_path + '/' + i)
        cv2.imwrite(save_path + '/' + i, image)
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