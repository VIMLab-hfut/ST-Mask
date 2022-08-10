# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
torch.cuda.current_device()
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import sys
sys.path.append("./detection")#根目录
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import cv2
import cv2_util

import random

class PennFudanDataset(object):#定义数据类
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        #在当前工作目录下获取所有排序号的文件名存入一个list
        self.imgs = list(sorted(os.listdir(os.path.join(root, "jpg_koletor"))))#原图存放位置
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask_kolektor"))))#掩膜存放文件夹

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "jpg_koletor", self.imgs[idx])#图片存放位置
        mask_path = os.path.join(self.root, "mask_kolektor", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        #图片打开方式，将原图转换为RGB格式,mask不用转换为RGB模式，因为背景为零，其他每种颜色代表一个实例
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        #把PIL图像转换为NUMPY数组，得到mask中的实例编码并去掉背景
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it，去掉背景像素，每张图片中只有目标
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]#将掩膜图片转换为单通道图片，PIL读入时仍为3通道处理

        # get bounding box coordinates for each mask，获取每个掩膜的边框位置
        num_objs = len(obj_ids)#每一个mask边界框坐标
        # print(num_objs)
        boxes = []
        # print(type(masks))
        #labels = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # labels = torch.tensor((num_objs,), dtype=torch.int64)
        # print(labels)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)#使用的maskrcnn模型，以resnet50，fpn为骨干框架

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
 
    return (b,g,r)


def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)  # 255也可以改为256

def PredictImg( image, model,device):
    #img, _ = dataset_test[0] 
    img = cv2.imread(image)
    result = img.copy()
    dst=img.copy()
    img=toTensor(img)

    names = {'0': 'background', '1': 'kolektor'}
    # put the model in evaluati
    # on mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks=prediction[0]['masks']
 
    m_bOK=False;
    for idx in range(boxes.shape[0]):
        if scores[idx] >= 0.8:  #阈值
            m_bOK=True;
            color=random_color()
            mask=masks[idx, 0].mul(255).byte().cpu().numpy()
            thresh = mask
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(dst, contours, -1, color, -1)

            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(result,(x1,y1),(x2,y2),color,thickness=2)
            cv2.putText(result, text=name, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)

            dst1 = cv2.addWeighted(result, 0.7, dst, 0.3, 0)

            
 
    if m_bOK:
        cv2.imshow('result', dst1)
        cv2.waitKey()
        cv2.destroyAllWindows()


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2  #需要修改种类,种类加背景
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:2000]) #训练集张数,torch.utils.data.Subset，获取指定一个索引序列对应的子数据集
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[2000:])#测试集张数，分类测试机和训练集##############需修改

    dataset = torch.utils.data.Subset(dataset,indices[:int(len(indices)*0.9)])
    dataset_test = torch.utils.data.Subset(dataset_test,indices[int(len(indices)*0.9):])



    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)#batch_size

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002,#学习率
                                momentum=0.9, weight_decay=0.0005) #momentum用于梯度下降中加速模型收敛，后者为损失函数正则化项的系数
    # and a learning rate scheduler，等间隔减小学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=120,
                                                   gamma=0.1)

    model_without_ddp = model

    # let's train it for 10 epochs
    num_epochs = 301  #训练次数
    #
    # Resume = True
    # if Resume:
    #     path_checkpoint = 'E:\TorchVision_Maskrcnn\Maskrcnn\pth_xz_all/model_cracks_110_split_RPN_CBMA.pth'
    #     checkpoint = torch.load(path_checkpoint,map_location = torch.device('cuda'))
    #     model.load_state_dict(checkpoint)

    if os.path.exists(os.path.join('D:/maskrcnn pytorch/TorchVision_Maskrcnn/Maskrcnn/pth_03_cfx_best/model_cfx_100_update.pth')):
        checkpoint = torch.load(os.path.join('D:/maskrcnn pytorch/TorchVision_Maskrcnn/Maskrcnn/pth_03_cfx_best/model_cfx_100_update.pth'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = 0
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # epoch = checkpoint(['epoch'])
        # start_epoch = checkpoint[epoch]
        print('加载epoch{}成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，从头开始训练！')

    # model.load_state_dict(checkpoint['model'])
    for epoch in range(start_epoch+1, num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        # checkpoint = torch.load('model_cfx1117_50.pth')
        if epoch % 5 == 0:
            utils.save_on_master({
                'model': model_without_ddp.state_dict()},
                os.path.join('./pth_org_kolektor/', 'model_kolektor_datasets_{}_org.pth'.format(epoch)))

            # utils.save_on_master({
    #         'model': model_without_ddp.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'lr_scheduler': lr_scheduler.state_dict()},
    #         os.path.join('./', 'model_{}.pth'.format(epoch)))

    # utils.save_on_master({
    # 'model': model_without_ddp.state_dict()},
    # os.path.join('./', 'model_cfx_50.pth'))#训练完之后生成脚本

        # # #在预训练过的模型上继续训练
        # target_model = model.to(device)
        # checkpoint = torch.load('model_cfx1117_50.pth')
        # model_without_ddp = target_model.load_state_dict(checkpoint)


    print("That's it!")
    # PredictImg("2.jpg",model,device)

    
if __name__ == "__main__":
    main()
