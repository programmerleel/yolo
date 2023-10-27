# 数据加载类
# 整个yolo系列的精髓部分，在制作数据的Dataset类时基本就能够理清
# 一句话总结，知道数据如何进行制作，那么就知道yolo是如何进行运作的
import argparse
import cv2
import math
import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class CocoDataset(Dataset):
    def __init__(self,images_path,labels_path,size,type,anchors,nc):
        self.size = size
        self.type = type
        self.anchors = anchors
        self.nc = nc
        if self.type=="train":
            self.train_dataset = []
            train_folder = "train"
            for file in os.listdir(os.path.join(images_path,train_folder)):
                image_path = os.path.join(images_path,train_folder,file)
                label_path = os.path.join(labels_path,train_folder,file[0:-4]+".txt")
                self.train_dataset.append((image_path,label_path))
        elif self.type=="val":
            self.val_dataset = []
            val_folder = "val"
            for file in os.listdir(os.path.join(images_path,val_folder)):
                image_path = os.path.join(images_path,val_folder,file)
                label_path = os.path.join(labels_path,val_folder,file[0:-4]+".txt")
                self.val_dataset.append((image_path,label_path))
 
    def __len__(self):
        if self.type=="train":
            return len(self.train_dataset)
        elif self.type=="val":
            return len(self.val_dataset)
 
    def __getitem__(self, index):
        if self.type=="train":
            data = self.train_dataset[index]
        elif self.type=="val":
            data = self.val_dataset[index]
        image_path = data[0]
        label_path = data[1]
        resize_image,labels = resize(image_path,label_path,self.size)
        #三组尺寸 13 26 52
        labels_feature_size = {}
        for feature_size,anchor in self.anchors.items():
            #三个形状的anchor
            labels_feature_size[feature_size] = np.zeros((feature_size,feature_size,3,1+4+self.nc))
            for label in labels:
                class_num = label[0]
                center_x,center_y = label[1],label[2]
                x,x_index = math.modf(center_x*feature_size)
                y,y_index = math.modf(center_y*feature_size)
                width_,height_ = label[3],label[4]
                for i in range(len(anchor)//2):
                    iou = min(width_*self.size*height_*self.size,anchor[i*2]*anchor[i*2+1])/max(width_*self.size*height_*self.size,anchor[i*2]*anchor[i*2+1])
                    w_ = width_*self.size/anchor[i*2]
                    h_ = height_*self.size/anchor[i*2+1]
                    labels_feature_size[feature_size][int(y_index),int(x_index),i]=np.array([iou,x,y,np.log(w_),np.log(h_),*one_hot(self.nc,int(class_num))])
        return resize_image,torch.from_numpy(labels_feature_size[13]),torch.from_numpy(labels_feature_size[26]),torch.from_numpy(labels_feature_size[52])
 
 # 处理图片尺寸与坐标
def resize(image_path,label_path,size):
    image = cv2.imread(image_path)
    h,w,_ = image.shape
    len = max(h,w)
    top = int((len-h)/2)
    bottom = int(len-h-top)
    left = int((len-w)/2)
    right = int(len-w-left)
    image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,(0,0,0))
    resize_image = cv2.resize(image,(size,size))
    image_tensor = transforms.ToTensor()(resize_image)
    # image_tensor = torch.unsqueeze(image_tensor,0)
    label = open(label_path,"r")
    labels = []
    for data in label.readlines():
        data = data.split(" ")
        class_num = int(data[0])
        center_x,center_y = float(data[1]),float(data[2])
        width_,height_ = float(data[3]),float(data[4])
        label = [class_num,center_x,center_y,width_,height_]
        labels.append(label)
    return image_tensor,labels

def one_hot(nc,class_num):
    oh=np.zeros(nc)
    oh[class_num]=1
    return oh

def main(args):
    images_path = args.images_path
    labels_path = args.labels_path
    size = args.size
    anchors = args.anchors
    nc = args.nc
    dataset = CocoDataset(images_path,labels_path,size,"val",anchors,nc)
    dataloader = DataLoader(dataset,16,True)
    for i,e in enumerate(dataloader):
        print(e[0].shape)
        print(e[1].shape)
        print(e[2].shape)
        print(e[3].shape)
        exit()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path',type=str,help='',default='/home/data/coco/images')
    parser.add_argument('--labels_path',type=str,help='',default='/home/data/coco/labels')
    parser.add_argument('--size',type=int,help='',default=416)
    parser.add_argument('--anchors',type=dict,help='',default={13:[116,90, 156,198, 373,326],26:[30,61, 62,45, 59,119],52:[10,13, 16,30, 33,23]})
    parser.add_argument('--nc',type=int,help='',default=80)
    return parser.parse_args(argv)

if __name__=='__main__':
    main(parse_arguments(sys.argv[1:]))
