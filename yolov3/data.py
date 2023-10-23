# 数据加载类

from torch.utils.data import Dataset,DataLoader
import os
import cv2
import numpy as np
class CocoDataset(Dataset):
    def __init__(self,images_path,labels_path):
        self.train_dataset = []
        self.val_dataset = []
        train_folder = "train"
        val_folder = "val"
        for file in os.listdir(images_path + "/" + train_folder + "/" + tag):
            image_path = images_path + "/" + train_folder + "/" + file
            label_path = labels_path + "/" + train_folder + "/" + file[:-4]+".txt"
            #存储图片的路径和标签
            self.train_dataset.append((image_path,label_path))
        for file in os.listdir(images_path + "/" + val_folder + "/" + tag):
            image_path = images_path + "/" + val_folder + "/" + file
            label_path = labels_path + "/" + val_folder + "/" + file[:-4]+".txt"
            #存储图片的路径和标签
            self.val_dataset.append((image_path,label_path))
 
    def __len__(self):
        #返回数据集的长度
        return len(self.dataset)
 
    def __getitem__(self, index):
        data = self.dataset[index]
        #获取图像的路径
        imaeg_path = data[0]
        #获取图像的标签
        label_path = data[1]
        #利用opencv读取图片
        img = cv2.imread(imaeg_path,0)
        #归一化处理
        img = img / 255
        return np.float32(img)
 
if __name__ == '__main__':
    pass
