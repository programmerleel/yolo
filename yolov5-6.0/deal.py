import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import cv2
import os
import random
import shutil
import sys
from tqdm import tqdm

'''
处理数据集：
    1.读取csv文件 对应图片的坐标与类别
    2.生成yolo格式的标签文件
'''
def csv2txt(csv_path,images_path,labels_path,class_names):
    pool = ThreadPoolExecutor(32)
    datas = []
    tasks = []
    csv_file = open(csv_path,"r")
    csv_reader = csv.reader(csv_file)
    for i,row in enumerate(csv_reader):
        if i == 0:
            continue
        else:
            datas.append(row)
    for data in datas:
        tasks.append(pool.submit(do_txt,images_path,labels_path,class_names,data))
    for task in tqdm(as_completed(tasks),desc='cvs2txt',total=len(tasks)):
        pass

def split_data(images_path,labels_path):
    image_paths = []
    for image_name in os.listdir(images_path):
        image_path = os.path.join(images_path,image_name)
        image_paths.append(image_path)
    random.shuffle(image_paths)
    train_path = os.path.join(images_path,'train')
    val_path = os.path.join(images_path,'val')
    test_path = os.path.join(images_path,'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(os.path.join(labels_path,'train'))
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        os.makedirs(os.path.join(labels_path, 'val'))
    if not os.path.exists(test_path):
        os.makedirs(test_path)
        os.makedirs(os.path.join(labels_path, 'test'))
    for i in range(len(image_paths)):
        if i <= int(len(image_paths)*0.9):
            image_path = image_paths[i]
            image_name = os.path.basename(image_path)
            label_name = 'labels'+image_name[6:-4]+'.txt'
            shutil.copy(image_path,os.path.join(train_path,image_name))
            shutil.copy(os.path.join(labels_path,label_name),os.path.join(labels_path,'train',label_name))
        if i <= int(len(image_paths)*0.9*0.2):
            image_path = image_paths[i]
            image_name = os.path.basename(image_path)
            label_name = 'labels'+image_name[6:-4]+'.txt'
            shutil.copy(image_path, os.path.join(val_path,image_name))
            shutil.copy(os.path.join(labels_path, label_name), os.path.join(labels_path, 'val',label_name))
        if i > int(len(image_paths)*0.9):
            image_path = image_paths[i]
            image_name = os.path.basename(image_path)
            label_name = 'labels'+image_name[6:-4]+'.txt'
            shutil.move(image_path, test_path)
            shutil.move(os.path.join(labels_path, label_name), os.path.join(labels_path, 'test'))
    for image_name in os.listdir(images_path):
        image_path = os.path.join(images_path,image_name)
        if os.path.isfile(images_path):
            os.remove(image_path)
    for label_name in os.listdir(labels_path):
        label_path = os.path.join(labels_path,label_name)
        if os.path.isfile(label_path):
            os.remove(label_path)

def do_txt(images_path,labels_path,class_names,data):
    image_name = data[0]
    x_min,y_min,x_max,y_max = float(data[1]),float(data[2]),float(data[3]),float(data[4])
    class_name = data[5]
    label_name = image_name[0:-4] + '.txt'
    image_path = os.path.join(images_path,image_name)
    label_path = os.path.join(labels_path,label_name)
    image = cv2.imread(image_path)
    label = open(label_path,'a')
    height,width,_ = image.shape
    h,w = float((y_max-y_min)/height),float((x_max-x_min)/width)
    center_x,center_y = float((x_min+(x_max-x_min)/2)/width),float((y_min+(y_max-y_min)/2)/height)
    class_index = class_names.index(class_name)
    label.write("{} {} {} {} {}\n".format(class_index,center_x,center_y,w,h))
    label.close()

def main(args):
    csv_path = args.csv_path
    images_path = args.images_path
    labels_path = args.labels_path
    class_names = args.class_names
    # csv2txt(csv_path,images_path,labels_path,class_names)
    split_data(images_path,labels_path)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path',type=str,help='',default=r'D:\yolov5-6.0\dataset\annotations.csv')
    parser.add_argument('--images_path',type=str,help='',default=r'/home/data/HumanDetection/images')
    parser.add_argument('--labels_path',type=str,help='',default=r'/home/data/HumanDetection/labels')
    parser.add_argument('--class_names',type=list,help='',default=['glass','paper','metal','plastic'])
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


# import xml.etree.ElementTree as ET
# import pickle
# import os
# from os import listdir, getcwd
# from os.path import join
 
 
# def convert(size, box):
#     x_center = (box[0] + box[1]) / 2.0
#     y_center = (box[2] + box[3]) / 2.0
#     x = x_center / size[0]
#     y = y_center / size[1]
#     w = (box[1] - box[0]) / size[0]
#     h = (box[3] - box[2]) / size[1]
#     return (x, y, w, h)
 
 
# def convert_annotation(xml_files_path, save_txt_files_path, classes):
#     xml_files = os.listdir(xml_files_path)
#     print(xml_files)
#     for xml_name in xml_files:
#         print(xml_name)
#         xml_file = os.path.join(xml_files_path, xml_name)
#         out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')
#         out_txt_f = open(out_txt_path, 'w')
#         tree = ET.parse(xml_file)
#         root = tree.getroot()
#         size = root.find('size')
#         w = int(size.find('width').text)
#         h = int(size.find('height').text)
 
#         for obj in root.iter('object'):
#             difficult = obj.find('difficult').text
#             cls = obj.find('name').text
#             if cls not in classes or int(difficult) == 1:
#                 continue
#             cls_id = classes.index(cls)
#             xmlbox = obj.find('bndbox')
#             b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
#                  float(xmlbox.find('ymax').text))
#             # b=(xmin, xmax, ymin, ymax)
#             print(w, h, b)
#             bb = convert((w, h), b)
#             out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
 
# if __name__ == "__main__":
#     # 需要转换的类别，需要一一对应
#     classes1 = ['person']
#     # 2、voc格式的xml标签文件路径
#     xml_files1 = r'/home/data/VOCdevkit/VOC2007/Annotations'
#     # 3、转化为yolo格式的txt标签文件存储路径
#     save_txt_files1 = r'/home/data/VOCdevkit/VOC2007/labels'
 
#     convert_annotation(xml_files1, save_txt_files1, classes1)