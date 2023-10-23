# COCO格式数据集 转化为 YOLO格式数据集
 
import os
import json
from tqdm import tqdm
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='/home/data/coco/annotations_trainval2017/annotations/instances_val2017.json',type=str, help="coco json")
parser.add_argument('--save_path', default='/home/data/coco', type=str, help="yolo txt")
arg = parser.parse_args()
 
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)
 
if __name__ == '__main__':
    json_file = arg.json_path
    ana_txt_save_path = arg.save_path
 
    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
 
    # coco数据集json文件中id并不连续
    id_map = {}
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i

    # 图像路径文件位置
    list_file = open(os.path.join(ana_txt_save_path, 'train2017.txt'), 'w')
    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(ana_txt_save_path, "labels", "val", ana_txt_name), 'w')
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        #将图片的相对路径写入train2017或val2017的路径
        list_file.write('/home/data/coco/val2017/%s.jpg\n' %(head))
    list_file.close()