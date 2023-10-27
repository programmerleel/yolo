import argparse
from dataset import CocoDataset
from net import yolov3
import os
import sys
import torch
from torch import nn,optim,cuda,version,backends
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Train():
    def __init__(self,images_path,labels_path,size,anchors,nc,c,c1,logs_path,weights_path,epochs,batch_size) -> None:
        super(Train,self).__init__()
        self.train_dataset = CocoDataset(images_path,labels_path,size,"train",anchors,nc)
        self.val_dataset = CocoDataset(images_path,labels_path,size,"val",anchors,nc)
        self.train_dataloader = DataLoader(self.train_dataset,batch_size,True)
        self.val_dataloader = DataLoader(self.val_dataset,batch_size,True)
        self.net = yolov3(c,c1,nc)
        self.net.to('cuda')
        self.opt = optim.Adam(self.net.parameters())
        self.summaryWriter = SummaryWriter(logs_path)
        self.weights_path = weights_path
        self.epochs = epochs
        print("显卡信息")
        print(cuda.get_device_properties(0))
        print("cuda版本")
        print(version.cuda)
        print("cudnn版本")
        print(backends.cudnn.version())
        print("pytorch版本")
        print(torch.__version__)
        print("网络结构")
        print(self.net)
    
    def __call__(self):
        # 断点续训
        if os.path.exists(os.path.join(self.weights_path,"best.pt")):
            torch.load(os.path.join(self.weights_path,"best.pt"))
        for epoch in range(self.epochs):
            sum_loss = 0
            sum_loss_pre = 0
            sum_loss_cls = 0
            sum_loss_box = 0
            last_loss = 0
            self.net.train()
            loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),leave=False)
            for i,e in loop:
                # loop.update(self.batch_size)
                image,label_13,label_26,label_52 = e[0],e[1],e[2],e[3]
                image,label_13,label_26,label_52 = image.to('cuda'),label_13.to('cuda'),label_26.to('cuda'),label_52.to("cuda")
                out_13,out_26,out_52 = self.net(image)
                loss_13,loss_13_pre,loss_13_cls,loss_13_box = loss_func(label_13,out_13,0.2)
                loss_26,loss_26_pre,loss_26_cls,loss_26_box = loss_func(label_26,out_26,0.2)
                loss_52,loss_52_pre,loss_52_cls,loss_52_box = loss_func(label_52,out_52,0.2)
                loss = loss_13+loss_26+loss_52
                loss_pre = loss_13_pre+loss_26_pre+loss_52_pre
                loss_cls = loss_13_cls+loss_26_cls+loss_52_cls
                loss_box = loss_13_box+loss_26_box+loss_52_box
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                sum_loss = sum_loss + loss.item()
                sum_loss_pre = sum_loss_pre + loss_pre.item()
                sum_loss_cls = sum_loss_cls + loss_cls.item()
                sum_loss_box = sum_loss_box + loss_box.item()
                loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss = loss.item(),pre_loss = loss_pre.item(),cls_loss = loss_cls.item(),box_loss = loss_box.item())
            avg_loss = sum_loss / len(self.train_dataloader)
            avg_loss_pre = sum_loss_pre / len(self.train_dataloader)
            avg_loss_cls = sum_loss_cls / len(self.train_dataloader)
            avg_loss_box = sum_loss_box / len(self.train_dataloader)
            self.summaryWriter.add_scalar("avg_loss",avg_loss,epoch)
            self.summaryWriter.add_scalar("pre_avg_loss",avg_loss_pre,epoch)
            self.summaryWriter.add_scalar("cls_avg_loss",avg_loss_cls,epoch)
            self.summaryWriter.add_scalar("box_avg_loss",avg_loss_box,epoch)
            torch.save(self.net,os.path.join(self.weights_path,"last.pt"))
            if epoch == 0:
                last_loss = avg_loss
                torch.save(self.net,os.path.join(self.weights_path,"best.pt"))
            else:
                if avg_loss < last_loss:
                    last_loss = avg_loss
                    torch.save(self.net,os.path.join(self.weights_path,"best.pt"))
            sum_loss = 0
            sum_loss_pre = 0
            sum_loss_cls = 0
            sum_loss_box = 0
            last_loss = 0
            with torch.no_grad():
                loop_test = tqdm(enumerate(self.val_dataloader),total=len(self.val_dataloader),leave=False)
                for i,e in loop_test:
                    self.net.eval()
                    image,label_13,label_26,label_52 = e[0],e[1],e[2],e[3]
                    image,label_13,label_26,label_52 = image.to('cuda'),label_13.to('cuda'),label_26.to('cuda'),label_52.to("cuda")
                    out_13,out_26,out_52 = self.net(image)
                    loss_13,loss_13_pre,loss_13_cls,loss_13_box = loss_func(label_13,out_13,0.2)
                    loss_26,loss_26_pre,loss_26_cls,loss_26_box = loss_func(label_26,out_26,0.2)
                    loss_52,loss_52_pre,loss_52_cls,loss_52_box = loss_func(label_52,out_52,0.2)
                    loss = loss_13+loss_26+loss_52
                    loss_pre = loss_13_pre+loss_26_pre+loss_52_pre
                    loss_cls = loss_13_cls+loss_26_cls+loss_52_cls
                    loss_box = loss_13_box+loss_26_box+loss_52_box
                    sum_loss = sum_loss + loss.item()
                    sum_loss_pre = sum_loss_pre + loss_pre.item()
                    sum_loss_cls = sum_loss_cls + loss_cls.item()
                    sum_loss_box = sum_loss_box + loss_box.item()
                    loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(val_loss = loss.item(),val_pre_loss = loss_pre.item(),val_cls_loss = loss_cls.item(),val_box_loss = loss_box.item())
                avg_loss = sum_loss / len(self.train_dataloader)
                avg_loss_pre = sum_loss_pre / len(self.train_dataloader)
                avg_loss_cls = sum_loss_cls / len(self.train_dataloader)
                avg_loss_box = sum_loss_box / len(self.train_dataloader)
                self.summaryWriter.add_scalar("val_avg_loss",avg_loss,epoch)
                self.summaryWriter.add_scalar("val_pre_avg_loss",avg_loss_pre,epoch)
                self.summaryWriter.add_scalar("val_cls_avg_loss",avg_loss_cls,epoch)
                self.summaryWriter.add_scalar("val_box_avg_loss",avg_loss_box,epoch)

# 损失函数包括三部分 box损失 置信度损失 类别损失
def loss_func(target,output,c):
    # 1 255 13 13 -> 1 13 13 255 
    # 13 13 3 85
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0),output.size(1), output.size(2), 3, -1)
    # 是否存在目标
    mask_obj = target[..., 0] > 0
    # 置信度损失
    pre_loss = nn.BCELoss()
    loss_pre = pre_loss(torch.sigmoid(output[..., 0]).float(), target[..., 0].float())
    # box损失
    box_loss = nn.MSELoss()
    loss_box = box_loss(output[mask_obj][..., 1:5].float(), target[mask_obj][..., 1:5].float())
    # 类别损失
    cls_loss = nn.CrossEntropyLoss()
    loss_cls = cls_loss(output[mask_obj][..., 5:],
                                    torch.argmax(target[mask_obj][..., 5:], dim=1, keepdim=True).squeeze(dim=1))
    loss = c * loss_pre + (1 - c) * 0.5 * loss_box + (1 - c) * 0.5 * loss_cls
    return loss,loss_pre,loss_cls,loss_box

def main(args):
    images_path = args.images_path
    labels_path = args.labels_path
    size = args.size
    anchors = args.anchors
    nc = args.nc
    c = args.c
    c1 = args.c1
    logs_path = args.logs_path
    weights_path = args.weights_path
    epochs = args.epochs
    batch_size = args.batch_size
    train = Train(images_path,labels_path,size,anchors,nc,c,c1,logs_path,weights_path,epochs,batch_size)
    train()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path',type=str,help='',default='/home/data/trash/images')
    parser.add_argument('--labels_path',type=str,help='',default='/home/data/trash/labels')
    parser.add_argument('--size',type=int,help='',default=416)
    parser.add_argument('--anchors',type=dict,help='',default={13:[116,90, 156,198, 373,326],26:[30,61, 62,45, 59,119],52:[10,13, 16,30, 33,23]})
    parser.add_argument('--nc',type=int,help='',default=80)
    parser.add_argument('--c',type=int,help='',default=3)
    parser.add_argument('--c1',type=int,help='',default=32)
    parser.add_argument('--logs_path',type=str,help='',default='/home/code/yolo/yolov3/logs')
    parser.add_argument('--weights_path',type=str,help='',default='/home/code/yolo/yolov3/weights')
    parser.add_argument('--epochs',type=int,help='',default=300)
    parser.add_argument('--batch_size',type=int,help='',default=4)
    return parser.parse_args(argv)

if __name__=='__main__':
    main(parse_arguments(sys.argv[1:]))
