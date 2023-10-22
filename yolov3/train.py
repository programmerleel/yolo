from typing import Any
from data import train_dataloader,test_dataloader
from torch import nn,optim,cuda,version,backends
from net import ResNet_18
import torch
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot

class Train():
    def __init__(self) -> None:
        super(Train,self).__init__()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.net = ResNet_18()
        self.net.linear_1[0] = nn.Linear(512,10,bias=False)
        self.net.linear_1[1] = nn.BatchNorm1d(10)
        self.net.to('cuda')
        self.opt = optim.Adam(self.net.parameters())
        self.loss = nn.MSELoss()
        self.summaryWriter = SummaryWriter("/home/code/ResNet18/logs")
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
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if os.path.exists("/home/code/ResNet18/weights/best.pt"):
            torch.load("/home/code/ResNet18/weights/best.pt")
        for epoch in range(50):
            sum_loss = 0
            last_loss = 0
            sum_accuracy = 0
            loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),leave=False)
            for i,e in loop:
                self.net.train()
                image,label = e[0],e[1]
                label_one_hot = one_hot(label,10)
                image,label,label_one_hot = image.to('cuda'),label.to('cuda'),label_one_hot.to('cuda')
                label_one_hot = label_one_hot.float()
                out = self.net(image)
                loss = self.loss(out,label_one_hot)
                out_argmax = torch.argmax(out,dim=1)
                accuracy = (label == out_argmax).sum()/label.shape[0]
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                sum_loss = sum_loss + loss.item()
                sum_accuracy = sum_accuracy + accuracy
                loop.set_description(f'Epoch [{epoch}/{50}]')
                loop.set_postfix(loss = loss.item(),acc = accuracy)
            avg_loss = sum_loss / len(self.train_dataloader)
            avg_accuracy = sum_accuracy / len(self.train_dataloader)
            self.summaryWriter.add_scalar("avg_loss",avg_loss,epoch)
            self.summaryWriter.add_scalar("avg_accuracy",avg_accuracy,epoch)
            torch.save(self.net,"/home/code/ResNet18/weights/last.pt")
            if epoch == 0:
                last_loss = avg_loss
                torch.save(self.net,"/home/code/ResNet18/weights/best.pt")
            else:
                if avg_loss < last_loss:
                    last_loss = avg_loss
                    torch.save(self.net,"/home/code/ResNet18/weights/best.pt")
            
            sum_test_loss = 0
            sum_test_accuracy = 0
            with torch.no_grad():
                loop_test = tqdm(enumerate(self.test_dataloader),total=len(self.test_dataloader),leave=False)
                for i,e in loop_test:
                    self.net.eval()
                    image,label = e[0],e[1]
                    label_one_hot = one_hot(label,10)
                    image,label,label_one_hot = image.to('cuda'),label.to('cuda'),label_one_hot.to('cuda')
                    label_one_hot = label_one_hot.float()
                    out = self.net(image)
                    loss = self.loss(out,label_one_hot)
                    out_argmax = torch.argmax(out,dim=1)
                    accuracy = (label == out_argmax).sum()/label.shape[0]
                    sum_test_loss = sum_loss + loss.item()
                    sum_test_accuracy = sum_accuracy + accuracy
                    #更新信息
                    loop_test.set_description(f'Epoch [{epoch}/{50}]')
                    loop_test.set_postfix(test_loss = loss.item(),test_acc = accuracy)
                avg_test_loss = sum_test_loss / len(self.train_dataloader)
                avg_test_accuracy = sum_test_accuracy / len(self.train_dataloader)
                self.summaryWriter.add_scalar("avg_test_loss",avg_test_loss,epoch)
                self.summaryWriter.add_scalar("avg_test_accuracy",avg_test_accuracy,epoch)

if __name__ == "__main__":
    train = Train()
    train()
