# yolov3 网络构建

import argparse
import sys
import torch
import torch.nn as nn

# 自适应padding
def auto_pad(k,p):
    if p==None:
        return k//2
    
# cbl模块
class Conv(nn.Module):
    def __init__(self,c1,c2,k,s,p=None,d=1,g=1,b=False) -> None:
        super().__init__()
        self.cv = nn.Conv2d(c1,c2,k,s,p,d,g,b)
        self.bn = nn.BatchNorm2d(c2)
        self.ac = nn.LeakyReLU()

    def forward(self,x):
        return self.ac(self.bn(self.cv(x)))

# 残差模块
class ResidualConv(nn.Module):
    def __init__(self,c1,c2) -> None:
        super().__init__()
        # 降低通道
        self.cv1 = Conv(c1,c2,k=1,s=1,p=None,g=1,d=1,b=False)
        # 还原通道理
        self.cv2 = Conv(c2,c1,k=3,s=2,p=None,g=1,d=1,b=False)

    def forward(self,x):
        # 残差结构
        return self.cv2(self.cv1(x))+x

# # 通道加倍 尺寸减半
# class HalfConv(nn.Module):
#     def __init__(self,c1,c2) -> None:
#         super().__init__()
#         self.cv = Conv(c1,c2,k=3,s=2,p=None,d=1,g=1,b=False)

#     def forward(self,x):
#         return self.cv(x)

# 连续卷积
class SetConv(nn.Module):
    def __init__(self,c1,c2) -> None:
        super().__init__()
        self.cv1 = Conv(c1,c2,k=1,s=1,p=None,d=1,g=1,b=False)
        self.cv2 = Conv(c2,c1,k=3,s=1,p=None,d=1,g=1,b=False)
        self.cv3 = Conv(c1,c2,k=1,s=1,p=None,d=1,g=1,b=False)
        self.cv4 = Conv(c2,c1,k=3,s=1,p=None,d=1,g=1,b=False)
        self.cv5 = Conv(c1,c2,k=1,s=1,p=None,d=1,g=1,b=False)

    def forward(self,x):
        return self.cv5(self.cv4(self.cv3(self.cv2(self.cv1))))

# 一般是采用UpSample来进行上采样，这里使用反卷积
class TransposeConv(nn.Module):
    def __init__(self,c,k=2,s=2) -> None:
        super().__init__()
        self.tc = nn.LazyConvTranspose2d(c,k,s)

    def forward(self,x):
        return self.tc(x)
    
# 拼接基本模块
class yolov3(nn.Module):
    # c 初始化通道数 3
    # c1 第一次升维通道数 32
    def __init__(self,c,c1,c2) -> None:
        super().__init__()
        self.cv1 = Conv(c,c1,k=3,s=1,p=None,d=1,g=1,b=False)
        self.hc1 = Conv(c1,c1*2,k=3,s=2,p=None,d=1,g=1,b=False)
        self.rc1 = ResidualConv(c1*2,c1)
        self.hc2 = Conv(c1*2,c1*4,k=3,s=2,p=None,d=1,g=1,b=False)
        self.rc2 = ResidualConv(c1*4,c1*2)
        self.rc3 = ResidualConv(c1*4,c1*2)
        self.hc3 = Conv(c1*4,c1*8,k=3,s=2,p=None,d=1,g=1,b=False)
        self.rc4 = ResidualConv(c1*8,c1*4)
        self.rc5 = ResidualConv(c1*8,c1*4)
        self.rc6 = ResidualConv(c1*8,c1*4)
        self.rc7 = ResidualConv(c1*8,c1*4)
        self.rc8 = ResidualConv(c1*8,c1*4)
        self.rc9 = ResidualConv(c1*8,c1*4)
        self.rc10 = ResidualConv(c1*8,c1*4)
        self.rc11 = ResidualConv(c1*8,c1*4)
        self.hc4 = Conv(c1*8,c1*16,k=3,s=2,p=None,d=1,g=1,b=False)
        self.rc12 = ResidualConv(c1*16,c1*8)
        self.rc13 = ResidualConv(c1*16,c1*8)
        self.rc14 = ResidualConv(c1*16,c1*8)
        self.rc15 = ResidualConv(c1*16,c1*8)
        self.rc16 = ResidualConv(c1*16,c1*8)
        self.rc17 = ResidualConv(c1*16,c1*8)
        self.rc18 = ResidualConv(c1*16,c1*8)
        self.rc19 = ResidualConv(c1*16,c1*8)
        self.hc4 = Conv(c1*16,c1*32,k=3,s=2,p=None,d=1,g=1,b=False)
        self.rc20 = ResidualConv(c1*32,c1*16)
        self.rc21 = ResidualConv(c1*32,c1*16)
        self.rc22 = ResidualConv(c1*32,c1*16)
        self.rc23 = ResidualConv(c1*32,c1*16)
        self.sc1 = SetConv(c1*32,c1*16)
        self.cv2 = Conv(c1*16,c1*8,k=1,s=1,p=None,d=1,g=1,b=False)
        # 上采样 tc1 与 rc19进行拼接
        self.tc1 = TransposeConv(c1*8)
        self.sc2 = SetConv(c1*16+c1*8,c1*8)
        self.cv3 = Conv(c1*8,c1*4,k=1,s=1,p=None,d=1,g=1,b=False)
        # 上采样 tc2 与 rc11进行拼接
        self.tc2 = TransposeConv(c1*4)
        self.sc3 = SetConv(c1*8+c1*4,c1*4)
        # 输出头在forward中进行呈现

    def forward(self,x):
        cv1 = self.cv1(x)
        hc1 = self.hc1(cv1)
        rc1 = self.rc1(hc1)
        self.hc2
        self.rc2
        self.rc3
        self.hc3
        self.rc4
        self.rc5
        self.rc6
        self.rc7
        self.rc8
        self.rc9
        self.rc10
        self.rc11
        self.hc4
        self.rc12
        self.rc13
        self.rc14
        self.rc15
        self.rc16
        self.rc17
        self.rc18
        self.rc19
        self.hc4
        self.rc20
        self.rc21
        self.rc22
        self.rc23
