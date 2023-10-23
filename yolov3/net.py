# yolov3 网络构建
# 本身代码没有做良好的封装，照着网络结构图快速的手撸了一个
# 可以参照v5的结构去做封装（v5本身也有v3网络结构的配置文件）

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
        self.cv = nn.Conv2d(c1,c2,k,s,auto_pad(k,p),dilation=d,groups=g,bias=b)
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
        # 还原通道
        self.cv2 = Conv(c2,c1,k=3,s=1,p=None,g=1,d=1,b=False)

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
        return self.cv5(self.cv4(self.cv3(self.cv2(self.cv1(x)))))

# 一般是采用UpSample来进行上采样，这里使用反卷积
class TransposeConv(nn.Module):
    def __init__(self,c,k=2,s=2) -> None:
        super().__init__()
        self.tc = nn.LazyConvTranspose2d(c,k,s)

    def forward(self,x):
        return self.tc(x)

class HeadConv(nn.Module):
    def __init__(self,c1,c2,nc) -> None:
        super().__init__()
        self.cv1 = Conv(c1,c2,k=3,s=1,p=None,d=1,g=1,b=False)
        self.cv2 = nn.Conv2d(c2,(nc+1+4)*3,1,1)

    def forward(self,x):
        return self.cv2(self.cv1(x))
    
# 拼接基本模块
class yolov3(nn.Module):
    # c 初始化通道数 3
    # c1 第一次升维通道数 32
    # nc 类别数
    def __init__(self,c,c1,nc) -> None:
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
        self.hc5 = Conv(c1*16,c1*32,k=3,s=2,p=None,d=1,g=1,b=False)
        self.rc20 = ResidualConv(c1*32,c1*16)
        self.rc21 = ResidualConv(c1*32,c1*16)
        self.rc22 = ResidualConv(c1*32,c1*16)
        self.rc23 = ResidualConv(c1*32,c1*16)
        self.sc1 = SetConv(c1*32,c1*16)
        # 输出头1
        self.h1 = HeadConv(c1*16,c1*16,nc)
        self.cv2 = Conv(c1*16,c1*8,k=1,s=1,p=None,d=1,g=1,b=False)
        # 上采样 tc1 与 rc19进行拼接
        self.tc1 = TransposeConv(c1*8)
        self.sc2 = SetConv(c1*16+c1*8,c1*8)
        # 输出头2
        self.h2 = HeadConv(c1*8,c1*8,nc)
        self.cv3 = Conv(c1*8,c1*4,k=1,s=1,p=None,d=1,g=1,b=False)
        # 上采样 tc2 与 rc11进行拼接
        self.tc2 = TransposeConv(c1*4)
        self.sc3 = SetConv(c1*8+c1*4,c1*4)
        # 输出头3
        self.h3 = HeadConv(c1*4,c1*4,nc)

    def forward(self,x):
        cv1 = self.cv1(x)
        hc1 = self.hc1(cv1)
        rc1 = self.rc1(hc1)
        hc2 = self.hc2(rc1)
        rc2 = self.rc2(hc2)
        rc3 = self.rc3(rc2)
        hc3 = self.hc3(rc3)
        rc4 = self.rc4(hc3)
        rc5 = self.rc5(rc4)
        rc6 = self.rc6(rc5)
        rc7 = self.rc7(rc6)
        rc8 = self.rc8(rc7)
        rc9 = self.rc9(rc8)
        rc10 = self.rc10(rc9)
        rc11 = self.rc11(rc10)
        hc4 = self.hc4(rc11)
        rc12 = self.rc12(hc4)
        rc13 = self.rc13(rc12)
        rc14 = self.rc14(rc13)
        rc15 = self.rc15(rc14)
        rc16 = self.rc16(rc15)
        rc17 = self.rc17(rc16)
        rc18 = self.rc18(rc17)
        rc19 = self.rc19(rc18)
        hc5 = self.hc5(rc19)
        rc20 = self.rc20(hc5)
        rc21 = self.rc21(rc20)
        rc22 = self.rc22(rc21)
        rc23 = self.rc23(rc22)
        sc1 = self.sc1(rc23)
        # 输出头1
        h1 = self.h1(sc1)
        cv2 = self.cv2(sc1)
        # 上采样 tc1 与 rc19进行拼接
        tc1 = self.tc1(cv2)
        c1 = torch.cat((rc19,tc1),dim=1)
        sc2 = self.sc2(c1)
        # 输出头2
        h2 = self.h2(sc2)
        cv3 = self.cv3(sc2)
        # 上采样 tc2 与 rc11进行拼接
        tc2 = self.tc2(cv3)
        c2 = torch.cat((rc11,tc2),dim=1)
        sc3 = self.sc3(c2)
        # 输出头3
        h3 = self.h3(sc3)

        return h1,h2,h3
    
def main(args):
    c = args.c
    c1 = args.c1
    nc = args.nc
    size = args.size
    x = torch.randn((1,c,size,size))
    print(x.shape)
    net = yolov3(c,c1,nc)
    results = net.forward(x)
    print(results[0].shape)
    print(results[1].shape)
    print(results[2].shape)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--c',type=int,help='',default=3)
    parser.add_argument('--c1',type=int,help='',default=32)
    parser.add_argument('--nc',type=int,help='',default=80)
    parser.add_argument('--size',type=int,help='',default=416)
    return parser.parse_args(argv)

if __name__=='__main__':
    main(parse_arguments(sys.argv[1:]))