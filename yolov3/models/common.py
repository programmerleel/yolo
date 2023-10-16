import torch
import torch.nn as nn

# 自动padding函数
def auto_pad(k,p=None):
    if p is None:
        p = k//2 if isinstance(k,int) else [x//2 for x in k]
    return p

class Conv(nn.Module):
    # 默认激活函数
    default_act = nn.SiLU()

    def __init__(self,c1,c2,k=1,s=1,p=None,g=1,d=1,act=True) -> None:
        super().__init__()
        self.cv = nn.Conv2d(c1,c2,k,s,auto_pad(k,p),groups=g,dilation=d,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act,nn.Module) else nn.Identity()

    def forward(self,x):
        return self.act(self.bn(self.cv(x)))

class ResidualBlock(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        c_ = c2//2
        self.cv1 = Conv(c1,c_,k=1,s=1,p=None,g=1,d=1,act=True)
        self.cv2 = Conv(c_,c2,k=3,s=1,p=None,g=1,d=1,act=True)

    def forward(self,x):
        return x + self.cv2(self.cv1(x))

class FeatureBlock(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        c_ = c1//2
        self.cv1 = Conv(c1,c_,k=1,s=1,p=None,g=1,d=1,act=True)
        self.cv2 = Conv(c_,c1,k=3,s=1,p=None,g=1,d=1,act=True)
        self.cv3 = Conv(c1, c_, k=1, s=1, p=None, g=1, d=1, act=True)
        self.cv4 = Conv(c_, c1, k=3, s=1, p=None, g=1, d=1, act=True)
        self.cv5 = Conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)

    def forward(self,x):
        return self.cv5(self.cv4(self.cv3(self.cv2(self.cv1(x)))))

class UpSampleConv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self,c1,c2,k=2,s=2,p=0,op=0,act = True):
        super().__init__()
        self.tcv = nn.ConvTranspose2d(c1,c2,k,s,p,op)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act,nn.Module) else nn.Identity()

    def forward(self,x):
        self.act(self.bn(self.tcv(x)))

class UpBlock(nn.Module):
    def __init__(self,c1,c2,k=2,s=2,p=0,op=0,act=True):
        super().__init__()
        c_ = c1//2
        self.cv = Conv(c1,c_)
        self.up = UpSampleConv(c_,c2,k,s,p,op,act)

    def forward(self,x):
        return self.up(self.cv(x))

class ConcatBlock(nn.Module):
    def __init__(self,d = 1):
        super().__init__()
        self.d = d

    def forward(self,x):
        return torch.cat(x,self.d)

class HeadBlock(nn.Module):
    def __init__(self,c1,c2):
        super().__init__()
        c_ = c1
        self.cv1 = Conv(c1,c_,k=3,s=1,p=None,g=1,d=1,act=True)
        self.cv2 = nn.Conv2d(c_,c2,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        return self.cv2(self.cv1(x))