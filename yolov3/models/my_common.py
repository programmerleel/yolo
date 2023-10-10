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
        self.conv = nn.Conv2d(c1,c2,k,s,auto_pad(k,p),groups=g,dilation=d,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act,nn.Module) else nn.Identity()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self,x):
        return self.act(self.conv(x))