import argparse
import contextlib
from copy import deepcopy
from models.common import *
import os
from pathlib import Path
import platform
import sys
import yaml

class Model(nn.Module):
    def __init__(self,cfg,ch=3,nc=None,anchors=None):
        super().__init__()
        if isinstance(cfg ,dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f,Loader=yaml.SafeLoader)

        self.model,self.save = parse_model(deepcopy(self.yaml),ch = [ch])
    def forward(self, x):
        y = []  # outputs
        print(self.save)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                print(m.f)
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                # print(x[0].shape)
                print(x[1].shape)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x


def parse_model(d, ch):
    # 读取模型文件（字典类型）的相关参数
    anchors, nc = d['anchors'], d['nc']
    # 每一个predict head上的anchor数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # 每一个predict head层的输出个数
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # f(from): 当前层输入来自哪些层
    # n(number): 当前层数，
    # m(module): 当前层类别
    # args: 当前层类参数列表，包括channel、kernel_size、stride、padding和bias等
    # 遍历backbone和head的每一层
    for i, (f, n, m, args) in enumerate(d['backbone']+d['head']):  # from, number, module, args
        # 得到当前层的真实类名，例如：m = Focus -> <class 'models.common.Focus'>
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        if m in {Conv,ResidualBlock,FeatureBlock,UpBlock}:
            # c1: 当前层的输入channel数，设定为3; c2: 当前层的输出channel数(初定); ch: 记录着所有层的输出channel数
            c1, c2 = ch[f], args[0]
            # 在初始args的基础上更新，加入当前层的输入channel并更新当前层
            # [in_channels, out_channels, *args[1:]]
            args = [c1, c2, *args[1:]]
        elif m in {ConcatBlock}:
            c2 = sum([ch[x] for x in f])
        elif m in {HeadBlock}:
            c1,c2 = ch[f],no
            args = [c1, c2, *args[1:]]
        # m_: 得到当前层的module，如果n>1就创建多个m(当前层结构)，如果n=1就创建一个m
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        m_.i, m_.f = i, f
        # 把所有层结构中的from不是-1的值记下
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将当前层结构module加入layers中
        layers.append(m_)
        if i == 0:
            ch = []  # 去除输入channel[3]
        # 把当前层的输出channel数加入ch
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def main(args):
    cfg = args.cfg
    model = Model(cfg)
    x = torch.randn((1,3,416,416))
    model.forward(x)

def parser_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str,help='',default=r"D:\资料\yolov3\models\yolov3.yaml")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parser_arguments(sys.argv[1:]))
