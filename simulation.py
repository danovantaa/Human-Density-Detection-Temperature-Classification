import streamlit as st
import torch 
import torch.nn as nn
import math 
import numpy as np 
from torch.utils.data import DataLoader, Dataset
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import cv2
from types import SimpleNamespace
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class Conv(nn.Module): 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False) 
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    =  nn.SiLU(inplace=True) if act else nn.Identity()
        
    def forward (self, x):
        return self.act(self.bn(self.conv(x)))

class ConvPE(nn.Module): 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d) 
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    =  nn.SiLU(inplace=True) if act else nn.Identity()
        
    def forward (self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module) : 
    def __init__(self, c1, c2 ,shortcut=True, k=3, e=0.5):
        super().__init__()
        c_ = int(e*c2)  
        
        self.cv1 = Conv(c1, c_, k, 1) 
        self.cv2 = Conv(c_, c2, k, 1) 
        self.add = shortcut and c1 == c2  # Residual connection jika c1 == c2
        
    def forward (self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  

class C2f(nn.Module): 
    def __init__(self, c1, c2, n=2, shortcut=False,g=1, e=0.5):
        super().__init__()
        self.c = int(c2*e)

        self.cv1 = Conv(c1, 2*self.c, 1, 1)

        self.cv2 = Conv((2+n)*self.c, c2, 1)

        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, k=3, e = 1.0) 
            for _ in range(n)
            )
        
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        
        #bottleneck  berantai
        y.extend(m(y[-1]) for m in self.m)
        
        # Concate y =>((2+n)c, H, W)
        return self.cv2(torch.cat(y, dim=1))
        
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, s=1):
        super().__init__()
        c_ = c1//2
        
        self.cv1 =Conv(c1, c_,1,1 ) 
        self.cv2 = Conv(c_*4, c2, 1,1) 
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k//2)
        
    def forward(self, x):
        # Chanel Reduction
        x = self.cv1(x)
        
        # Pyramid Pooling
        max1 = self.m(x)
        max2 = self.m(max1)
        max3 = self.m(max2) 
        
        #concat : [x, max1, max2, max3]
        return self.cv2(
            torch.cat((x, max1, max2, max3),1)
        )

class C3K2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, expand=False, reduce=True):
        super().__init__()
        c_ = c2 // 2

        if expand:
            self.cv1 = Conv(c1, c1, 1, 1)  # backbone
        else:
            if reduce:
                self.cv1 = Conv(c1, c2, 1, 1)  # td1
            else:
                self.cv1 = Conv(c1, c1, 1, 1)  # td2 FIX

        out_c = c2 * 2 if expand else c2
        self.cv2 = Conv((2 + n) * c_, out_c, 1)

        self.m = nn.ModuleList(
            C3k(c_, c_, 2, shortcut)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.qkv = Conv(c, c * 2, 1, act=False)   # 256→512
        self.proj = Conv(c, c, 1, act=False)
        self.pe = Conv(c, c, 3, 1, g=c, act=False)

    def forward(self, x):
        qk = self.qkv(x)
        q, k = qk.chunk(2, 1)

        attn = torch.sigmoid(q * k)  # lightweight attention
        x = x + self.pe(x * attn)

        return self.proj(x)
    
class FFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cv1 = Conv(c, c * 2, 1)
        self.cv2 = Conv(c * 2, c, 1, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class PSABlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = Attention(c)
        self.ffn = FFN(c)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class C2PSA(nn.Module):
    def __init__(self, c, n=1):
        super().__init__()
        self.cv1 = Conv(c, c, 1)
        self.cv2 = Conv(c, c, 1)

        c_ = c // 2
        self.m = nn.Sequential(*[PSABlock(c_) for _ in range(n)])

    def forward(self, x):
        x = self.cv1(x)
        x1, x2 = x.chunk(2, 1)
        x1 = self.m(x1)
        x = torch.cat((x1, x2), 1)
        return self.cv2(x)

class Attention12(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.qkv = Conv(c, c * 3, 1, act=False)

        self.proj = Conv(c, c, 1, act=False)

        self.pe = ConvPE(c, c, 7, 1, g=c, act=False)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, 1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(-1)

        x = attn @ v
        x = self.proj(x)
        x = x + self.pe(v)
        return x

class ABlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.attn = Attention12(c)
        hidden = int(c * 1.2)

        self.mlp = nn.Sequential(
            Conv(c, hidden, 1),
            Conv(hidden, c, 1, act=False)
        )

    def forward(self, x):

        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x

class C3k(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, g=1):
        super().__init__()
        c_ = c2 // 2

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, k=3, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class A2C2f(nn.Module):
    def __init__(self, c1, c2, n=2, block_type="attn"):
        super().__init__()
        c_ = c2 // 2

        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv((n + 1) * c_, c2, 1)

        if block_type == "attn":
            self.m = nn.ModuleList(
                nn.Sequential(ABlock(c_), ABlock(c_)) for _ in range(n)
            )
            self.gamma = nn.Parameter(torch.ones(c2))
            
        elif block_type == "c3k":
            self.m = nn.ModuleList(
                C3k(c_, c_) for _ in range(n)
            )
            self.gamma = None

        self.add = c1 == c2

    def forward(self, x):
        shortcut = x
        x = self.cv1(x)
        y = [x]
        
        for m in self.m:
            x = m(x)
            y.append(x)
        out = self.cv2(torch.cat(y, 1))

        if self.add:
            out = shortcut + self.gamma.view(1, -1, 1, 1) * out
        return out

class C3K2V12(nn.Module):

    def __init__(self, c1, c2, n=2):
        super().__init__()
        c_ = c1 // 2

        self.cv1 = Conv(c1, c1, 1)   
        self.cv2 = Conv((2 + n) * c_, c2, 1)

        self.m = nn.ModuleList(
            [C3k(c_, c_, 2) for _ in range(n)]
        )

    def forward(self, x):
        x = self.cv1(x)
        x1, x2 = x.chunk(2,1)
        y = [x1, x2]

        for m in self.m:
            x2 = m(x2)
            y.append(x2)

        return self.cv2(torch.cat(y,1))

class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)    

class DFL(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        self.conv.weight.data[:] = torch.arange(reg_max).view(1, reg_max, 1, 1)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, 4, self.reg_max, H, W)
        x = x.softmax(2)
        x = x.view(B * 4, self.reg_max, H, W)
        x = self.conv(x)
        return x.view(B, 4, H, W)
    
class C3K2_Neck(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()

        c_ = c2 // 2  

        self.cv1 = Conv(c1, c2, 1) 
        self.cv2 = Conv((2 + n) * c_, c2, 1)

        self.m = nn.ModuleList(
            C3k(c_, c_) for _ in range(n)
        )

    def forward(self, x):
        x = self.cv1(x)

        x1, x2 = x.chunk(2, 1)

        y = [x1, x2]

        for m in self.m:
            x2 = m(x2)
            y.append(x2)

        return self.cv2(torch.cat(y, 1))

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.d = dim

    def forward(self, x):
        return torch.cat(x, self.d)
    
SCALES8 = {
    "n": (0.33, 0.25, 1024),
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
} # Baseline YOLO

def make_divisible(x, divisor=8): #memproses data dengan paralel dengan Vector SIMD 
    return math.ceil(x / divisor) * divisor

def scale_channel(c, width_mul, max_channels):
    return make_divisible(min(c, max_channels) * width_mul, 8) # return {[math.ceil (c * 1.25)] / 8 }

class YOLOv8Backbone(nn.Module):
    def __init__(self, scale="x"):
        super().__init__()
        depth_mul, width_mul, max_channels = SCALES8[scale]

        base_c = [64, 128, 256, 512, 1024]
        base_n = [3, 6, 6, 3]
        c1, c2, c3, c4, c5 = [
            scale_channel(c, width_mul, max_channels) for c in base_c
        ]
        n2, n3, n4, n5 = [
            max(round(n * depth_mul), 1) for n in base_n
        ]
        self.out_channels = (c3, c4, c5)
        
        self.stem = Conv(3, c1, 3, 2)
        
        self.dark2 = nn.Sequential(
            Conv(c1, c2, 3, 2),
            C2f(c2, c2, n=n2, shortcut=True)
        )
        self.dark3 = nn.Sequential(
            Conv(c2, c3, 3, 2),
            C2f(c3, c3, n=n3, shortcut=True)
        )
        self.dark4 = nn.Sequential(
            Conv(c3, c4, 3, 2),
            C2f(c4, c4, n=n4, shortcut=True)
        )
        self.dark5 = nn.Sequential(
            Conv(c4, c5, 3, 2),
            C2f(c5, c5, n=n5, shortcut=True),
            SPPF(c5, c5, 5)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        p3 = x
        x = self.dark4(x)
        p4 = x
        x = self.dark5(x)
        p5 = x
        return p3, p4, p5 
    
class YOLOv8Neck(nn.Module):
    def __init__(self, ch, scale="x"):
        super().__init__()
        depth_mul, _, _ = SCALES8[scale]
        c3, c4, c5 = ch 
        
        n = max(round(3 * depth_mul), 1)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_td1 = C2f(c5 + c4, c4, n=n, shortcut=False)
        self.c2f_td2 = C2f(c4 + c3, c3, n=n, shortcut=False)
        self.down1 = Conv(c3, c3, 3, 2)
        self.c2f_bu1 = C2f(c3 + c4, c4, n=n, shortcut=False)
        self.down2 = Conv(c4, c4, 3, 2)
        self.c2f_bu2 = C2f(c4 + c5, c5, n=n, shortcut=False)

    def forward(self, p3, p4, p5):
        p5_up = self.upsample(p5)
        p4_td = self.c2f_td1(torch.cat([p5_up, p4], dim=1))
        p4_up = self.upsample(p4_td)
        p3_out = self.c2f_td2(torch.cat([p4_up, p3], dim=1))
        p3_down = self.down1(p3_out)
        p4_out = self.c2f_bu1(torch.cat([p3_down, p4_td], dim=1))
        p4_down = self.down2(p4_out)
        p5_out = self.c2f_bu2(torch.cat([p4_down, p5], dim=1))
        return p3_out, p4_out, p5_out 
    
class YOLOv8Head(nn.Module):
    def __init__(self, ch, nc=3, reg_max=16):
        super().__init__()
        # semakin besar reg_max → prediksi bbox bisa lebih halus
        # semakin besar reg_max → channel output lebih banyak, komputasi lebih berat
        self.nc = nc
        self.reg_max = reg_max 

        self.cv2 = nn.ModuleList()  # regression
        self.cv3 = nn.ModuleList()  # classification

        c2 = max(16, ch[0] // 4, reg_max * 4) 
        c3 = max(ch[0], min(nc, 100)) 

        for c in ch:
            # regression branch
            self.cv2.append(
                nn.Sequential(
                    Conv(c, c2, 3), 
                    Conv(c2, c2, 3),  
                    nn.Conv2d(c2, 4 * reg_max, 1) 
                )
            )
            # classification branch
            self.cv3.append(
                nn.Sequential(
                    Conv(c, c3, 3),
                    Conv(c3, c3, 3), 
                    nn.Conv2d(c3, nc, 1)
                )
            )

    def forward(self, x):
        outputs = []

        for i in range(len(x)):
            reg = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])

            outputs.append(torch.cat([reg, cls], 1))

        return outputs 
    
class TemperatureClsHead(nn.Module):
    def __init__(self, in_channels, num_temp_classes=3, hidden_channels=320, dropout=0.1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )    
        self.pool = nn.AdaptiveAvgPool2d(20)
        self.drop = nn.Dropout(p=dropout, inplace=True)
        self.linear = nn.Linear(hidden_channels*20*20, num_temp_classes)

    def forward(self, x):
        x = self.conv(x) 
        x = self.pool(x)     
        x = torch.flatten(x, 1) 
        x = self.drop(x) 
        x = self.linear(x)   
        return x
    
class YOLOv8(nn.Module):
    def __init__(self, scale="x", nc=3, reg_max=16, num_temp_classes=3):
        super().__init__()

        self.nc = nc 
        self.reg_max = reg_max 
        self.no = nc + reg_max * 4 
        self.stride = torch.tensor([8.,16.,32.])
        
        self.backbone = YOLOv8Backbone(scale) 
        ch = self.backbone.out_channels  

        self.neck = YOLOv8Neck(ch, scale) 
        self.head = YOLOv8Head(ch, nc, reg_max)

        # Mengacu pada YOLOv8-CLS
        self.temp_head = TemperatureClsHead(
            in_channels=ch[1],  # 640   FEATURE KEDUA
            num_temp_classes=num_temp_classes, # 3
            hidden_channels=320,
            dropout=0.1
        )

    def forward(self, x):
        p3, p4, p5 = self.backbone(x) 
        p3, p4, p5 = self.neck(p3, p4, p5) 

        det_out = self.head([p3, p4, p5])

        temp_logits = self.temp_head(p4) 

        return det_out, temp_logits

SCALES = {
    "n": (0.50, 0.25),
    "s": (0.50, 0.50),
    "m": (0.50, 1.00),
    "l": (1.00, 1.00),
    "x": (1.00, 1.50),
}

class YOLOv11Backbone(nn.Module):
    def __init__(self, scale="x"):
        super().__init__()

        depth_mul, width_mul = SCALES[scale]

        base_c = [64,128,256,512]
        base_n = [2,2,2,2]

        c1,c2,c3,c4 = [make_divisible(c*width_mul) for c in base_c]
        n2,n3,n4,n5 = [max(round(n*depth_mul),1) for n in base_n]

        # stem
        self.stem = Conv(3,c1,3,2)

        # stage2
        self.dark2 = nn.Sequential(
            Conv(c1,c2,3,2),
            C3K2(c2,c2,n=n2,shortcut=False, expand=True)
        )

        self.dark3 = nn.Sequential(
            Conv(c2*2,c3,3,2),  
            C3K2(c3,c3,n=n3,shortcut=False, expand=True)
        )

        self.dark4 = nn.Sequential(
            Conv(c4,c4,3,2),  
            C3K2(c4,c4,n=n4,shortcut=True)
        )

        self.dark5 = nn.Sequential(
            Conv(c4,c4,3,2),
            C3K2(c4,c4,n=n5,shortcut=True),
            SPPF(c4,c4),
            C2PSA(c4,n=2)
        )
        self.out_channels = (2*c3,c4,c4)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)

        return p3, p4, p5
    
class YOLOv11Neck(nn.Module):
    def __init__(self, ch, scale="x"):
        super().__init__()
        depth_mul, _ = SCALES[scale]

        p3_in, p4_in, p5_in = ch
        n = max(round(2 * depth_mul), 1)

        c3_out = p3_in // 2
        c4_out = p4_in
        c5_out = p5_in
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.td1 = C3K2(p5_in + p4_in, c4_out, n=n, shortcut=False)
        self.td2 = C3K2(c4_out + p3_in, c3_out, n=n, shortcut=False)
        self.down1 = Conv(c3_out, c3_out, 3, 2)
        self.bu1 = C3K2(c3_out + c4_out, c4_out, n=n, shortcut=False)
        self.down2 = Conv(c4_out, c4_out, 3, 2)
        self.bu2 = C3K2(c4_out + c5_out, c5_out, n=n, shortcut=False)

        self.out_channels = (c3_out, c4_out, c5_out)

    def forward(self, p3, p4, p5):
        p5_up = self.upsample(p5)
        p4_td = self.td1(torch.cat([p5_up, p4], 1))
        
        p4_up = self.upsample(p4_td)
        p3_out = self.td2(torch.cat([p4_up, p3], 1))

        p3_down = self.down1(p3_out)
        p4_out = self.bu1(torch.cat([p3_down, p4_td], 1))

        p4_down = self.down2(p4_out)
        p5_out = self.bu2(torch.cat([p4_down, p5], 1))

        return p3_out, p4_out, p5_out
    
class YOLOv11Head(nn.Module):
    def __init__(self, ch, nc=3, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max

        c2 = max(16, ch[0] // 4, reg_max * 4)   # reg branch hidden
        c3 = max(ch[0], min(nc, 100))           # cls branch hidden

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * reg_max, 1)
            ) for x in ch
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(
                    DWConv(x, x, 3),
                    Conv(x, c3, 1),
                ),
                nn.Sequential(
                    DWConv(c3, c3, 3),
                    Conv(c3, c3, 1),
                ),
                nn.Conv2d(c3, nc, 1)
            ) for x in ch
        )
        self.dfl = DFL(reg_max)

    def forward(self, feats):
        out = []
        for i, x in enumerate(feats):
            reg = self.cv2[i](x)
            cls = self.cv3[i](x)
            out.append(torch.cat([reg, cls], 1))
        return out

class YOLOv11(nn.Module):
    def __init__(self, scale="x", nc=3, reg_max=16, num_temp_classes=3):
        super().__init__()

        self.num_classes = nc
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max
        self.stride = torch.tensor([8., 16., 32.])

        self.backbone = YOLOv11Backbone(scale)
        self.neck = YOLOv11Neck(self.backbone.out_channels, scale)
        self.head = YOLOv11Head(self.neck.out_channels, nc, reg_max)

        self.temp_head = TemperatureClsHead(
            in_channels=self.neck.out_channels[1],  # 640   FEATURE KEDUA
            num_temp_classes=num_temp_classes, # 3
            hidden_channels=320,
            dropout=0.1
        )

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)

        det_out = self.head([p3, p4, p5])

        temp_logits = self.temp_head(p4) 

        return det_out, temp_logits

class YOLOv12Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_c = [96, 192, 384, 768]
        c1, c2, c3, c4 = base_c

        self.stem = Conv(3, c1, 3, 2)

        self.dark2 = nn.Sequential(
            Conv(c1, c2, 3, 2),
            C3K2V12(c2, c3, n=2)
        )
        self.dark3 = nn.Sequential(
            Conv(c3, c3, 3, 2),
            C3K2V12(c3, c4, n=2)
        )
        self.dark4 = nn.Sequential(
            Conv(c4, c4, 3, 2),
            A2C2f(c4, c4, n=4, block_type="attn")
        )
        self.dark5 = nn.Sequential(
            Conv(c4, c4, 3, 2),
            A2C2f(c4, c4, n=4, block_type="attn")
        )

        self.out_channels = (c4, c4, c4)

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        p3 = x   

        x = self.dark4(x)
        p4 = x   

        x = self.dark5(x)
        p5 = x   
        return p3, p4, p5

class YOLOv12Neck(nn.Module):
    def __init__(self, ch=(768, 768, 768)):
        super().__init__()
        p3_ch, p4_ch, p5_ch = ch

        self.up = nn.Upsample(scale_factor=2.0, mode="nearest")

        self.fpn1 = A2C2f(
            p5_ch + p4_ch,  
            p4_ch,           
            n=2,
            block_type="c3k"
        )
        self.fpn2 = A2C2f(
            p4_ch + p3_ch,   
            384,           
            n=2,
            block_type="c3k"
        )
        self.down1 = Conv(384, 384, 3, 2)

        self.pan1 = A2C2f(
            384 + p4_ch,   
            p4_ch,           
            n=2,
            block_type="c3k"
        )
        self.down2 = Conv(p4_ch, p5_ch, 3, 2)

        self.pan2 = C3K2_Neck(
            p5_ch + p5_ch,   
            p5_ch,          
            n=2
        )
        self.out_channels = (384, 768, 768)

    def forward(self, p3, p4, p5):
        x = self.up(p5)
        x = torch.cat([x, p4], dim=1)
        n4 = self.fpn1(x)

        x = self.up(n4)
        x = torch.cat([x, p3], dim=1)
        n3 = self.fpn2(x)

        x = self.down1(n3)
        x = torch.cat([x, n4], dim=1)
        n4 = self.pan1(x)

        x = self.down2(n4)
        x = torch.cat([x, p5], dim=1)
        n5 = self.pan2(x)

        return n3, n4, n5

class YOLOv12Head(nn.Module):
    def __init__(self, ch, nc=80, reg_max=16):
        super().__init__()

        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max

        self.cv2 = nn.ModuleList()  
        self.cv3 = nn.ModuleList()  

        for i, c in enumerate(ch):
            c2 = min(c, 96)

            self.cv2.append(
                nn.Sequential(
                    Conv(c, c2, 3),
                    Conv(c2, c2, 3),
                    nn.Conv2d(c2, 4 * reg_max, 1)
                )
            )

            if i == 0:
                c_mid = c
            else:
                c_mid = c // 2

            self.cv3.append(
                nn.Sequential(
                    nn.Sequential(
                        DWConv(c, c, 3),
                        Conv(c, c_mid, 1, act=False)
                    ),
                    nn.Sequential(
                        DWConv(c_mid, c_mid, 3),
                        Conv(c_mid, c_mid, 1, act=False)
                    ),
                    nn.Conv2d(c_mid, nc, 1)
                )
            )
        self.dfl = DFL(reg_max)

    def forward(self, feats):
        outputs = []
        for i, x in enumerate(feats):
            reg = self.cv2[i](x)
            cls = self.cv3[i](x)

            out = torch.cat([reg, cls], 1)
            outputs.append(out)
        return outputs
    
class YOLOv12(nn.Module):
    def __init__(self, nc=3, reg_max=16, num_temp_classes=3):
        super().__init__()
        
        self.num_classes = nc
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.tensor([8., 16., 32.])

        self.backbone = YOLOv12Backbone()
        self.neck = YOLOv12Neck(self.backbone.out_channels)
        self.head = YOLOv12Head(self.neck.out_channels,nc)
        
        self.temp_head = TemperatureClsHead(
            in_channels=self.neck.out_channels[1],  # 640   FEATURE KEDUA
            num_temp_classes=num_temp_classes, # 3
            hidden_channels=320,
            dropout=0.1
        )

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)

        det_out = self.head([p3, p4, p5])

        temp_logits = self.temp_head(p4)

        return det_out, temp_logits   

def make_anchors(feats, strides, grid_cell_offset=0.5): 
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    
    # Loop tiap feature map YOLO
    for i in range(len(feats)): 
        stride = strides[i] # 8, 16, 32
        h, w = feats[i].shape[2:]  # [80,80], [40, 40], [20, 20]  => [80, 40, 20], [80, 40, 20]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # [0.5, 1.5, 2.5 ... len(w[feats])]
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # [0.5, 1.5, 2.5 ... len(h[feats])]
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") # 1D => 2D
        
        anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2)) # 2D -> 3D :  [80, 80, 2] => [6400, 2] 
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device)) # isi nilai stride_tensor jadi stride sebanyak w*x 
        
    anchor_points = torch.cat(anchor_points, dim=0) # [8400, 2]
    stride_tensor = torch.cat(stride_tensor, dim=0) # [8400, 1]
        
    return anchor_points, stride_tensor

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim) 
    x1y1 = anchor_points - lt 
    x2y2 = anchor_points + rb 
    if xywh:
        c_xy = (x1y1 + x2y2) / 2 # Center Point (titik tengah dari bbox hasil prediksi)
        wh = x2y2 - x1y1 
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

class v8DetectionLoss(nn.Module):
    def __init__(self, model: torch.nn.Module, tal_topk: int = 10):
        super().__init__()
        device = next(model.parameters()).device

        # Bobot Hyperparameter loss YOLO    
        h = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5
        )

        self.bce = nn.BCEWithLogitsLoss(reduction="none") # Classification Loss
        self.hyp = h # Hyperparameter loss

        # Ambil atribut dari model custom
        self.stride = model.stride.to(device) if isinstance(model.stride, torch.Tensor) else torch.tensor(model.stride, device=device)
        self.nc = model.nc # 3
        self.reg_max = model.reg_max # jumlah bin distribus DFL
        self.no = self.nc + self.reg_max * 4 # total chanel output pada head (3 + 16*4) = 67
        self.device = device 
        
        # mengubah distribusi probabilitas menjadi jarak kontinu.
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device) # [0-15]

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        b, a, c = pred_dist.shape  # [B, A, 64]
        
        #[B, A, 64] => [B, A, 4, 16] logits => [B, A, 4, 16] prob 
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        
        # [B, A, 4, 16] distribusi => [B, A, 4]nilai jarak menggunakan expected value 
        pred_dist = pred_dist.matmul(self.proj.type(pred_dist.dtype))
        
        pred_bboxes = dist2bbox(pred_dist, anchor_points, xywh=False)
        return pred_bboxes
    
    def parse_output(self, preds): 
        feats = preds # List Feature Map 
        bs = feats[0].shape[0] # batch size
    
        pred_distri, pred_scores = torch.cat(
             # [B,no,H,W] -> [B,no,H*W]
            [xi.view(bs, self.no, -1) 
             for xi in feats
             ],
            dim=2 # Gabung semua anchor dari p3, p4, p5
        ).split(
            (
                self.reg_max * 4, # distribusi DFL bbox
                self.nc     # class logits
            ), 
            dim=1
        )
    
        return {
            "boxes": pred_distri, # prediksi regresi bbox sebelum decode. [B, 4*reg_max, A]
            "scores": pred_scores, # logit class  [B, nc, A]
            "feats": feats # List feature map asli.
        }

    def forward(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        return self.loss(self.parse_output(preds), batch)

class YOLOvPredictor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.device = next(model.parameters()).device

        self.nc = model.nc
        self.reg_max = model.reg_max
        self.no = model.no
        self.stride = model.stride.to(self.device)

        self.decoder = v8DetectionLoss(model)
    
    @torch.no_grad()
    def forward(self, preds):

        parsed = self.decoder.parse_output(preds)
        pred_dist = parsed["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = parsed["scores"].permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(
            parsed["feats"],
            self.stride,
            0.5,
        )

        pred_boxes = self.decoder.bbox_decode(
            anchor_points,
            pred_dist,
        )

        pred_boxes *= stride_tensor

        return {
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores.sigmoid(),
        }

class Metric:
    def __init__(self):
        self.p = np.array([])
        self.r = np.array([])
        self.f1 = np.array([])
        self.all_ap = np.array([])
        self.ap_class_index = np.array([])
        self.nc = 0

    @property
    def mp(self):
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def update(self, results):
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            *_
        ) = results
        
def match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
) -> torch.Tensor:

    correct = np.zeros(
        (pred_classes.shape[0], iouv.shape[0]),
        dtype=bool,
    )
    true_classes = true_classes.to(device)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        func = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        ap = func(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16,
) -> tuple:
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections
    x, prec_values = np.linspace(0, 1, 1000), []
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values

def non_max_suppression(
    pred_boxes,
    pred_scores,
    conf_thres=0.25,
    iou_thres=0.6,
    max_det=300,
):
    outputs = []
    bs = pred_boxes.shape[0]
    max_wh = 7680

    for b in range(bs):

        boxes = pred_boxes[b]
        scores = pred_scores[b]
        conf, cls = scores.max(1)
        keep = conf > conf_thres
        boxes = boxes[keep]
        conf = conf[keep]
        cls = cls[keep]

        if len(boxes) == 0:
            outputs.append({
                "boxes": boxes,
                "scores": conf,
                "labels": cls,
            })
            continue

        offsets = cls.float().unsqueeze(1) * max_wh
        keep_idx = torchvision.ops.nms(
            boxes + offsets,
            conf,
            iou_thres,
        )
        keep_idx = keep_idx[:max_det]
        outputs.append(
            {
                "boxes": boxes[keep_idx],
                "scores": conf[keep_idx],
                "labels": cls[keep_idx],
            }
        )

    return outputs

class DetectionEvaluator:
    def __init__(self, names):

        self.nc = len(names)
        # IoU threshold
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = len(self.iouv)
        self.metric = Metric()
        self.reset()

    def reset(self):
        self.stats = {
            "tp": [],
            "conf": [],
            "pred_cls": [],
            "target_cls": []
        }

    def prepare_gt_boxes(self, gt_boxes, image_shape):
        scale = gt_boxes.new_tensor(
            [image_shape[1], image_shape[0],
            image_shape[1], image_shape[0]]
        )
        gt_boxes = torchvision.ops.box_convert(
            gt_boxes,
            "cxcywh",
            "xyxy"
        )

        return gt_boxes * scale
    
    def append_stats(
        self,
        tp,
        conf,
        pred_cls,
        target_cls,
    ):
        self.stats["tp"].append(tp)
        self.stats["conf"].append(conf)
        self.stats["pred_cls"].append(pred_cls)
        self.stats["target_cls"].append(target_cls)

    @torch.no_grad()
    def update(self, preds, targets, image_shape):
        bs = len(preds)
        for si in range(bs):
            pred = preds[si]
            # Ground truth dari dataset
            gt = targets[si]
            if len(gt):
                gt_cls = gt[:, 0].long()
                gt_boxes = gt[:, 1:]

                gt_boxes = self.prepare_gt_boxes(
                    gt_boxes,
                    image_shape
                )

            else:
                gt_cls = torch.tensor([], dtype=torch.long)
                gt_boxes = torch.empty((0, 4))

            # Prediction hasil NMS
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_cls = pred["labels"]
            tp = np.zeros((0, self.niou), dtype=bool)

            if len(pred_boxes) == 0:
                self.append_stats(
                    tp,
                    pred_scores.cpu().numpy(),
                    pred_cls.cpu().numpy(),
                    gt_cls.cpu().numpy()
                )
                continue
            gt_boxes = gt_boxes.to(device)
            pred_boxes = pred_boxes.to(device)

            if len(gt_boxes):
                iou = torchvision.ops.box_iou(
                    gt_boxes,
                    pred_boxes
                )
                tp = match_predictions(
                    pred_cls,
                    gt_cls,
                    iou,
                    self.iouv
                ).cpu().numpy()

            # Tidak ada ground truth
            else:
                tp = np.zeros(
                    (len(pred_boxes), self.niou),
                    dtype=bool
                )
                
            self.append_stats(
                np.asarray(tp),
                pred_scores.cpu().numpy(),
                pred_cls.cpu().numpy(),
                gt_cls.cpu().numpy()
            )

    def compute(self):
        stats = {}
        for k, v in self.stats.items():
            if len(v):
                stats[k] = np.concatenate(v, axis=0)
            else:
                stats[k] = np.array([])

        if len(stats["tp"]) == 0:
            return {
                "precision": 0,
                "recall": 0,
                "map50": 0,
                "map50_95": 0,
            }

        results = ap_per_class(
            tp=stats["tp"],
            conf=stats["conf"],
            pred_cls=stats["pred_cls"],
            target_cls=stats["target_cls"],
        )[2:]

        self.metric.nc = self.nc
        self.metric.update(results)

        return {
            "precision": float(self.metric.mp),
            "recall": float(self.metric.mr),
            "map50": float(self.metric.map50),
            "map50_95": float(self.metric.map),
        }
        
CLASS_NAMES = ["arm", "head", "upper_body"]

COLORS = {
    0: (255, 165, 0),  
    1: (50, 205, 50),    # Lime Green
    2: (138, 43, 226),   # Ungu
}

def draw_boxes(image, boxes, labels, class_names, colors, scores=None):

    img = image.copy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cls = int(labels[i])

        # Ambil warna sesuai kelas
        color = colors.get(cls, (255, 255, 255))

        text = class_names[cls]

        if scores is not None:
            text += f" {float(scores[i]):.2f}"

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            2
        )
        (text_w, text_h), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(
            img,
            (x1, max(0, y1 - text_h - 8)),
            (x1 + text_w + 4, y1),
            color,
            -1
        )

        cv2.putText(
            img,
            text,
            (x1 + 2, max(text_h, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    return img

def evaluate_temperature_classification(all_preds, all_labels):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    results = {
        "accuracy": accuracy_score(all_labels, all_preds),

        "precision_macro": precision_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0 
        ),

        "f1_macro": f1_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        ),
    }

    return results

MODEL_PATHS = {
    "Best YOLOv8": "YOLOV8X SGD LR 0.001 BS 16/best_model_SGD_lr0.001_bs16.pth",
    "Best YOLOv11": "YOLOV11X SGD LR 0.001 BS 16/best_model_SGD_lr0.001_bs16.pth",
    "Best YOLOv12": "YOLOV12X SGD LR 0.001 BS 32/best_model_SGD_lr0.001_bs32.pth"
}
csv_path = "Dataset_rapih/Informasi Dataset - TESTING.csv"
df_test = pd.read_csv(csv_path)

def load_yolo_label(path):

    if not os.path.exists(path):
        return torch.empty((0,5), dtype=torch.float32)

    gt = []

    with open(path) as f:
        for line in f:
            gt.append(list(map(float, line.split())))

    return torch.tensor(gt, dtype=torch.float32)

st.sidebar.title("Control")
st.title("Object Detection Simulation")
st.write("Simulation of human body part detection and temperature classification using YOLO-based models.")
selected_model = st.sidebar.selectbox(
    "Pilih Model",
    list(MODEL_PATHS.keys())
)

conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,      
    step=0.01
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

@st.cache_resource

def load_model(model_name):

    if model_name == "Best YOLOv8":
        model = YOLOv8("x", nc=3)

    elif model_name == "Best YOLOv11":
        model = YOLOv11("x", nc=3)
        model = torch.nn.DataParallel(model)

    elif model_name == "Best YOLOv12":
        model = YOLOv12(nc=3)
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(
        MODEL_PATHS[model_name],
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    predictor = YOLOvPredictor(
    model.module if isinstance(model, torch.nn.DataParallel) else model
    )
    return model, predictor

transform = transforms.Compose([
    transforms.ToTensor(),
])

temp_map = {
    "Dingin":0,
    "Sedang":1,
    "Sejuk":2
}

TEMP_CLASSES = ["Dingin", "Sedang", "Sejuk"]

if uploaded_file:
    filename = uploaded_file.name
    model, predictor = load_model(selected_model)
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    label_path = os.path.join(
        "Dataset_rapih",
        "labels",
        "test",
        os.path.splitext(filename)[0] + ".txt"
    )
    gt = load_yolo_label(label_path)
    tensor = transform(image).unsqueeze(0).to(device)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np)
        
    with torch.no_grad():
        det_out, temp_logits = model(tensor)

        decoded = predictor(det_out)

        preds = non_max_suppression(
            decoded["pred_boxes"],
            decoded["pred_scores"],
            conf_thres=conf_thres,
            iou_thres=0.6
        )
        prediction = preds[0]

        boxes = prediction["boxes"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()

        result = draw_boxes(
            image_np,
            boxes,
            labels,
            CLASS_NAMES,
            COLORS,
            scores=scores
        )

    with col2:
        st.subheader("Detection")
        st.image(result)
        
    evaluator = DetectionEvaluator(CLASS_NAMES)
    evaluator.reset()

    evaluator.update(
        preds,
        [gt],
        image_shape=(640,640)
    )

    metric = evaluator.compute()

    st.subheader("Model Performance")

    row = df_test[df_test["NamaFile"] == filename]
    gt_temp = temp_map[row.iloc[0]["Label"]]
    prob = torch.softmax(temp_logits, dim=1)
    pred = torch.argmax(prob, dim=1).item()
    pred_conf = prob[0, pred].item()
    pred = torch.argmax(temp_logits,1).item()
    col1, col2 = st.columns(2)
            
    rows = []

    for box, label, score in zip(boxes, labels, scores):

        x1, y1, x2, y2 = map(int, box)

        rows.append({
            "Object": CLASS_NAMES[int(label)],
            "Confidence": round(float(score), 3),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    df = pd.DataFrame(rows)
    
    st.subheader("Detection Result")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Ground Truth Temperature",
            TEMP_CLASSES[gt_temp]
        )

    with col2:
        st.metric(
            "Predicted Temperature",
            TEMP_CLASSES[pred],
            help=f"Confidence: {pred_conf:.2%}"
        )