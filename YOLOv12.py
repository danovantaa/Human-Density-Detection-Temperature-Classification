import torch 
import torch.nn as nn
import math
from ultralytics import YOLO


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
        self.add = shortcut and c1 == c2    
        
    def forward (self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  


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

class Attention12(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.qkv = Conv(c, c * 3, 1, act=False)

        self.proj = Conv(c, c, 1, act=False)
        #self.pe = nn.Conv2d(c,c,7,1,3,groups=c)
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
    
class YOLOv12Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        base_c = [96, 192, 384, 768, 768]
        c1, c2, c3, c4, c5 = base_c

        # layer 0
        self.stem = Conv(3, c1, 3, 2)

        # layer 1-2
        self.dark2 = nn.Sequential(
            Conv(c1, c2, 3, 2),
            C3K2V12(c2, c3, n=2)
        )

        # layer 3-4
        self.dark3 = nn.Sequential(
            Conv(c3, c3, 3, 2),
            C3K2V12(c3, c4, n=2)
        )

        # layer 5-6
        self.dark4 = nn.Sequential(
            Conv(c4, c4, 3, 2),
            A2C2f(c4, c4, n=4, block_type="attn")
        )

        # layer 7-8
        self.dark5 = nn.Sequential(
            Conv(c4, c4, 3, 2),
            A2C2f(c4, c4, n=4, block_type="attn")
        )

        # karena p3, p4, p5 diambil dari dark3, dark4, dark5
        self.out_channels = (c4, c4, c4)

    def forward(self, x):
        x = self.stem(x)

        x = self.dark2(x)

        x = self.dark3(x)
        p3 = x   # 768 channel

        x = self.dark4(x)
        p4 = x   # 768 channel

        x = self.dark5(x)
        p5 = x   # 768 channel

        return p3, p4, p5
    
class YOLOv12Neck(nn.Module):
    def __init__(self, ch=(768, 768, 768)):
        super().__init__()

        p3_ch, p4_ch, p5_ch = ch

        self.up = nn.Upsample(scale_factor=2.0, mode="nearest")

        # layer 11
        # concat: p5 upsample + p4 = 768 + 768 = 1536
        self.fpn1 = A2C2f(
            p5_ch + p4_ch,   # 1536
            p4_ch,           # 768
            n=2,
            block_type="c3k"
        )

        # layer 14
        # concat: fpn1 upsample + p3 = 768 + 768 = 1536
        self.fpn2 = A2C2f(
            p4_ch + p3_ch,   # 1536
            384,             # output layer 14
            n=2,
            block_type="c3k"
        )

        # layer 15
        self.down1 = Conv(384, 384, 3, 2)

        # layer 17
        # concat: down1 + fpn1 = 384 + 768 = 1152
        self.pan1 = A2C2f(
            384 + p4_ch,     # 1152
            p4_ch,           # 768
            n=2,
            block_type="c3k"
        )

        # layer 18
        self.down2 = Conv(p4_ch, p5_ch, 3, 2)

        # layer 20
        # concat: down2 + p5 = 768 + 768 = 1536
        self.pan2 = C3K2_Neck(
            p5_ch + p5_ch,   # 1536
            p5_ch,           # 768
            n=2
        )

        self.out_channels = (384, 768, 768)

    def forward(self, p3, p4, p5):
        # layer 9-11
        x = self.up(p5)
        x = torch.cat([x, p4], dim=1)
        n4 = self.fpn1(x)

        # layer 12-14
        x = self.up(n4)
        x = torch.cat([x, p3], dim=1)
        n3 = self.fpn2(x)

        # layer 15-17
        x = self.down1(n3)
        x = torch.cat([x, n4], dim=1)
        n4 = self.pan1(x)

        # layer 18-20
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
    def __init__(self, nc=80):
        super().__init__()

        self.backbone = YOLOv12Backbone()
        self.neck = YOLOv12Neck(self.backbone.out_channels)
        self.head = YOLOv12Head(self.neck.out_channels,nc)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)

        return self.head([p3, p4, p5])
    
model = YOLOv12(nc=3)

u = YOLO(r"yolo12x.pt")

official_model = u.model
my_params = sum(p.numel() for p in model.parameters())
official_params = sum(p.numel() for p in official_model.parameters())

print("My Params:", my_params)
print("Official Params:", official_params)

my_model = YOLOv12(nc=3)
model_dict = my_model.state_dict() 
    
official_model = YOLO("yolo12x.pt") 
official_dict = official_model.state_dict() 
new_dict = {} 
loaded_pairs = [] 

for (custom_key, custom_value), (official_key, official_value) in zip(model_dict.items(),official_dict.items()):
    if custom_value.shape == official_value.shape:
        new_dict[custom_key] = official_value
        loaded_pairs.append((custom_key, official_key, tuple(custom_value.shape)))
                
    else:
        print(
            f"SKIP | custom: {custom_key} {tuple(custom_value.shape)} "
            f"!= official: {official_key} {tuple(official_value.shape)}"
        )
        
    
missing, unexpected = my_model.load_state_dict(new_dict, strict=False)

print("Missing:")
for k in missing:
    print(k)
    