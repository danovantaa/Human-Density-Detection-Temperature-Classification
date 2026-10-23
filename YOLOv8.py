import torch 
import torch.nn as nn
from torchinfo import summary
import math 
from ultralytics import YOLO

SCALES = {
    "n": (0.33, 0.25, 1024),
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}

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

class Bottleneck(nn.Module) : 
    def __init__(self, c1, c2 ,shortcut=True, k=3, e=0.5):
        super().__init__()
        c_ = int(e*c2)
        self.cv1 = Conv(c1, c_, k, 1)
        self.cv2 = Conv(c_, c2, k, 1)
        self.add = shortcut and c1 == c2    
        
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
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))
        
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, s=1):
        super().__init__()
        c_ = c1//2
        self.cv1 =Conv(c1, c_,1,1 ) 
        self.cv2 = Conv(c_*4, c2, 1,1) 
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k//2)
        
    def forward(self, x):
        x = self.cv1(x)
        max1 = self.m(x)
        max2 = self.m(max1)
        return self.cv2(torch.cat((x, max1, max2, self.m(max2)),1))

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

def make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.d = dim

    def forward(self, x):
        return torch.cat(x, self.d)

def scale_channel(c, width_mul, max_channels):
    return make_divisible(min(c, max_channels) * width_mul, 8)

class YOLOv8Backbone(nn.Module):
    def __init__(self, scale="m"):
        super().__init__()

        depth_mul, width_mul, max_channels = SCALES[scale]

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
    def __init__(self, ch, scale="m"):
        super().__init__()

        depth_mul, _, _ = SCALES[scale]
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
    def __init__(self, ch, nc=80, reg_max=16):
        super().__init__()

        self.nc = nc
        self.reg_max = reg_max

        self.cv2 = nn.ModuleList()  # regression
        self.cv3 = nn.ModuleList()  # classification
        self.dfl = DFL(reg_max)

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
    
class YOLOv8(nn.Module):
    def __init__(self, scale="m",nc=80):
        super().__init__()
        
        self.backbone = YOLOv8Backbone(scale)
        ch = self.backbone.out_channels
        
        self.neck = YOLOv8Neck(ch, scale)
        self.head = YOLOv8Head(ch, nc)
        
    def forward(self, x):
        p3, p4, p5= self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head([p3, p4, p5])

    
model = YOLOv8("x", nc=80)

u = YOLO(r"yolov8x.pt")

official_model = u.model
my_params = sum(p.numel() for p in model.parameters())
official_params = sum(p.numel() for p in official_model.parameters())

print("My Params:", my_params)
print("Official Params:", official_params)

x = torch.randn(1, 3, 640, 640)

official_model.eval()

with torch.no_grad():
    y = official_model(x)
    
model.eval()

with torch.no_grad():
    y2 = model(x)
    
print("Official")

for i, out in enumerate(y):

    if torch.is_tensor(out):
        print(i, out.shape)

    elif isinstance(out, dict):
        print(i, "DICT")

        for k, v in out.items():

            if torch.is_tensor(v):
                print("   ", k, v.shape)

            else:
                print("   ", k, type(v))

    else:
        print(i, type(out))


print("\nModel Saya")

for i, out in enumerate(y2):

    if torch.is_tensor(out):
        print(i, out.shape)

    elif isinstance(out, dict):
        print(i, "DICT")

        for k, v in out.items():

            if torch.is_tensor(v):
                print("   ", k, v.shape)

            else:
                print("   ", k, type(v))

    else:
        print(i, type(out))
        
my_model = YOLOv8("x", nc=3)
model_dict = my_model.state_dict() 
    
official_model = YOLO("yolov8x.pt") 
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
                        
print(f"Loaded keys: {len(new_dict)} / {len(model_dict)}")
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")   