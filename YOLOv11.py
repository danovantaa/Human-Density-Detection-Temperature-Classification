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
    
class Bottleneck(nn.Module) : 
    def __init__(self, c1, c2 ,shortcut=True, k=3, e=0.5):
        super().__init__()
        c_ = int(e*c2)
        self.cv1 = Conv(c1, c_, k, 1)
        self.cv2 = Conv(c_, c2, k, 1)
        self.add = shortcut and c1 == c2    
        
    def forward (self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  
    
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

SCALES = {
    "n": (0.50, 0.25),
    "s": (0.50, 0.50),
    "m": (0.50, 1.00),
    "l": (1.00, 1.00),
    "x": (1.00, 1.50),
}

def make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.d = dim

    def forward(self, x):
        return torch.cat(x, self.d)

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

        # stage3
        self.dark3 = nn.Sequential(
            Conv(c2*2,c3,3,2),  
            C3K2(c3,c3,n=n3,shortcut=False, expand=True)
        )

        # stage4
        self.dark4 = nn.Sequential(
            Conv(c4,c4,3,2),  
            C3K2(c4,c4,n=n4,shortcut=True)
        )

        # stage5
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
        
class TemperatureClsHead(nn.Module):
    def __init__(self, in_channels, num_temp_classes=3, hidden_channels=320, dropout=0.1):
        super().__init__()

        # (640, 40, 40) => (640, 40, 40)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )    

        self.pool = nn.AdaptiveAvgPool2d(20)
        self.drop = nn.Dropout(p=dropout, inplace=True)
        self.linear = nn.Linear(hidden_channels*20*20, num_temp_classes)

    def forward(self, x):
        x = self.conv(x)  # (640, 40, 40) => (320, 40, 40)
        x = self.pool(x) # (320, 40, 40) => (320, 20, 20)         
        x = torch.flatten(x, 1) # (B,320, 20, 20) => (B,128000)  
        x = self.drop(x)  # (B,128000) => (B,128000)
        x = self.linear(x) # (B,128000) => (B,3)       
        return x        

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
        p3, p4, p5 = self.backbone(x) # (320, 80, 80), (640, 40, 40), (1280, 20, 20)
        p3, p4, p5 = self.neck(p3, p4, p5) # (320, 80, 80), (640, 40, 40), (1280, 20, 20)
        
        det_out = self.head([p3, p4, p5]) # [(67,80,80), (67,40,40), (67,20,20)]

        temp_logits = self.temp_head(p4) # (B, 640, 40, 40) => (B, 3)

        return det_out, temp_logits # [(67,80,80), (67,40,40), (67,20,20)], (B, 3)



my_model = YOLOv11("x", nc=80)
model_dict = my_model.state_dict() 
    
official_model = YOLO("yolo11x.pt") 
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
    
print(f"Loaded keys: {len(new_dict)} / {len(model_dict)}")
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

