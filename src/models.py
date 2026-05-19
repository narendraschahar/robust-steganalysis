import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_srm_kernels():
    kernels = [
        np.array([[0,0,0],[0,1,-1],[0,0,0]], dtype=np.float32),
        np.array([[0,0,0],[0,1,0],[0,-1,0]], dtype=np.float32),
        np.array([[0,0,0],[0,1,-2],[0,0,1]], dtype=np.float32),
        np.array([[0,0,0],[0,1,0],[0,-2,1]], dtype=np.float32),
        np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32),
        np.array([[0,0,0],[1,-2,1],[0,0,0]], dtype=np.float32),
        np.array([[0,1,0],[0,-2,0],[0,1,0]], dtype=np.float32),
        np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32),
        np.array([[-1,2,-2,2,-1],
                  [2,-6,8,-6,2],
                  [-2,8,-12,8,-2],
                  [2,-6,8,-6,2],
                  [-1,2,-2,2,-1]], dtype=np.float32) / 12.0,
    ]
    max_k = max(k.shape[0] for k in kernels)
    padded = []
    for k in kernels:
        pad = (max_k - k.shape[0]) // 2
        if pad > 0:
            k = np.pad(k, ((pad, pad), (pad, pad)), mode="constant")
        padded.append(k)
    return torch.tensor(np.stack(padded)[:, None, :, :], dtype=torch.float32)

class FixedSRMConv(nn.Module):
    def __init__(self):
        super().__init__()
        w = get_srm_kernels()
        self.register_buffer("weight", w)
        self.out_channels = w.shape[0]
        self.padding = w.shape[-1] // 2

    def forward(self, x):
        x = x * 255.0
        return F.conv2d(x, self.weight, padding=self.padding)

class TLU(nn.Module):
    def __init__(self, threshold=3.0):
        super().__init__()
        self.threshold = threshold
    def forward(self, x):
        return torch.clamp(x, -self.threshold, self.threshold) / self.threshold

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, relu=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
                  nn.BatchNorm2d(out_ch)]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        self.pool = pool
        self.net = nn.Sequential(
            ConvBNAct(in_ch, out_ch),
            ConvBNAct(out_ch, out_ch, relu=False)
        )
        if in_ch != out_ch or pool:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2 if pool else 1, bias=False)
        else:
            self.shortcut = nn.Identity()
            
        self.act = nn.ReLU(inplace=True)
        self.pool_layer = nn.AvgPool2d(3, stride=2, padding=1) if pool else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.net(x)
        out = self.pool_layer(out)
        return self.act(out + residual)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))
    def forward(self, x):
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class CBAMLite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class SRMTLUCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = FixedSRMConv()
        self.tlu = TLU(3.0)
        self.features = nn.Sequential(
            ConvBNAct(self.srm.out_channels, 32), ConvBNAct(32, 32), nn.AvgPool2d(2),
            ConvBNAct(32, 64), ConvBNAct(64, 64), nn.AvgPool2d(2),
            ConvBNAct(64, 128), ConvBNAct(128, 128), nn.AvgPool2d(2),
            ConvBNAct(128, 192), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.35), nn.Linear(192, 64),
            nn.ReLU(inplace=True), nn.Dropout(0.20), nn.Linear(64, 2)
        )
    def forward(self, x):
        x = self.tlu(self.srm(x))
        x = self.features(x)
        return self.classifier(x)

class ProposedResAttnTLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = FixedSRMConv()
        self.tlu = TLU(3.0)
        
        self.stem = ConvBNAct(self.srm.out_channels, 64)
        
        # Stage 1: Unpooled (preserves high-freq spatial resolution)
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, pool=False),
            ResidualBlock(64, 64, pool=False),
            ResidualBlock(64, 64, pool=False)
        )
        
        # Stage 2: Pooled with Attention
        self.layer2 = nn.Sequential(ResidualBlock(64, 64, pool=True), CBAMLite(64))
        self.layer3 = nn.Sequential(ResidualBlock(64, 128, pool=True), CBAMLite(128))
        self.layer4 = nn.Sequential(ResidualBlock(128, 256, pool=True), CBAMLite(256))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.tlu(self.srm(x))
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)

def create_model(name):
    if name == "srm_tlu_cnn":
        return SRMTLUCNN()
    if name == "proposed_resattn_tlu":
        return ProposedResAttnTLU()
    raise ValueError(f"Unknown model: {name}")
