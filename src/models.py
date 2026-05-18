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
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(channels, channels),
                                 ConvBNAct(channels, channels, relu=False))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.net(x))

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
        self.stem = nn.Sequential(ConvBNAct(self.srm.out_channels, 32), ResidualBlock(32))
        self.down1 = nn.Sequential(ConvBNAct(32, 64, s=2), ResidualBlock(64), CBAMLite(64))
        self.down2 = nn.Sequential(ConvBNAct(64, 128, s=2), ResidualBlock(128), CBAMLite(128))
        self.down3 = nn.Sequential(ConvBNAct(128, 192, s=2), ResidualBlock(192), CBAMLite(192))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.35),
            nn.Linear(192, 64), nn.ReLU(inplace=True), nn.Dropout(0.20), nn.Linear(64, 2)
        )
    def forward(self, x):
        x = self.tlu(self.srm(x))
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.classifier(x)

def create_model(name):
    if name == "srm_tlu_cnn":
        return SRMTLUCNN()
    if name == "proposed_resattn_tlu":
        return ProposedResAttnTLU()
    raise ValueError(f"Unknown model: {name}")
