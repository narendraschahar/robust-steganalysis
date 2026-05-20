import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# 1.  EXPANDED SRM FILTER BANK (30 kernels)
#     Based on the full 30-filter SRM used in
#     Fridrich & Kodovsky (2012) TIFS paper.
# ─────────────────────────────────────────────
def get_srm_kernels():
    """Return 30 fixed high-pass SRM residual filters."""
    K = []

    # 3×3 first-order
    K.append(np.array([[0,0,0],[0,1,-1],[0,0,0]], np.float32))
    K.append(np.array([[0,0,0],[0,1,0],[0,-1,0]], np.float32))
    K.append(np.array([[0,0,0],[0,1,0],[-1,0,0]], np.float32))
    K.append(np.array([[0,0,0],[0,1,0],[0,0,-1]], np.float32))

    # 3×3 second-order
    K.append(np.array([[0,0,0],[0,1,-2],[0,0,1]], np.float32))
    K.append(np.array([[0,0,0],[0,1,0],[0,-2,1]], np.float32))
    K.append(np.array([[1,-2,1],[0,0,0],[0,0,0]], np.float32))
    K.append(np.array([[0,0,0],[1,-2,1],[0,0,0]], np.float32))
    K.append(np.array([[0,1,0],[0,-2,0],[0,1,0]], np.float32))
    K.append(np.array([[0,0,1],[0,-2,0],[1,0,0]], np.float32))
    K.append(np.array([[1,0,0],[0,-2,0],[0,0,1]], np.float32))

    # 3×3 third-order
    K.append(np.array([[0,0,0],[0,1,-3],[0,1,1]], np.float32))
    K.append(np.array([[0,0,0],[-1,3,-3],[0,0,1]], np.float32))
    K.append(np.array([[0,0,0],[1,-3,3],[-1,0,0]], np.float32))
    K.append(np.array([[0,0,0],[0,3,-3],[1,-1,0]], np.float32))

    # 3×3 Laplacian-type
    K.append(np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], np.float32))
    K.append(np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], np.float32))
    K.append(np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.float32))
    K.append(np.array([[-1,0,-1],[0,4,0],[-1,0,-1]], np.float32))

    # 5×5 kernels
    K.append(np.array([[-1, 2,-2, 2,-1],
                        [ 2,-6, 8,-6, 2],
                        [-2, 8,-12, 8,-2],
                        [ 2,-6, 8,-6, 2],
                        [-1, 2,-2, 2,-1]], np.float32) / 12.0)
    K.append(np.array([[ 1,-4, 6,-4, 1],
                        [-4,16,-24,16,-4],
                        [ 6,-24,36,-24, 6],
                        [-4,16,-24,16,-4],
                        [ 1,-4, 6,-4, 1]], np.float32) / 12.0)
    K.append(np.array([[ 0, 0,-1, 0, 0],
                        [ 0, 0, 2, 0, 0],
                        [-1, 2,-4, 2,-1],
                        [ 0, 0, 2, 0, 0],
                        [ 0, 0,-1, 0, 0]], np.float32))
    K.append(np.array([[ 0, 0, 0, 0, 0],
                        [ 0,-1, 2,-1, 0],
                        [ 0, 2,-4, 2, 0],
                        [ 0,-1, 2,-1, 0],
                        [ 0, 0, 0, 0, 0]], np.float32))
    K.append(np.array([[ 0, 0, 0, 0, 0],
                        [ 0, 0,-1, 0, 0],
                        [ 0,-1, 4,-1, 0],
                        [ 0, 0,-1, 0, 0],
                        [ 0, 0, 0, 0, 0]], np.float32))
    K.append(np.array([[ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0,-1, 2,-1, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0]], np.float32))

    # Diagonal edge detectors
    K.append(np.array([[ 0, 0, 0],[ 0, 1, 0],[ 0, 0,-1]], np.float32))
    K.append(np.array([[ 0, 0, 0],[ 0, 1, 0],[-1, 0, 0]], np.float32))
    K.append(np.array([[ 0, 1,-1],[ 0,-1, 1],[ 0, 0, 0]], np.float32))
    K.append(np.array([[ 0,-1, 1],[ 0, 1,-1],[ 0, 0, 0]], np.float32))
    K.append(np.array([[-1, 2,-1],[ 0, 0, 0],[ 1,-2, 1]], np.float32))

    # Pad all to 5×5
    out = []
    for k in K:
        pad = (5 - k.shape[0]) // 2
        if pad > 0:
            k = np.pad(k, pad, mode="constant")
        out.append(k)

    kernels = torch.tensor(np.stack(out)[:, None, :, :], dtype=torch.float32)
    return kernels


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


# ─────────────────────────────────────────────
# 2.  BUILDING BLOCKS
# ─────────────────────────────────────────────
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, relu=True, groups=1):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p,
                            bias=False, groups=groups),
                  nn.BatchNorm2d(out_ch)]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module):
    """Squeeze-Excitation block (lighter than CBAM, faster convergence)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """Residual block with optional stride-2 average pooling downsampling."""
    def __init__(self, in_ch, out_ch, pool=False, se=True):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, out_ch),
            ConvBNAct(out_ch, out_ch, relu=False)
        )
        # Shortcut projection
        if in_ch != out_ch or pool:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

        self.pool_layer = nn.AvgPool2d(2, stride=2) if pool else nn.Identity()
        self.pool = pool
        self.se = SEBlock(out_ch) if se else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.pool_layer(self.shortcut(x))
        out = self.pool_layer(self.net(x))
        out = self.se(out)
        return self.act(out + residual)


# ─────────────────────────────────────────────
# 3.  ORIGINAL SIMPLE BASELINE
# ─────────────────────────────────────────────
class SRMTLUCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = FixedSRMConv()
        self.tlu = TLU(3.0)
        srm_ch = self.srm.out_channels
        self.features = nn.Sequential(
            ConvBNAct(srm_ch, 32), ConvBNAct(32, 32), nn.AvgPool2d(2),
            ConvBNAct(32, 64), ConvBNAct(64, 64), nn.AvgPool2d(2),
            ConvBNAct(64, 128), ConvBNAct(128, 128), nn.AvgPool2d(2),
            ConvBNAct(128, 256), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.40), nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.tlu(self.srm(x))
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# 4.  IMPROVED PROPOSED MODEL
#     • 30-filter SRM + TLU
#     • Deep unpooled stem (4 blocks, 512×512)
#     • Three SE-ResBlocks with aggressive pooling
#     • Cosine-friendly dropout
# ─────────────────────────────────────────────
class ProposedResAttnTLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = FixedSRMConv()
        self.tlu = TLU(3.0)
        srm_ch = self.srm.out_channels          # 30

        # Stem conv to 64 channels
        self.stem = ConvBNAct(srm_ch, 64, k=3, p=1)

        # Stage 1 – Unpooled (keep 512×512)
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, pool=False, se=False),
            ResBlock(64, 64, pool=False, se=False),
            ResBlock(64, 64, pool=False, se=False),
            ResBlock(64, 64, pool=False, se=False),
        )

        # Stage 2 – Pooled with SE
        self.layer2 = ResBlock(64,  128, pool=True, se=True)   # → 256×256
        self.layer3 = ResBlock(128, 256, pool=True, se=True)   # → 128×128
        self.layer4 = ResBlock(256, 512, pool=True, se=True)   # → 64×64
        self.layer5 = ResBlock(512, 512, pool=True, se=True)   # → 32×32

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.tlu(self.srm(x))
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
# 5.  SRNET-INSPIRED MODEL (new option)
#     Based on Boroumand et al. SRNet TIFS 2019
#     Group 1: Type 1 (no pooling, fixed SRM)
#     Group 2: Type 2 (no pooling, learned)
#     Group 3: Type 3 (pool at each block)
#     Group 4: Type 4 (global avg pool)
# ─────────────────────────────────────────────
class SRNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False) \
            if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1) if pool else nn.Identity()

    def forward(self, x):
        out = self.act(self.shortcut(x) + self.conv(x))
        return self.pool(out)


class SRNetProposed(nn.Module):
    """SRNet-style architecture retaining SRM preprocessing."""
    def __init__(self):
        super().__init__()
        self.srm = FixedSRMConv()
        self.tlu = TLU(3.0)
        srm_ch = self.srm.out_channels      # 30

        # Type 1: no pool, fixed SRM already applied
        self.type1 = nn.Sequential(
            SRNetBlock(srm_ch, 64, pool=False),
            SRNetBlock(64, 16, pool=False),
        )
        # Type 2: no pool, learned
        self.type2 = nn.Sequential(
            SRNetBlock(16, 16, pool=False),
            SRNetBlock(16, 16, pool=False),
            SRNetBlock(16, 16, pool=False),
            SRNetBlock(16, 16, pool=False),
        )
        # Type 3: pool every block
        self.type3 = nn.Sequential(
            SRNetBlock(16, 16, pool=True),
            SRNetBlock(16, 64, pool=True),
            SRNetBlock(64, 128, pool=True),
            SRNetBlock(128, 256, pool=True),
        )
        # Type 4: global average → classifier
        self.type4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.tlu(self.srm(x))
        x = self.type1(x)
        x = self.type2(x)
        x = self.type3(x)
        return self.type4(x)


# ─────────────────────────────────────────────
# 6.  FACTORY
# ─────────────────────────────────────────────
def create_model(name):
    if name == "srm_tlu_cnn":
        return SRMTLUCNN()
    if name == "proposed_resattn_tlu":
        return ProposedResAttnTLU()
    if name == "srnet_proposed":
        return SRNetProposed()
    raise ValueError(f"Unknown model: {name}")
