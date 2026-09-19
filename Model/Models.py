import torch
from torch import nn

# torch.manual_seed(42)

# ---------------------
# 模型定义（适配 BCEWithLogitsLoss 二分类）
# ---------------------

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=1, drop=0.1):
        super().__init__()
        # num_classes=1 表示输出一个 logit
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch, in_dim)
        # 输出: (batch, num_classes)
        return self.net(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # -> (B, C)
        y = y.unsqueeze(1)                           # -> (B, 1, C)
        y = self.conv(y)                             # -> (B, 1, C)
        y = self.sigmoid(y).squeeze(1)               # -> (B, C)
        y = y.unsqueeze(-1).unsqueeze(-1)            # -> (B, C, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(p=drop_prob) if drop_prob > 0 else nn.Identity()
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ComplexMultiStreamCNN(nn.Module):
    def __init__(
        self,
        input_shapes,
        feature_dim=256, # 512/256
        num_classes=1,       # 二分类：1 个输出 logit
        drop_prob=0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # 多路分支
        self.branches = nn.ModuleList([
            self._make_branch(feature_dim, drop_prob)
            for _ in input_shapes
        ])

        # 多头自注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )

        # Transformer 风格残差 + LayerNorm + FFN
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)

        # 最终分类器：输出 (batch, 1)
        self.classifier = MLPHead(feature_dim, feature_dim, num_classes, drop=drop_prob)

    def _make_branch(self, feature_dim, drop_prob):
        i1=32 #64/32
        m1=64 #128/64
        m2=128 #256/128
        o1=128 #512/128
        # i1=64 #64/32
        # m1=128 #128/64
        # m2=256 #256/128
        # o1=512 #512/128
        return nn.Sequential(
            SeparableConv2d(1, i1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(i1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            ResidualBlock(i1, m1, stride=1, drop_prob=drop_prob),
            ResidualBlock(m1, m2, stride=1, drop_prob=drop_prob),
            ResidualBlock(m2, o1, stride=1, drop_prob=drop_prob),

            ECABlock(o1, k_size=3),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Dropout(p=drop_prob),
            nn.Linear(o1, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        # inputs: list of tensors, 每个 (B,1,H,W)
        orig_feats = [branch(x) for branch, x in zip(self.branches, inputs)]
        # -> list of (B, feature_dim)

        # 拼接 -> (B, N, feature_dim)
        feat_stack = torch.stack(orig_feats, dim=1)

        # 多头自注意力
        attn_out, _ = self.multihead_attn(feat_stack, feat_stack, feat_stack)

        # 残差 + Norm + FFN
        res1 = feat_stack + attn_out
        normed1 = self.norm1(res1)
        ffn_out = self.ffn(normed1)
        res2 = normed1 + ffn_out
        normed2 = self.norm2(res2)

        # 融合并分类
        fused = normed2.mean(dim=1)          # -> (B, feature_dim)
        logits = self.classifier(fused)      # -> (B, 1)
        return logits.squeeze(-1)            # -> (B,)
