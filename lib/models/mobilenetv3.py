import torch
import torch.nn as nn

# MobileNetV3-Large核心结构（仅保留多视角2D关节热图提取所需逻辑）
class MobileNetV3_Large(nn.Module):
    def __init__(self, num_joints=17, heatmap_size=(64,64)):
        super(MobileNetV3_Large, self).__init__()
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size

        # 核心特征提取层（MobileNetV3-Large主干）
        self.features = nn.Sequential(
            # 初始卷积层（输入3通道图像）
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),

            # 瓶颈层1（16→16，relu激活）
            self._bottleneck(16, 16, 3, 1, 'relu', 1),
            # 瓶颈层2-3（16→24，relu激活）
            self._bottleneck(16, 24, 3, 2, 'relu', 4),
            self._bottleneck(24, 24, 3, 1, 'relu', 3),
            # 瓶颈层4-6（24→40，relu激活）
            self._bottleneck(24, 40, 5, 2, 'relu', 3),
            self._bottleneck(40, 40, 5, 1, 'relu', 3),
            self._bottleneck(40, 40, 5, 1, 'relu', 3),
            # 瓶颈层7-10（40→80，hardswish激活）
            self._bottleneck(40, 80, 3, 2, 'hardswish', 6),
            self._bottleneck(80, 80, 3, 1, 'hardswish', 2),
            self._bottleneck(80, 80, 3, 1, 'hardswish', 2),
            self._bottleneck(80, 80, 3, 1, 'hardswish', 2),
            # 瓶颈层11-12（80→112，hardswish激活）
            self._bottleneck(80, 112, 3, 1, 'hardswish', 6),
            self._bottleneck(112, 112, 3, 1, 'hardswish', 6),
            # 瓶颈层13-15（112→160，hardswish激活）
            self._bottleneck(112, 160, 5, 2, 'hardswish', 6),
            self._bottleneck(160, 160, 5, 1, 'hardswish', 6),
            self._bottleneck(160, 160, 5, 1, 'hardswish', 6),

            # 输出层（适配关节热图，输出通道数=关节数×4）
            nn.Conv2d(160, self.num_joints * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_joints * 4),
            nn.Hardswish(),
        )

    # MobileNetV3瓶颈层（含SE注意力模块，核心）
    def _bottleneck(self, in_channels, out_channels, kernel_size, stride, act, expand_ratio):
        # 通道扩张
        expanded_channels = in_channels * expand_ratio
        bottleneck = nn.Sequential(
            # 1×1卷积扩张通道
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU() if act == 'relu' else nn.Hardswish(),

            # 深度可分离卷积
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride,
                      padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU() if act == 'relu' else nn.Hardswish(),

            # SE注意力模块（提升特征关注度）
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, expanded_channels, kernel_size=1),
            nn.Sigmoid(),

            # 1×1卷积压缩通道
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        return bottleneck

    def forward(self, x):
        # 输入：多视角图像 → 形状(B×N_cam, 3, H, W)（B=批次，N_cam=视角数）
        # 输出：多视角关节热图特征 → 形状(B×N_cam, num_joints×4, H//32, W//32)
        x = self.features(x)
        return x