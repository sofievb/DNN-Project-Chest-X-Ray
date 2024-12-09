import torch.nn as nn
import torch

class ResBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, use_bottleneck=False, downsample=False):
        super(ResBlocks, self).__init__()
        if use_bottleneck:
            self.block = self.bottleneck_block(in_channels, out_channels)
        else:
            self.block = self.residual_block(in_channels, out_channels)

        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def bottleneck_block(self, in_channels, out_channels):
        bottleneck_channels = out_channels // 4
        return nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.block(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(64, 256, n_blocks=3, use_bottleneck=True)
        self.conv3_x = self._make_layer(256, 512, n_blocks=4, use_bottleneck=True, downsample=True)
        self.conv4_x = self._make_layer(512, 1024, n_blocks=6, use_bottleneck=True, downsample=True)
        self.conv5_x = self._make_layer(1024, 2048, n_blocks=3, use_bottleneck=True, downsample=True)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, out_channels, n_blocks, use_bottleneck, downsample=False):
        layers = [ResBlocks(in_channels, out_channels, use_bottleneck, downsample)]
        for _ in range(1, n_blocks):
            layers.append(ResBlocks(out_channels, out_channels, use_bottleneck))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.global_avg_pooling(x)
        return self.fc(torch.flatten(x, 1))  
