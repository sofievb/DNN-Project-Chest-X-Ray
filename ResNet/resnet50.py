import torch.nn as nn
import torch
class ResBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, use_bottleneck=False, downsample=False):
        super(ResBlocks, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bottleneck = use_bottleneck
        self.downsample = downsample

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
        layers = [
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False),
            nn.BatchNorm2d(out_channels)                        
        ]
        
        return nn.Sequential(*layers)
        
    def bottleneck_block(self,in_channels, out_channels):
        layers = [
                    nn.Conv2d(in_channels,out_channels//4,kernel_size=(1,1), 
                            stride=(1,1), padding=0, bias=False),
                    nn.BatchNorm2d(out_channels//4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels//4,out_channels//4,kernel_size=(3,3),
                            stride=(1,1), padding=1, bias=False),
                    nn.BatchNorm2d(out_channels//4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                ]
        return nn.Sequential(*layers)
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        out += identity
        return nn.ReLU(inplace=True)(out)
    
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        #res blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #Modules for ResNet 50
        self.conv2_x  = self._make_layer(64,256,n_blocks=3, use_bottleneck=True)
        self.conv3_x = self._make_layer(256,512, n_blocks=4, use_bottleneck=True, downsample=True)
        self.conv4_x = self._make_layer(512,1024, n_blocks=6, use_bottleneck=True, downsample=True)
        self.conv5_x = self._make_layer(1024,2048, n_blocks=3, use_bottleneck=True, downsample=True)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def _make_layer(self, in_channels, out_channels, n_blocks, use_bottleneck, downsample=False):
        layers = []
        for i in range(n_blocks):
            layers.append(
                ResBlocks(in_channels, out_channels, use_bottleneck=use_bottleneck, downsample=(i == 0 and downsample))
            )
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #input resblocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.global_avg_pooling(x)
        #flatten
        x = torch.flatten(x,1)
        self.fc(x)

        return x
