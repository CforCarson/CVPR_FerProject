import torch
import torch.nn as nn
import torch.nn.functional as F

class LBPDiscriminatorModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lbp_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # Apply LBP-inspired convolution
        lbp_features = self.lbp_conv(x)
        # Extract texture score
        return self.project(self.pooling(lbp_features)).view(-1, 1)

class DualBranchDiscriminator(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 48x48 -> 24x24
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 24x24 -> 12x12
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 12x12 -> 6x6
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Real/Fake classification branch
        self.realfake_branch = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),  # 6x6 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # 4x4 -> 1x1
            nn.Sigmoid()
        )
        
        # Expression classification branch
        self.expr_branch = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),  # 6x6 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
        # Texture assessment module
        self.texture_module = LBPDiscriminatorModule(256)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Real/fake prediction
        realfake_output = self.realfake_branch(features).view(-1, 1)
        
        # Expression classification
        expr_output = self.expr_branch(features)
        
        # Texture assessment
        texture_score = self.texture_module(features)
        
        return realfake_output, expr_output, texture_score 