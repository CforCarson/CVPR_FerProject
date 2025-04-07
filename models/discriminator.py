import torch
import torch.nn as nn

class LBPDiscriminatorModule(nn.Module):
    """Texture assessment module using LBP-inspired convolutions"""
    def __init__(self, in_channels):
        super().__init__()
        self.lbp_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Conv2d(in_channels//2, 1, kernel_size=1)
        
    def forward(self, x):
        # Apply LBP-inspired convolution
        lbp_features = self.lbp_conv(x)
        # Process with 1x1 convolution
        lbp_features = self.conv1x1(lbp_features)
        # Extract texture score
        return self.project(self.pooling(lbp_features)).view(-1, 1)

class DualBranchDiscriminator(nn.Module):
    """Discriminator with branches for real/fake detection and expression classification"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 6x6 -> 6x6
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to 1x1
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Expression classification branch
        self.expr_branch = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 6x6 -> 6x6
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
        # Texture assessment module
        self.texture_module = LBPDiscriminatorModule(256)
        
        # Initialize weights
        self.apply(self._init_weights)
        self.to(self.device)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Real/fake prediction
        realfake_output = self.realfake_branch(features).view(-1, 1)
        
        # Expression classification
        expr_output = self.expr_branch(features)
        
        # Texture assessment
        texture_score = self.texture_module(features)
        
        return realfake_output, expr_output, texture_score