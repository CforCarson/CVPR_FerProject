import torch
import torch.nn as nn

from models.complex_generator import spectral_norm, SelfAttention

class LBPDiscriminatorModule(nn.Module):
    """Enhanced texture assessment module using LBP-inspired convolutions"""
    def __init__(self, in_channels):
        super().__init__()
        # Multi-scale texture analysis
        self.lbp_conv3 = spectral_norm(nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, groups=in_channels//4))
        self.lbp_conv5 = spectral_norm(nn.Conv2d(in_channels, in_channels//2, kernel_size=5, padding=2, groups=in_channels//4))
        
        # Feature fusion
        self.fusion = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, in_channels//2, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.project = spectral_norm(nn.Conv2d(in_channels//2, 1, kernel_size=1))
        
    def forward(self, x):
        # Apply multi-scale LBP-inspired convolutions
        lbp_features3 = self.lbp_conv3(x)
        lbp_features5 = self.lbp_conv5(x)
        
        # Combine features
        lbp_features = torch.cat([lbp_features3, lbp_features5], dim=1)
        lbp_features = self.fusion(lbp_features)
        
        # Extract texture score
        return self.project(self.pooling(lbp_features)).view(-1, 1)

class DualBranchComplexDiscriminator(nn.Module):
    """Enhanced Discriminator with deeper architecture and improved feature extraction"""
    def __init__(self, num_classes=7, ndf=64):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ndf = ndf
        
        # Shared feature extractor with deeper architecture
        self.feature_extractor = nn.Sequential(
            # Input is 1 x 64 x 64 (grayscale image)
            spectral_norm(nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1)),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),  # 32x32 -> 16x16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),  # 16x16 -> 8x8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Self-attention at 8x8 resolution
            # Will be added in forward pass
            
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)),  # 8x8 -> 4x4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Self-attention module at 8x8 feature map
        self.attention = SelfAttention(ndf * 4)
        
        # Real/Fake classification branch with spectral normalization
        self.realfake_branch = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, kernel_size=3, stride=1, padding=1)),  # 4x4 -> 4x4
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to 1x1
            nn.Flatten(),
            spectral_norm(nn.Linear(ndf * 16, 1)),
            nn.Sigmoid()
        )
        
        # Expression classification branch
        self.expr_branch = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, kernel_size=3, stride=1, padding=1)),  # 4x4 -> 4x4
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            spectral_norm(nn.Linear(ndf * 16, num_classes))
        )
        
        # Enhanced texture assessment module
        self.texture_module = LBPDiscriminatorModule(ndf * 8)
        
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
        # Extract features up to 8x8 resolution
        x = self.feature_extractor[0:9](x)  # Up to ndf*4 x 8 x 8
        
        # Apply self-attention
        x = self.attention(x)
        
        # Continue feature extraction
        features = self.feature_extractor[9:](x)  # From 8x8 to 4x4 with ndf*8 channels
        
        # Real/fake prediction
        realfake_output = self.realfake_branch(features)
        
        # Expression classification
        expr_output = self.expr_branch(features)
        
        # Texture assessment
        texture_score = self.texture_module(features)
        
        return realfake_output, expr_output, texture_score 