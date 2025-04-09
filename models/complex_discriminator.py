import torch
import torch.nn as nn
import torch.nn.functional as F

from models.complex_generator import spectral_norm

# Add a fixed SelfAttention implementation that matches the generator version
class SelfAttention(nn.Module):
    """Self attention module with simplified implementation"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_channels = max(in_channels // 8, 1)
        
        # Use standard convolutions without spectral norm for simplicity
        self.query = nn.Conv1d(in_channels, self.query_channels, kernel_size=1)
        self.key = nn.Conv1d(in_channels, self.query_channels, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Flatten spatial dimensions
        flat_x = x.view(batch_size, C, -1)  # B x C x (WH)
        
        # Get projections
        q = self.query(flat_x)  # B x (C/8) x (WH)
        k = self.key(flat_x)    # B x (C/8) x (WH)
        v = self.value(flat_x)  # B x C x (WH)
        
        # Attention mechanism
        attention = torch.bmm(q.permute(0, 2, 1), k)  # B x (WH) x (WH)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value projection
        out = torch.bmm(v, attention.permute(0, 2, 1))  # B x C x (WH)
        
        # Reshape to original dimensions
        out = out.view(batch_size, C, width, height)
        
        # Residual connection with learned scale
        return self.gamma * out + x

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
        
        # Self-attention module at 8x8 feature map with ndf*4 channels
        # The discriminator has 4x the channels compared to the generator at this point
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
        # Fix for spectral normalized layers where direct weight access causes error
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # For spectral normalized modules, the weight might be inaccessible directly
            try:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            except AttributeError:
                # Skip if wrapped by spectral normalization
                pass
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Process in stages to correctly apply self-attention
        
        # First three convolutional blocks (1→ndf→ndf*2→ndf*4)
        x = self.feature_extractor[0:8](x)  # Process up to ndf*4 x 8 x 8
        
        # Apply self-attention at the ndf*4 level
        x = self.attention(x)
        
        # Final convolutional block (ndf*4→ndf*8)
        features = self.feature_extractor[8:](x)
        
        # Apply the three branches
        realfake_output = self.realfake_branch(features)
        expr_output = self.expr_branch(features)
        texture_score = self.texture_module(features)
        
        return realfake_output, expr_output, texture_score 