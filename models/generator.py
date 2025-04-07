import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # B, C, H, W -> B, E, H/p, W/p
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: (B, N, C)
        # Self-attention
        x_ln = self.norm1(x)
        # Transpose for multihead attention: (B, N, C) -> (N, B, C)
        x_ln = x_ln.transpose(0, 1)
        attn_output, _ = self.attn(x_ln, x_ln, x_ln)
        # Transpose back: (N, B, C) -> (B, N, C)
        attn_output = attn_output.transpose(0, 1)
        x = x + attn_output
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class LBPTextureModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.lbp_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.attention = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, class_embed):
        # x shape: B, C, H, W
        # Apply LBP-inspired convolution
        lbp_features = self.lbp_conv(x)
        
        # Generate attention map based on class embedding
        b = x.size(0)
        class_map = class_embed.view(b, -1, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        attention_map = self.attention(torch.cat([x, class_map], dim=1))
        
        # Apply attention to enhance texture details
        return x + attention_map * lbp_features

class TextureEnhancedGenerator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=7, embed_dim=256, num_heads=8):
        super().__init__()
        
        # Initial projection of latent vector
        self.latent_proj = nn.Linear(latent_dim, 12*12*64)
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, embed_dim//2)
        
        # Patch embedding
        self.patch_size = 4
        self.patch_embed = PatchEmbed(embed_dim, patch_size=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 12//self.patch_size, 12//self.patch_size))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0)
            for _ in range(4)
        ])
        
        # LBP-guided texture preservation module
        self.texture_module = LBPTextureModule(embed_dim)
        
        # Decoder (upsampling path)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1),  # 12x12 -> 24x24
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 24x24 -> 48x48
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 3, 1, 1),  # Final output: 48x48x1
            nn.Tanh()  # Output range: [-1, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, z, class_label):
        # Process latent vector and reshape
        x = self.latent_proj(z).view(-1, 64, 12, 12)
        
        # Get class embedding
        c = self.class_embed(class_label)
        
        # Patch embedding
        x = self.patch_embed(x)  # B, E, H', W'
        x = x + self.pos_embed
        
        # Reshape for transformer: B, E, H', W' -> B, E, N -> B, N, E
        b, e, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, E
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Reshape back to spatial: B, N, E -> B, E, H', W'
        x = x.transpose(1, 2).reshape(b, e, h, w)
        
        # Apply LBP-guided texture enhancement
        x = self.texture_module(x, c)
        
        # Decode to generate the final image
        return self.decoder(x) 