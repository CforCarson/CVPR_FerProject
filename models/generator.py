import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """Self attention module for the transformer blocks"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention weights
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        return out

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class LBPTextureModule(nn.Module):
    """LBP-guided texture preservation module"""
    def __init__(self, channels):
        super().__init__()
        self.lbp_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.attention = nn.Sequential(
            nn.Conv2d(channels + 64, channels // 2, kernel_size=1),  # +64 for class embedding
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, class_embed):
        # Apply LBP-inspired convolution
        lbp_features = self.lbp_conv(x)
        
        # Generate attention map using class embedding
        b, c, h, w = x.shape
        class_map = class_embed.view(b, 64, 1, 1).expand(b, 64, h, w)
        concat_features = torch.cat([x, class_map], dim=1)
        attention_map = self.attention(concat_features)
        
        # Apply attention to enhance texture details
        return x + attention_map * lbp_features

class TextureEnhancedGenerator(nn.Module):
    """Generator architecture with texture-preserving capabilities"""
    def __init__(self, latent_dim=128, num_classes=7, embed_dim=256, num_heads=8):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initial projection of latent vector
        self.latent_proj = nn.Linear(latent_dim, 12*12*64)
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, 64)
        
        # Patch embedding
        self.patch_size = 4
        self.patch_embed = nn.Conv2d(64, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (12//self.patch_size)**2, embed_dim))
        
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
        self.to(self.device)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            
    def forward(self, z, class_labels):
        # Process latent vector
        x = self.latent_proj(z).view(-1, 64, 12, 12)
        
        # Get class embedding
        c = self.class_embed(class_labels)
        
        # Patch embedding
        x_patches = self.patch_embed(x)  # B, C, H/p, W/p
        
        # Reshape for transformer: [B, C, H, W] -> [B, C, HW] -> [B, HW, C]
        b, e, h, w = x_patches.shape
        x_seq = x_patches.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Add positional embedding
        x_seq = x_seq + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x_seq = block(x_seq)
        
        # Reshape back: [B, N, C] -> [B, C, H, W]
        x_transformed = x_seq.transpose(1, 2).reshape(b, e, h, w)
        
        # Apply LBP-guided texture enhancement
        x_textured = self.texture_module(x_transformed, c)
        
        # Decode to generate the final image
        return self.decoder(x_textured)