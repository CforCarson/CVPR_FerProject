import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedViT(nn.Module):
    def __init__(self, img_size=48, patch_size=4, in_chans=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: B, C, H, W
        x = self.proj(x)  # B, E, H/p, W/p
        x = x.flatten(2)  # B, E, N
        x = x.transpose(1, 2)  # B, N, E
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
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
        # Self-attention block
        norm_x = self.norm1(x)
        # Transpose for multihead attention: (B, N, C) -> (N, B, C)
        norm_x = norm_x.transpose(0, 1)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        # Transpose back: (N, B, C) -> (B, N, C)
        attn_output = attn_output.transpose(0, 1)
        x = x + attn_output
        
        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x

class ExpressionViT(nn.Module):
    def __init__(self, img_size=48, patch_size=4, in_chans=1, num_classes=7, 
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = PatchEmbedViT(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Get patch embeddings
        x = self.patch_embed(x)  # B, N, E
        
        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Get cls token output
        x = self.norm(x[:, 0])
        
        # Classification head
        return self.head(x) 