import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralNorm(nn.Module):
    """Spectral Normalization for GAN stability (Miyato et al. 2018)"""
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def forward(self, *args):
        self._update_u_v()
        return self.module(*args)

def spectral_norm(module, name='weight', power_iterations=1):
    """Apply spectral normalization to a module's weight"""
    return SpectralNorm(module, name, power_iterations)

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

class LBPTextureModule(nn.Module):
    """Enhanced LBP-guided texture preservation module"""
    def __init__(self, channels):
        super().__init__()
        # Multiple kernel sizes to capture different texture patterns
        self.lbp_conv3 = nn.Conv2d(channels, channels//2, kernel_size=3, padding=1, groups=channels//4)
        self.lbp_conv5 = nn.Conv2d(channels, channels//2, kernel_size=5, padding=2, groups=channels//4)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(channels)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(channels + 64, channels // 2, kernel_size=1),  # +64 for class embedding
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, class_embed):
        # Apply multi-scale LBP-inspired convolution
        lbp_features3 = self.lbp_conv3(x)
        lbp_features5 = self.lbp_conv5(x)
        
        # Combine features
        lbp_features = torch.cat([lbp_features3, lbp_features5], dim=1)
        lbp_features = self.fusion(lbp_features)
        
        # Generate attention map using class embedding
        b, c, h, w = x.shape
        class_map = class_embed.view(b, 64, 1, 1).expand(b, 64, h, w)
        concat_features = torch.cat([x, class_map], dim=1)
        attention_map = self.attention(concat_features)
        
        # Apply attention to enhance texture details
        return x + attention_map * lbp_features

class ComplexGenerator(nn.Module):
    """Enhanced Generator with deeper architecture and improved texture handling"""
    def __init__(self, latent_dim=128, num_classes=7, ngf=64):
        super(ComplexGenerator, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.ngf = ngf
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, 64)
        
        # Initial projection of latent vector
        self.initial = nn.Sequential(
            # Input is Z concatenated with class embedding, going into a convolution
            nn.ConvTranspose2d(latent_dim + 64, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True)
            # State size: (ngf*16) x 4 x 4
        )
        
        # Main generator blocks with deeper architecture inspired from complex.py
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 16 x 16
            
            # Add self-attention at 16x16 resolution for better feature correlation
            # SelfAttention is added externally in forward pass
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 32 x 32
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 64 x 64
        )
        
        # Self-attention module at 16x16 feature map
        self.attention = SelfAttention(ngf * 4)
        
        # LBP texture module to enhance texture details
        self.texture_module = LBPTextureModule(ngf)
        
        # Final layer to generate images
        self.final = nn.Sequential(
            # Modified to output grayscale images (1 channel)
            nn.Conv2d(ngf, 1, 3, 1, 1, bias=False),
            nn.Tanh()
            # State size: 1 x 64 x 64
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        self.to(self.device)
    
    def _init_weights(self, m):
        # Fix for spectral normalized layers where direct weight access causes error
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            try:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            except AttributeError:
                # Skip if wrapped by spectral normalization
                pass
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            try:
                nn.init.normal_(m.weight, std=0.02)
            except AttributeError:
                pass
    
    def forward(self, z, class_labels):
        # Get class embedding
        c = self.class_embed(class_labels)
        
        # Concatenate latent vector with class embedding
        batch_size = z.size(0)
        z_c = torch.cat([z, c], dim=1)
        z_c = z_c.view(batch_size, self.latent_dim + 64, 1, 1)
        
        # Initial block
        x = self.initial(z_c)
        
        # First two upsampling blocks
        x = self.main[0:6](x)  # Up to ngf*4 x 16 x 16
        
        # Apply self-attention at 16x16 resolution
        x = self.attention(x)
        
        # Continue with remaining upsampling blocks
        x = self.main[6:](x)  # From 16x16 to 64x64
        
        # Apply LBP-guided texture enhancement
        x = self.texture_module(x, c)
        
        # Final output layer
        return self.final(x) 