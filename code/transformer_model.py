import torch.nn as nn
import torch as torch
import torch.nn.functional as F 
from loss import *
from model_module import *

class PatchEmbed(nn.Module):
    """
    Patch embedding for 2D or 3D inputs.
    Produces tokens and remembers spatial shape.
    """
    def __init__(self, in_ch, embed_dim, patch_size=2, dim=2):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.norm = nn.LayerNorm(embed_dim)
        Conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.proj = Conv(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B,C,H,W) or (B,C,D,H,W)
        x = self.proj(x)

        spatial_shape = x.shape[-self.dim:]
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = self.norm(tokens)
        return tokens, spatial_shape

class TokensToFeatureMap(nn.Module):
    """
    Convert tokens back to spatial feature maps (2D or 3D).
    """
    def __init__(self, dim=2):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim

    def forward(self, tokens, spatial_shape):
        # tokens: (B, N, C)
        B, N, C = tokens.shape

        if self.dim == 3:
            D, H, W = spatial_shape
            return tokens.transpose(1, 2).reshape(B, C, D, H, W)
        else:
            H, W = spatial_shape
            return tokens.transpose(1, 2).reshape(B, C, H, W)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth=4, heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, heads=heads)
            for _ in range(depth)
        ])

    def forward(self, x):
        # x: (B, N, C)
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, init_scale= 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        self.gamma1 = nn.Parameter(init_scale * torch.ones(embed_dim))
        self.gamma2 = nn.Parameter(init_scale * torch.ones(embed_dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.gamma1  # Residual connection
        x = x + self.mlp(self.norm2(x)) * self.gamma2
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape

        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
#-----------
class TransformerStage(nn.Module):
    """
    Transformer encoder stage with patch embedding and re-projection.
    Supports 2D and 3D.
    """
    def __init__(
        self,
        in_ch,
        embed_dim,
        depth=2,
        heads=8,
        patch_size=2,
        dim=2,
    ):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim

        self.patch_embed = PatchEmbed(
            in_ch=in_ch,
            embed_dim=embed_dim,
            patch_size=patch_size,
            dim=dim
        )

        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            heads=heads
        )

        self.tokens_to_map = TokensToFeatureMap(dim=dim)

    def forward(self, x):
        # x: (B,C,H,W) or (B,C,D,H,W)
        tokens, spatial_shape = self.patch_embed(x)
        tokens = self.transformer(tokens)
        x_out = self.tokens_to_map(tokens, spatial_shape)
        return x_out
