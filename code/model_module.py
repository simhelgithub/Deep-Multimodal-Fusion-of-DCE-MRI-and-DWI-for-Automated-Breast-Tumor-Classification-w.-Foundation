import torch.nn as nn
import torch as torch
import torch.nn.functional as F 
from loss import *
import warnings
from transformer_model import *
from torch.nn import BatchNorm3d, Identity, init 
from dataclasses import dataclass
import math

# -----------------------------
# Small utilities & losses
# -----------------------------
def smooth_l1_loss(a, b):
    return F.smooth_l1_loss(a, b)
@dataclass
class FeatureSpec:
    channels: int
    stride: int


# -----------------------------
# Lightweight SE module handles 2d / 3d input
# -----------------------------        
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=2, dim = 2):
        super().__init__()
        Conv = nn.Conv3d if dim==3 else nn.Conv2d
        AdaptiveAvgPool = nn.AdaptiveAvgPool3d if dim==3 else nn.AdaptiveAvgPool2d
      
        
        mid = max(channels // reduction, 1)
        
        self.fc = nn.Sequential(
            AdaptiveAvgPool(1),
            Conv(channels, mid, kernel_size=1, bias=True),
            nn.GELU(),
            Conv(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        ) 
    def forward(self, x):
        w = self.fc(x)
        return x * w, w

# Convenience aliases for modality attentions
class TemporalAttention(SEBlock): pass
class ChannelAttention(SEBlock): pass

class MaskGuidedSpatialAttention(nn.Module):
    """
    Mask-guided spatial attention for 2D/3D.
    """
    def __init__(self, in_channels_img, in_channels_mask,
                 hidden_channels=16, dim=2):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim

        Conv = nn.Conv3d if dim == 3 else nn.Conv2d

        self.interp_mode = "trilinear" if dim == 3 else "bilinear"

        # Learnable scaling (optional effect strength)
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # Mask → 1-channel attention
        self.mask_processor = nn.Sequential(
            Conv(in_channels_mask, hidden_channels, 1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            Conv(hidden_channels, 1, 1),
            nn.Sigmoid()  # produces attention in (0,1)
        )

    def forward(self, img_features, mask_features):
        # ---- static-safe shape extraction ----
        target = img_features.shape[-self.dim:]
        source = mask_features.shape[-self.dim:]

        # ---- static-controlled interpolation ----
        if source != target: 
            mask_up = F.interpolate(
                mask_features,
                size=target, 
                mode=self.interp_mode,
                align_corners=False
            )
        else:
            mask_up = mask_features

        # ---- compute attention ----
        A = self.mask_processor(mask_up)     # (B,1,H,W) or (B,1,D,H,W)
        A = torch.clamp(A, 1e-4, 1.0 - 1e-4)

        # attention: modulate contrast of img_features
        out = img_features * (1 + self.gamma * A)
        return out, A


class ReconHead(nn.Module):
    """
    Lightweight reconstruction head:
    - Uses 3x3 convs to capture local context
    - Optionally upsamples if needed
    """
    def __init__(self, in_ch, recon_ch=1, upsample=False, dim = 2):
        super().__init__()
        self.upsample = upsample
        self.dim = dim
        Convd = nn.Conv3d if dim==3 else nn.Conv2d
        BatchNorm = nn.BatchNorm3d if dim==3 else nn.BatchNorm2d

        self.conv = nn.Sequential(
            Convd(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            BatchNorm(in_ch),
            nn.GELU(),
            Convd(in_ch, recon_ch, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        out = self.conv(x)
        if self.upsample:
            mode = "trilinear" if self.dim == 3 else "bilinear"
            out = F.interpolate(out, scale_factor=2, mode=mode, align_corners=False)
        return out


#currently handles only 128x128, 64x64, 32x32 or 8x8 to do cleanup


class MaskHeadResize(nn.Module):
    """
    compile()-friendly 2D/3D mask head.
    cleanly supports input spatial sizes: 512, 256, 128, 64, 32
    These map to output: 32.
    """

    def __init__(self, in_ch, mid_ch=64, out_ch=1, out_size=32, dim=2):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.out_size = out_size

        Conv = nn.Conv3d if dim == 3 else nn.Conv2d
        Act = nn.GELU
        mode = "trilinear" if dim == 3 else "bilinear"
        self.interp_mode = mode

        # shared 1×1 projection
        self.pre = Conv(in_ch, mid_ch, kernel_size=1)

        # downsampling blocks
        self.down_64_to_32 = nn.Sequential(
            Conv(mid_ch, mid_ch, 3, stride=2, padding=1),
            Act(),
        )

        self.down_128_to_32 = nn.Sequential(
            Conv(mid_ch, mid_ch, 3, stride=2, padding=1),  # 128 -> 64
            Act(),
            Conv(mid_ch, mid_ch, 3, stride=2, padding=1),  # 64 -> 32
            Act(),
        )

        self.down_256_to_32 = nn.Sequential(
            Conv(mid_ch, mid_ch, 3, stride=2, padding=1),  # 256 -> 128
            Act(),
            Conv(mid_ch, mid_ch, 3, stride=2, padding=1),  # 128 -> 64
            Act(),
            Conv(mid_ch, mid_ch, 3, stride=2, padding=1),  # 64 -> 32
            Act(),
        )

        self.down_512_to_32 = nn.Sequential(
            Conv(mid_ch, mid_ch, 3, stride=2, padding=1),  # 512 -> 256
            Act(),
            Conv(mid_ch, mid_ch, 3, stride=2, padding=1),  # 256 -> 128
            Act(),
            Conv(mid_ch, mid_ch, 3, stride=2, padding=1),  # 128 -> 64
            Act(),
            Conv(mid_ch, mid_ch, 3, stride=2, padding=1),  # 64 -> 32
            Act(),
        )


        # output conv
        self.out = Conv(mid_ch, out_ch, kernel_size=1)

        # ---- STATIC DISPATCH TABLE ----
        self.dispatch = {
            32: None,                 # identity
            64: self.down_64_to_32,
            128: self.down_128_to_32,
            256: self.down_256_to_32,
            512: self.down_512_to_32,
        }
    def forward(self, x):
        x = self.pre(x)

        # static shape (torch.compile-safe)
        size = x.shape[-1]            # square input
        op = self.dispatch.get(size, "interp")  # fallback to interpolation
        if op is None:
            pass
        elif op == "interp":
            x = F.interpolate(
                x,
                size=(self.out_size,) * self.dim,
                mode=self.interp_mode,
                align_corners=False,
            )
        else:
            x = op(x)

        return self.out(x)

# -----------------------------
# Residual Lite Block with reconstruction head
# -----------------------------
class ResNetLiteBlock_withRecon(nn.Module):
    """
    Lightweight residual bottleneck block with optional SE and reconstruction head.
    - Works for both 2D and 3D (dim=2 or dim=3)
    - recon_ch > 0 enables reconstruction
    - num_repeats > 1 stacks the bottleneck convs to increase depth
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        downsample=False,
        recon_ch=1,
        use_se=False,
        se_reduction=2,
        dropout=0.4,
        dim=2,
        num_repeats=1,  
        downsample_each_repeat = False,
        mid_squeeze = 2
    ):
        super().__init__()
        self.dim = dim
        self.num_repeats = num_repeats

        Conv = nn.Conv3d if dim == 3 else nn.Conv2d
        BatchNorm = nn.BatchNorm3d if dim == 3 else nn.BatchNorm2d
        Dropout = nn.Dropout3d if dim == 3 else nn.Dropout

        stride = 2 if downsample else 1
        mid_ch = max(out_ch // mid_squeeze, 1)

        # Build repeated bottleneck layers
        self.bottlenecks = nn.ModuleList()
        for i in range(num_repeats):
            if downsample_each_repeat:
              b_stride = stride
            else:
              b_stride = stride if i == 0 else 1  # only first bottleneck downsamples
            self.bottlenecks.append(nn.Sequential(
                Conv(in_ch if i == 0 else out_ch, mid_ch, kernel_size=1, stride=b_stride, bias=False),
                BatchNorm(mid_ch),
                nn.GELU(),
                Dropout(p=dropout),
                Conv(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
                BatchNorm(mid_ch),
                nn.GELU(),
                Conv(mid_ch, out_ch, kernel_size=1, bias=False),
                BatchNorm(out_ch)
            ))


        self.act = nn.GELU()
        self.dropout = Dropout(p=dropout)

        # Skip connection over the entire stack
        if stride > 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                Conv(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                BatchNorm(out_ch),
            )
        else:
            self.skip = None

        # Optional SE block
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_ch, reduction=se_reduction, dim=dim)
        else:
            self.se = None

        # Optional reconstruction head
        self.recon_ch = int(recon_ch)
        if self.recon_ch > 0:
            self.reconstruct = ReconHead(out_ch, recon_ch, upsample=False, dim=dim)
        else:
            self.reconstruct = None

    def forward(self, x):
        identity = self.skip(x) if self.skip is not None else x

        out = x
        for bottleneck in self.bottlenecks:
            out = bottleneck(out)

        # Add skip once after all bottlenecks
        out = self.act(out + identity)
        out = self.dropout(out)

        # Optional SE
        if self.use_se:
            out, _ = self.se(out)

        # Optional reconstruction
        f_rec = self.reconstruct(out) if self.reconstruct is not None else None

        return out, f_rec


#---
# Projector 
#-- 

class Projector(nn.Module):
    """
    Lightweight projector head for mimic/self-supervised loss.
    2D and 3D 
    """
    def __init__(self, in_ch, proj_dim=64, dim=2):
        super().__init__()
        assert dim in (2, 3), "dim must be 2 or 3"
        self.dim = dim
        
        # Select ops (2D ↔ 3D)
        Conv = nn.Conv3d if dim == 3 else nn.Conv2d
        BatchNorm = nn.BatchNorm3d if dim == 3 else nn.BatchNorm2d

        self.proj = nn.Sequential(
            Conv(in_ch, proj_dim, kernel_size=1, bias=False),
            BatchNorm(proj_dim),
            nn.GELU(),

            Conv(proj_dim, proj_dim, kernel_size=1, bias=False),
            BatchNorm(proj_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.proj(x)



#---
# classificaiton head
#---
class ClassificationHead(nn.Module):
    def __init__(self, in_ch, num_classes, dim=2, normalize=True):
        super().__init__()
        Pool = nn.AdaptiveAvgPool3d if dim==3 else nn.AdaptiveAvgPool2d
        self.pool = Pool((1,1,1) if dim==3 else (1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_ch, num_classes)
        self.normalize = normalize

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        if self.normalize:
            x = F.normalize(x, dim=1)
        return self.fc(x)

class FeatureDownAlign(nn.Module):
    """
    Align higher-resolution features (f1) to lower-resolution (f2)
    using strided conv instead of interpolation if downsample=True.
    Supports 2D and 3D.
    """
    def __init__(self, in_ch, out_ch, dim=2, downsample=True):
        super().__init__()
        assert dim in (2, 3)
        Conv = nn.Conv3d if dim == 3 else nn.Conv2d
        Norm = nn.BatchNorm3d if dim == 3 else nn.BatchNorm2d
        stride = 2 if downsample else 1
        kernel_size = 3 if downsample else 1
        padding = 1 if downsample else 0
        if in_ch != out_ch or downsample:
          self.proj = nn.Sequential(
              Conv(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
              Norm(out_ch),
              nn.GELU(),
          )

        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(x)
        
#---
# class for more complex backbone neck connection
#--
@torch._dynamo.disable
class BackboneAdapter(nn.Module):
    """
    Adapts arbitrary backbone outputs into 3 features (f1/f2/f3).
    Handles both CNN (spatial features) and transformer (token) backbones.
    """

    def __init__(
        self,
        backbone,
        selected_indices_chains,   # list of lists, one per f1/f2/f3
        out_channels=(64, 128, 256),
        dim=2,
        is_transformer=False,      # True if backbone outputs tokens
    ):
        super().__init__()

        assert len(selected_indices_chains) == 3, "Must provide 3 chains for f1/f2/f3"
        assert len(out_channels) == 3, "Must provide 3 output channels for f1/f2/f3"

        self.backbone = backbone
        self.selected_indices_chains = selected_indices_chains
        self.dim = dim
        self.is_transformer = is_transformer

        Conv = nn.Conv3d if dim == 3 else nn.Conv2d
        BN = nn.BatchNorm3d if dim == 3 else nn.BatchNorm2d

        # Discover backbone feature hierarchy
        info = backbone.feature_info
        self.features = [
            FeatureSpec(c, s) for c, s in zip(info.channels(), info.reduction())
        ]

        # Build necks for each feature chain
        self.necks = nn.ModuleDict()
        for i, indices_chain in enumerate(self.selected_indices_chains):
            in_ch = sum([self.features[idx].channels for idx in indices_chain])
            out_ch = out_channels[i]
            self.necks[f'f{i+1}'] = nn.Sequential(
                Conv(in_ch, out_ch, 3, padding=1),
                BN(out_ch),
                nn.GELU(),
                Conv(out_ch, out_ch, 3, padding=1),
                BN(out_ch),
                nn.GELU(),
            )

    def forward(self, x):
            feats = self.backbone(x)  # list of features from backbone
            outputs = []

            for i, indices_chain in enumerate(self.selected_indices_chains):
                chain_feats = []
                for idx in indices_chain:
                    f = feats[idx]
                    if self.is_transformer:
                        if f.ndim == 3:
                            # token sequence -> reshape to (B, C, H, W)
                            B, N, C = f.shape
                            H = W = int(N ** 0.5)
                            f = f.permute(0, 2, 1).contiguous().view(B, C, H, W)
                        elif f.ndim == 4:
                            # already (B,C,H,W), nothing to do
                            pass
                        else:
                            raise ValueError(f"Unexpected transformer feature shape: {f.shape}")
                    chain_feats.append(f)

                # Concatenate along channel dim and pass through neck
                chain_feats = torch.cat(chain_feats, dim=1)
                chain_out = self.necks[f'f{i+1}'](chain_feats)
                outputs.append(chain_out)

            # Unpack into f1_b, f2_b, f3_b
            return outputs[0], outputs[1], outputs[2]

# -------------------------
# ModelMaskHeadBackbone experimental backbone
# -------------------------
class ModelMaskHeadBackbone(nn.Module):
    """
    Encoder with optional backbone, modality attention, reconstruction blocks,
    and a mask head that can attach at f1, f2, or f3 depending on mask_stage.
    """

    def __init__(
        self,
        method,
        parameters_dict,
        backbone=None,

    ):
        super().__init__()

        # --
        # Load settings
        # ---
        self.channel_num = parameters_dict[f'{method}_channel_num']
        self.num_classes = parameters_dict['class_num']
        self.dim = parameters_dict['dim']
        
        model_parameters =  parameters_dict[f'{method}_model_parameters']
        self.enable_modality_attention = model_parameters['enable_modality_attention']
        self.use_se = model_parameters['use_se']
        self.use_hybrid_transformer = model_parameters["use_hybrid_transformer"]
        self.use_backbone = model_parameters["use_backbone"]
        self.channels = model_parameters['channels']
        self.proj_dim = model_parameters['proj_dim']
        self.dropout = model_parameters['dropout']
        self.num_repeats = model_parameters['repeat_blocks']
        self.mid_squeeze = model_parameters['mid_squeeze']
        self.downsample = model_parameters['downsample']
        self.downsample_each_repeat = model_parameters['downsample_each_repeat']
        self.selected_indices_chains = model_parameters['backbone_index_lists']
        self.backbone_out_channels = model_parameters['backbone_out_channels'] # not implemented
        self.transformer_backbone = model_parameters['transformer_backbone']

        mask_parameters =  model_parameters['mask_parameters']
        self.mask_enabled = mask_parameters["mask"]
        self.mask_stage =  mask_parameters["mask_stage"].lower()
        self.mask_size = mask_parameters['mask_target_size'][0]


        # ------------------------
        # Config + channels
        # ------------------------
        c1, c2, c3 = self.channels
        self.enable_modality_attention = self.enable_modality_attention
        
        if self.dim == 3:
            self.proj_pool = nn.AdaptiveAvgPool3d((self.proj_dim, self.proj_dim, self.proj_dim))
        else:
            self.proj_pool = nn.AdaptiveAvgPool2d((self.proj_dim, self.proj_dim))

        # ------------------------
        # Backbone + neck
        # ------------------------
        self.backbone = torch._dynamo.disable(backbone) #disable dynamo for backbone or issues
        #if self.backbone is not None:
        if self.use_backbone:
            #(c1,c2,c3,c4) = backbone.output_dims 
            #print(backbone.output_dims)
            self.backbone_adapter = BackboneAdapter(
                backbone=self.backbone,
                selected_indices_chains=self.selected_indices_chains,
                out_channels=(c1,c1,c2),
                is_transformer=self.transformer_backbone
            )
            block1_in = c1
        else:
            block1_in = self.channel_num

        # ------------------------
        # Blocks
        # ------------------------
        self.block1 = ResNetLiteBlock_withRecon(
            block1_in, c1, downsample=self.downsample[0], recon_ch=1, use_se=self.use_se, dim=self.dim, dropout = self.dropout, num_repeats=self.num_repeats[0], downsample_each_repeat = self.downsample_each_repeat, mid_squeeze=self.mid_squeeze
        )
        self.block2 = ResNetLiteBlock_withRecon(
            c1, c2, downsample=self.downsample[1], recon_ch=1, use_se=self.use_se, dim=self.dim, dropout = self.dropout, num_repeats=self.num_repeats[1], downsample_each_repeat = self.downsample_each_repeat, mid_squeeze=self.mid_squeeze
        )

        if not self.use_hybrid_transformer:
            self.block3 = ResNetLiteBlock_withRecon(c2, c3, downsample=self.downsample[2], recon_ch=0, use_se=self.use_se, dim=self.dim, dropout = self.dropout, num_repeats=self.num_repeats[2], downsample_each_repeat = self.downsample_each_repeat, mid_squeeze=self.mid_squeeze)
        else:
            # ---- HYBRID CNN → TRANSFORMER ----
            self.transformer = TransformerStage(
                in_ch=c2,
                embed_dim=model_parameters['transformer_embed_dim'],
                depth=model_parameters['transformer_depth'],
                heads=model_parameters['transformer_heads'],
                patch_size=model_parameters['transformer_patch_size'],
                dim=self.dim
            )

            # project transformer features to c3 for heads
            Conv = nn.Conv3d if self.dim == 3 else nn.Conv2d
            self.trans_out_proj = Conv(model_parameters['transformer_embed_dim'], c3, kernel_size=1)

        # ------------------------
        # Modality attention
        # ------------------------
        self.modality_attention = None
        if self.enable_modality_attention:
            if method == "dce":
                self.modality_attention = TemporalAttention(self.channel_num, reduction=2)
            elif method == "dwi":
                self.modality_attention = ChannelAttention(self.channel_num, reduction=2)
            else:
                raise ValueError("Unknown method for modality attention.")
        #self.f1_weight = nn.Parameter(torch.tensor(0.5))  # scalar for f1 + f1_b
        self.f2_weight = nn.Parameter(torch.tensor(0.5))  # scalar for f2 + f2_b
        self.f3_weight = nn.Parameter(torch.tensor(0.5))  # scalar for f3 + f3_b
        #self.norm_f1 = nn.GroupNorm(c1, c1)
        self.norm_f2 = nn.GroupNorm(c1, c1)
        self.norm_f3 = nn.GroupNorm(c2, c2)
        # ------------------------
        # Mask head depends on mask_stage
        # ------------------------

        if self.mask_enabled:
          #applied after the block, so downsample never needed
          self.f1_to_f2 = FeatureDownAlign(c1, c2, dim=self.dim, downsample = False) 
          self.f2_to_f3 =  FeatureDownAlign(c2, c3, dim=self.dim, downsample = False)  

          if self.mask_stage == "f1":
              mask_in = c1
          if self.mask_stage == "f2": 
              mask_in = c2
          if self.mask_stage == "f3":
              mask_in = c3

          self.mask_head = MaskHeadResize(in_ch = mask_in, out_size=self.mask_size, dim=self.dim)

          # Spatial attention 
          self.mask_spatial_attention = MaskGuidedSpatialAttention(
              in_channels_img=c3, in_channels_mask=1, dim=self.dim
          )
          
          
          
          if self.use_hybrid_transformer and self.mask_stage == "f3":
              raise ValueError("mask_stage='f3' not supported with hybrid transformer")




        # Classification head (always uses f3)
        self.classification_head = ClassificationHead(
            in_ch=c3, num_classes=self.num_classes, dim=self.dim
        )

        # ------------------------
        # Projectors
        # ------------------------
        self.proj_f1 = Projector(c1, self.proj_dim, dim=self.dim)
        self.proj_f2 = Projector(c2, self.proj_dim, dim=self.dim)
        self.proj_r1 = Projector(1, self.proj_dim, dim=self.dim)
        self.proj_r2 = Projector(1, self.proj_dim, dim=self.dim)

    # =====================================================================
    # Forward
    # =====================================================================
    def forward(self, x, masks=None):
        mask_pred = None
        mask_attn_map = None
        # Optional modality attention
        if self.modality_attention is not None:
            x_in, mod_attn_map = self.modality_attention(x)
        else:
            x_in = x
            mod_attn_map = None
        #f1_b, f2_b, f3_b = None, None, None
        # Optional backbone
        if self.use_backbone:
            f1_b, f2_b, f3_b = self.backbone_adapter(x_in)
        # ------------------------
        # Encoder blocks & mask head placements
        # ------------------------
        if self.use_backbone:
     
            f1_in = f1_b
        else:
            f1_in = x_in
        f1, r1 = self.block1(f1_in)

        if self.mask_enabled and self.mask_stage == "f1":
            mask_pred = self.mask_head(f1)
            f1, mask_attn_map = self.mask_spatial_attention(f1, mask_pred)

        #f2, r2 = self.block2(f1 + (f2_b if f2_b is not None else 0))
        if self.use_backbone:
          alpha = torch.sigmoid(self.f2_weight)
          f2_in = self.norm_f2(alpha * f2_b + (1 - alpha) * f1)
        else:
            f2_in = f1

        f2, r2 = self.block2(f2_in)
        
        if self.mask_enabled and self.mask_stage == "f2":
            f1_aligned = self.f1_to_f2(f1)
            f2_mask_input = f2 + f1_aligned
            mask_pred = self.mask_head(f2_mask_input)
            f2, mask_attn_map = self.mask_spatial_attention(f2, mask_pred)
        # ---- Final stage, hybrid option ----
        if not self.use_hybrid_transformer:
            if self.use_backbone:
                alpha = torch.sigmoid(self.f3_weight)
                f3_in = self.norm_f3(alpha * f3_b + (1 - alpha) * f2)
            else:
                f3_in = f2

            f3, _ = self.block3(f3_in)
            if self.mask_enabled and self.mask_stage == "f3":
                f2_aligned = self.f2_to_f3(f2)
                f3_mask_input = f3 + f2_aligned
                mask_pred = self.mask_head(f3_mask_input)
                f3, mask_attn_map = self.mask_spatial_attention(f3, mask_pred)

        else:
            f2_mid = self.transformer(f2)          # must keep spatial layout
            f3 = self.trans_out_proj(f2_mid)                  
        # ------------------------
        # Projections
        # ------------------------
        f1_p = self.proj_pool(f1)
        f2_p = self.proj_pool(f2)
        r1_p = self.proj_pool(r1)
        r2_p = self.proj_pool(r2)

        p1 = self.proj_f1(f1_p)
        p2 = self.proj_f2(f2_p)
        p1_r = self.proj_r1(r1_p)
        p2_r = self.proj_r2(r2_p)

        # ------------------------
        # Classification
        # ------------------------
        logits = self.classification_head(f3)
        
        # ------------------------
        # Aux dictionary
        # ------------------------
        
        aux = {
            "raw_feats": [f1, f2, f3],
            "recon_feats": [r1, r2],
            "proj_pairs": [p1, p1_r, p2, p2_r],
            "mask_attn_map": mask_attn_map,
            "mod_attn_map": mod_attn_map,
        }
        return logits, aux, mask_pred

#------------------------------
#-----------------------------
#Fusion model & helpers
#------------------------------
#------------------------------

# -----------------------------
# Modality gating attention 
# -----------------------------

class GatingAttention(nn.Module):
    """
    Learnable global gating weights for two modalities (DWI, DCE) 
    with optional mask confidence
    2D or 3D
    """
    def __init__(self, feat_dim, use_mask_attention=True, dim = 2):
        super().__init__()
        self.use_mask_attention = use_mask_attention
        in_dim = feat_dim * 2 + (2 if use_mask_attention else 0)
        self.fc = nn.Linear(in_dim, 2)
        self.dim = dim

    def forward(self, pvec_dwi, pvec_dce, dwi_mask=None, dce_mask=None):
        """
        pvec_*: B x C (already pooled)
        dwi_mask, dce_mask: B x 1 x H x W (2D) or B x 1 x D x H x W (3D)
        """
        if self.use_mask_attention and (dwi_mask is not None and dce_mask is not None):
            # auto-detect spatial dims (all except batch and channel)
            spatial_dims = tuple(range(2, dwi_mask.ndim))

            # mean over spatial dims -> shape (B,1)
            dwi_conf = dwi_mask.mean(dim=spatial_dims)
            dce_conf = dce_mask.mean(dim=spatial_dims)

            # flatten to 2D (B,1) for concatenation
            dwi_conf = dwi_conf.view(dwi_conf.shape[0], -1)
            dce_conf = dce_conf.view(dce_conf.shape[0], -1)

            x = torch.cat([pvec_dwi, pvec_dce, dwi_conf, dce_conf], dim=1)
        else:
            x = torch.cat([pvec_dwi, pvec_dce], dim=1)

        weights = torch.softmax(self.fc(x), dim=1)  # (B,2)
        return weights

class FusionReduce(nn.Module):
    #Reduces concatenated features (2*channels -> channels) with BN + activation
    def __init__(self, in_ch, out_ch, dim=2):
        super().__init__()
        Conv = nn.Conv3d if dim==3 else nn.Conv2d
        BatchNorm = nn.BatchNorm3d if dim==3 else nn.BatchNorm2d
        self.reduce = nn.Sequential(
            Conv(in_ch, out_ch, kernel_size=1, bias=False),
            BatchNorm(out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.reduce(x)




class CrossAttentionBlock(nn.Module):
    """
    Cross-attention on pooled tokens with optional small FFN.
    Works for any channel dimension.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.attn_ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, query_tokens, key_value_tokens):
        # query_tokens, key_value_tokens: B, N, C
        attn_out, attn_weights = self.cross_attn(query_tokens, key_value_tokens, key_value_tokens, need_weights=True)
        attn_out = attn_out + self.attn_ffn(attn_out)
        return attn_out, attn_weights


class FusionModel(nn.Module):
    """
    Fusion model for DWI + DCE with optional cross-attention, gating, SE, mask head, 
    reconstruction head, and projector. Fully configurable via a parameters dictionary.
    """

    def __init__(self, parameters_dict):
        super().__init__()

        # -- fetch fusion model config --
        fusion_config = parameters_dict['fusion_model_parameters']
        fusion_specific = fusion_config['fusion_specific_parameters']

        # general parameters
        self.dim = parameters_dict['dim']
        self.num_classes = parameters_dict['class_num']

        # fusion-specific parameters
        self.fusion_channels = fusion_specific['fusion_channels']
        self.token_pool = fusion_specific['token_pool']
        self.mha_heads = fusion_specific['mha_heads']
        self.dwi_ch = fusion_specific['dwi_out_channels']
        self.dce_ch = fusion_specific['dce_out_channels']
        self.use_cross_attention = fusion_specific['use_cross_attention']
        self.use_mask_attention = fusion_specific['use_mask_attention']
        self.fusion_recon_ch = fusion_specific['fusion_recon_ch']

        self.proj_dim = fusion_config['proj_dim']
        self.mask_size = fusion_config['mask_parameters']['mask_target_size'][0] 
        self.dropout = fusion_config['dropout']
        self.use_se_in_fusion = fusion_config['use_se']
        # If encoder_out_channels provided, create separate projectors; else assume f3 channels == fusion_channels


        Conv = nn.Conv3d if self.dim == 3 else nn.Conv2d
        # deterministic 1x1 projectors from encoder f3 -> fusion_channels
        self.proj_in_dwi = Conv(self.dwi_ch, self.fusion_channels, kernel_size=1, bias=False) if self.dwi_ch != self.fusion_channels else nn.Identity()
        self.proj_in_dce = Conv(self.dce_ch, self.fusion_channels, kernel_size=1, bias=False) if self.dce_ch != self.fusion_channels else nn.Identity()


        # reduce concat (2*fusion_channels -> fusion_channels)
        self.fusion_conv_reduce = FusionReduce(2*self.fusion_channels, self.fusion_channels, dim=self.dim)
        self.refine_act = nn.GELU()

        # Optional SE block after residual refine
        if self.use_se_in_fusion:
            self.fusion_se = SEBlock(self.fusion_channels, reduction=2, dim=self.dim)
        else:
            self.fusion_se = None

        # gating attention (global)
        self.gating = GatingAttention(feat_dim=self.fusion_channels, use_mask_attention=self.use_mask_attention, dim =  self.dim)
        self.refine = ResNetLiteBlock_withRecon(in_ch = self.fusion_channels, out_ch=self.fusion_channels, dim=self.dim, dropout=self.dropout, mid_squeeze = 2)

        # Optional cross-attention
        if self.use_cross_attention:
            self.cross_attn_block = CrossAttentionBlock(self.fusion_channels, num_heads=self.mha_heads)


        # fused mask head -> fixed 32x32 output
        self.mask_head = MaskHeadResize(in_ch = self.fusion_channels, out_size=self.mask_size, dim=self.dim)

        # fusion reconstruction head (1x1)
        self.fusion_reconstruct = ReconHead(in_ch=self.fusion_channels, recon_ch=self.fusion_recon_ch, upsample=False,dim=self.dim)


        # classifier
        if self.dim == 3:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Flatten(),
                nn.Linear(self.fusion_channels, self.num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(self.fusion_channels, self.num_classes)
            )
        # small projection for fusion mimic if desired
        self.projF = Projector(in_ch=self.fusion_channels, proj_dim=self.proj_dim, dim=self.dim)

    def _to_tokens(self, feat):
        """
        Convert a feature map to token sequence for cross-attention.
        Supports 2D (B,C,H,W) or 3D (B,C,D,H,W) inputs.
        """
        B, C = feat.shape[:2]
        if self.dim == 3:
            Dp, Hp, Wp = self.token_pool
            pooled = F.adaptive_avg_pool3d(feat, (Dp, Hp, Wp))
            tokens = pooled.view(B, C, Dp * Hp * Wp).permute(0, 2, 1).contiguous()  # B, N, C
        else:
            Hp, Wp = self.token_pool
            pooled = F.adaptive_avg_pool2d(feat, (Hp, Wp))
            tokens = pooled.view(B, C, Hp * Wp).permute(0, 2, 1).contiguous()  # B, N, C
        return tokens

    def forward(self, raw_feats_dwi, raw_feats_dce, dwi_mask_pred=None, dce_mask_pred=None):
        """
        Expect raw_feats_* to be lists [f1,f2,f3] where f3 is the deepest encoder output.
        dwi_mask_pred and dce_mask_pred are the predicted masks (logits or probs) from encoders (can be None).
        Returns: logits, fused_mask_logits, aux_dict
        """
        # pick deepest features
        f3_dwi = raw_feats_dwi[-1]
        f3_dce = raw_feats_dce[-1]

        # project encoder outputs to fusion_channels using deterministic 1x1 convs created in __init__
        p_dwi = self.proj_in_dwi(f3_dwi)
        p_dce = self.proj_in_dce(f3_dce)


        # concat and reduce
        cat = torch.cat([p_dwi, p_dce], dim=1)  # B, 2C, H, W
        reduced = self.fusion_conv_reduce(cat)  # B, C, H, W

        # refine residual
        residual, _ = self.refine(reduced)
        refined = self.refine_act(reduced + residual)

        # gating: use global pooled vectors from original projected deep features
        # pooled vectors
        if self.dim==3:
            pvec_dwi = F.adaptive_avg_pool3d(p_dwi, (1,1,1)).view(p_dwi.size(0), -1)
            pvec_dce = F.adaptive_avg_pool3d(p_dce, (1,1,1)).view(p_dce.size(0), -1)
        else:
            pvec_dwi = F.adaptive_avg_pool2d(p_dwi, (1,1)).view(p_dwi.size(0), -1)
            pvec_dce = F.adaptive_avg_pool2d(p_dce, (1,1)).view(p_dce.size(0), -1)

        # compute gating
        gating_weights = self.gating(pvec_dwi, pvec_dce, dwi_mask=dwi_mask_pred, dce_mask=dce_mask_pred)
        
        shape = [ -1 ] + [1]*(refined.ndim-1)
        alpha_dwi = gating_weights[:,0].view(*shape)
        alpha_dce = gating_weights[:,1].view(*shape)
        # fuse by gating original projected deep features (not the reduced map)
        fused = alpha_dwi * p_dwi + alpha_dce * p_dce

        # optional cross-attention on tokens (small)
        attn_weights = None
        # Cross-attention
        if self.use_cross_attention:
            t_dwi = self._to_tokens(p_dwi)
            t_dce = self._to_tokens(p_dce)
            attn_out, attn_weights = self.cross_attn_block(t_dwi, t_dce)
            # reshape back to spatial
            B, N, C = attn_out.shape
            Hp, Wp = self.token_pool
            lowres = attn_out.permute(0,2,1).contiguous().view(B, C, Hp, Wp)
            mode = 'trilinear' if self.dim==3 else 'bilinear'
            up = F.interpolate(lowres, size=fused.shape[-self.dim:], mode=mode, align_corners=False)
            fused = fused + up

        # Fusion refinement
        # optional use se
        if self.fusion_se is not None:
            fused_refined, _ = self.fusion_se(fused)
        else:
            fused_refined = fused
        
        # mask (32x32)
        fused_mask_logits = self.mask_head(fused_refined)

        # classifier
        logits = self.classifier(fused_refined)

        # reconstruction and projection
        recon_fused = self.fusion_reconstruct(fused_refined) if self.fusion_reconstruct is not None else None
        proj_fused = self.projF(fused_refined)

        aux = {
            "proj_fused": proj_fused,
            "recon_fused": recon_fused,
            "gating_weights": gating_weights,
            "attn_weights": attn_weights,
            "p_dwi": p_dwi,
            "p_dce": p_dce
        }
        return logits, fused_mask_logits, aux

def init_parameter(model):
    if isinstance(model, nn.Linear):
        if model.weight is not None:
            init.kaiming_uniform_(model.weight.data)
            #init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
                init.constant_(model.bias.data, 0) # Initialize biases to 0
    elif isinstance(model, nn.BatchNorm1d) or isinstance(model, nn.BatchNorm2d) or isinstance(model, nn.BatchNorm3d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=1, std=0.02) # Initialize BatchNorm weights to 1
        if model.bias is not None:
            init.constant_(model.bias.data, 0) # Initialize BatchNorm biases to 0
    else:
        pass


def initialize_model(model,requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
    # Apply custom initialization to the model's modules
    model.apply(init_parameter)
    return model