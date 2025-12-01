import torch.nn as nn
import torch as torch
import torch.nn.functional as F # Import F for resizing
from torch.nn import init # Import the init module
from loss import *


dce_channel_count = 6
dwi_channel_count = 13


# -----------------------------
# Small utilities & losses
# -----------------------------

def smooth_l1_loss(a, b):
    return F.smooth_l1_loss(a, b)


# -----------------------------
# Lightweight SE module (channel / temporal)
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ELU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

# Convenience aliases for modality attentions
class TemporalAttention(SEBlock): pass
class ChannelAttention(SEBlock): pass

# -----------------------------
# Residual Lite Block with reconstruction head
# -----------------------------
class ResNetLiteBlock_withRecon(nn.Module):
    """
    Lightweight residual bottleneck that optionally reconstructs to recon_ch channels.
    recon_ch: if 0 -> reconstruction disabled; >0 -> produce recon_ch-channel reconstruction map
    use_se: apply SEBlock on output (lightweight)
    """
    def __init__(self, in_ch, out_ch, downsample=False, recon_ch=1, use_se=False, se_reduction=2, dropout=0.3):
        super().__init__()
        stride = 2 if downsample else 1
        mid_ch = max(out_ch // 2, 1) # can be changed 

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(p = dropout)

        self.skip = None
        if stride > 1 or in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                                      nn.BatchNorm2d(out_ch))
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_ch, reduction=se_reduction)

        self.recon_ch = int(recon_ch)
        if self.recon_ch > 0:
            self.reconstruct = nn.Conv2d(out_ch, self.recon_ch, kernel_size=1)
        else:
            self.reconstruct = None

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.skip is not None:
            identity = self.skip(x)
        out = self.act(out + identity)
        out = self.dropout(out) 
        if self.use_se:
            out = self.se(out)
        f_rec = self.reconstruct(out) if self.reconstruct is not None else None
        return out, f_rec

# -----------------------------
# Small Projector for mimic loss (1x1 conv -> BN -> GELU -> 1x1 conv)
# -----------------------------
class Projector(nn.Module):
    def __init__(self, in_ch, proj_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
            nn.Conv2d(proj_dim, proj_dim, kernel_size=1, bias=False)
        )
    def forward(self, x):
        return self.proj(x)

# -------------------------
# ModelMaskHead experimental backbone
# -------------------------
class ModelMaskHeadBackbone(nn.Module):
    """
    encoder that:
      - uses ResNetLiteBlock_withRecon blocks
      - optionally uses modality attention (temporal for 6, channel for 13) when enable_modality_attention==True
      - always predicts masks via mask_head
      - returns: logits, aux, f3, mask_pred
    """
    def __init__(self, channel_num, num_classes=4, channels=(32,64,128), proj_dim=64, enable_modality_attention=False, use_se=False, backbone = None):
        """
        channel_num: number of input channels (6 or 13 expected for enabling modality attention)
        enable_modality_attention: bool
        """
        super().__init__()
        self.backbone = backbone
        if self.backbone is not None:
                    self.backbone_out_dim = backbone.output_dim
                    self.backbone_proj = nn.Conv2d(
                        self.backbone_out_dim, channels[0], kernel_size=1
                    )

        self.channel_num = channel_num
        c1, c2, c3 = channels
        self.enable_modality_attention = enable_modality_attention

        # Blocks with reconstruction
        # recon_ch=1 at stage1 and stage2; deep recon disabled at stage3
        self.block1 = ResNetLiteBlock_withRecon(channel_num, c1, downsample=False, recon_ch=1, use_se=use_se)
        self.block2 = ResNetLiteBlock_withRecon(c1, c2, downsample=True, recon_ch=1, use_se=use_se)
        self.block3 = ResNetLiteBlock_withRecon(c2, c3, downsample=True, recon_ch=0, use_se=use_se)

        # optional modality attention
        self.modality_attention = None
        if enable_modality_attention:
            if channel_num == dce_channel_count:
                self.modality_attention = TemporalAttention(channel_num, reduction=2)
            elif channel_num == dwi_channel_count or channel_num == dwi_channel_count+1:
                self.modality_attention = ChannelAttention(channel_num, reduction=2)
            else:
                self.modality_attention = None

        self.mask_head = MaskHeadSetSize(c1, out_size = 32)
        # Mask-guided spatial attention
        self.mask_spatial_attention = MaskGuidedSpatialAttention(in_channels_img=c3, in_channels_mask=c1)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(c3, num_classes)
        )
        # Projectors for mimic loss (stage1 & stage2)
        self.proj_f1 = Projector(c1, proj_dim)
        self.proj_f2 = Projector(c2, proj_dim)
        self.proj_r1 = Projector(1, proj_dim)
        self.proj_r2 = Projector(1, proj_dim)

    def forward(self, x, masks=None):
        """
        x: [B, channel_num, H, W]
        masks: optional GT masks (not used inside forward except maybe for guidance)
        returns: logits, aux_dict, f3, mask_pred
        """
        # Optionally apply modality attention at the input level (temporal/channel)
        x_in = x

        if self.backbone is not None:
          feats = self.backbone(x)   # expected [B, C, H', W']
          x = self.backbone_proj(feats)  # reshape to [B, c1, H', W']


        if self.modality_attention is not None:
            x_in = self.modality_attention(x_in)

        f1, r1 = self.block1(x_in)  # f1 [B,c1,H,W], r1 [B,1,H,W]

        # mask-guided spatial attention: apply before block2 to refine using mask-ish features
        # we use f1 (shallow) as mask feature source
        f1_att = f1
        # block2 input is f1_att (no change), but we can optionally apply spatial attention using f1
        f1_for_mask = f1  # keep original for proj
        f2_in = f1_att

        # Block2
        f2, r2 = self.block2(f2_in)  # f2 [B,c2,H/2,W/2]

        # Apply mask-guided attention after block2 (image features + mask features)
        # Use mask_spatial_attention to modulate f2 with f1 features (upsampled internally)
        f2_att = self.mask_spatial_attention(f2, f1_for_mask)

        # Block3 on attended features
        f3, _ = self.block3(f2_att)

        # Mask prediction
        mask_pred = self.mask_head(f1_for_mask)  # shape approx input spatial size (depending on upsampling convtranspose)

        # Projected features for mimic loss
        p1, p1_r = self.proj_f1(f1), self.proj_r1(r1)
        p2, p2_r = self.proj_f2(f2), self.proj_r2(r2)

        raw_feats = [f1, f2, f3]
        recon_feats = [r1, r2]
        proj_pairs = [p1, p1_r, p2, p2_r]

        # also return modality_attention output (for debugging / hooks), None if not present
        mod_attn_features = None
        if self.modality_attention is not None:
            mod_attn_features = x_in

        # Classification out
        classification_out = self.classification_head(f3)

        # Bundle Aux returns to keep signature (logits, aux, features, mask)
        aux = {
            "raw_feats":raw_feats,
            "recon_feats":recon_feats,
            "proj_pairs":proj_pairs,
            "mod_attn":mod_attn_features
        }


        return classification_out, aux, mask_pred


class MaskHeadSetSize(nn.Module):
    """
    Simple mask head that always outputs a (B, 1, out_size, out_size) mask
    regardless of input feature map size.

    """
    def __init__(self, in_ch, mid_ch=64, out_size=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, stride=2, padding=1),  # ↓ 1/2
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1),  # ↓ 1/4
            nn.ELU()
        )

        # squeeze to exactly out_size x out_size
        self.to_out_size = nn.AdaptiveAvgPool2d((out_size, out_size))

        self.out = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.to_out_size(x)
        return self.out(x)     # (B, 1, 32, 32)

# -----------------------------
# Modality gating attention (learn α_dwi / α_dce)
# -----------------------------

class DwiDceAttention(nn.Module):
    def __init__(self, feat_dim, use_mask_attention=True):
        super().__init__()
        self.use_mask_attention = use_mask_attention
        in_dim = feat_dim * 2 + (2 if use_mask_attention else 0)
        self.fc = nn.Linear(in_dim, 2)
    def forward(self, pvec_dwi, pvec_dce, dwi_mask=None, dce_mask=None):
        if self.use_mask_attention and (dwi_mask is not None and dce_mask is not None):
            dwi_conf = torch.sigmoid(dwi_mask).mean(dim=(1,2,3)).unsqueeze(1)
            dce_conf = torch.sigmoid(dce_mask).mean(dim=(1,2,3)).unsqueeze(1)
            x = torch.cat([pvec_dwi, pvec_dce, dwi_conf, dce_conf], dim=1)
        else:
            x = torch.cat([pvec_dwi, pvec_dce], dim=1)
        weights = torch.softmax(self.fc(x), dim=1)
        return weights

class FusionModel(nn.Module):
    def __init__(self,
                 fusion_channels=128,
                 proj_dim=64,
                 token_pool=(4,4),
                 use_cross_attention=False,
                 use_mask_attention=True,
                 mha_heads=4,
                 num_classes=4,
                 fusion_recon_ch=1,
                 use_se_in_fusion=False,
                 encoder_out_channels=None, # optional: if provided, tuple (dwi_f3_ch, dce_f3_ch)
                 dropout = 0.3):
        """
        encoder_out_channels: (dwi_f3_ch, dce_f3_ch) if encoder top features don't already match fusion_channels
        """
        super().__init__()
        self.token_pool = token_pool
        self.use_cross_attention = use_cross_attention
        self.mha_heads = mha_heads
        self.fusion_channels = fusion_channels
        self.fusion_recon_ch = fusion_recon_ch

        # deterministic 1x1 projectors from encoder f3 -> fusion_channels
        # If encoder_out_channels provided, create separate projectors; else assume f3 channels == fusion_channels
        if encoder_out_channels is not None:
            dwi_ch, dce_ch = encoder_out_channels
        else:
            dwi_ch = fusion_channels
            dce_ch = fusion_channels

        self.proj_in_dwi = nn.Conv2d(dwi_ch, fusion_channels, kernel_size=1, bias=False) if dwi_ch != fusion_channels else nn.Identity()
        self.proj_in_dce = nn.Conv2d(dce_ch, fusion_channels, kernel_size=1, bias=False) if dce_ch != fusion_channels else nn.Identity()

        # reduce concat (2*fusion_channels -> fusion_channels)
        self.fusion_conv_reduce = nn.Sequential(
            nn.Conv2d(2 * fusion_channels, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ELU(inplace=True)
        )

        # small residual refinement (1 conv residual block)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ELU(inplace=True),
            nn.Dropout(p = dropout),
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
        )
        self.refine_act = nn.ELU(inplace=True)

        # gating attention (global)
        self.gating = DwiDceAttention(feat_dim=fusion_channels, use_mask_attention=use_mask_attention)

        # optional small cross-attention (on pooled tokens)
        if self.use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_channels, num_heads=mha_heads, batch_first=True)
            self.attn_ffn = nn.Sequential(nn.LayerNorm(fusion_channels),
                                          nn.Linear(fusion_channels, fusion_channels),
                                          nn.ELU(),
                                          nn.Linear(fusion_channels, fusion_channels))

        # fusion refine convs
        self.fusion_refine = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ELU(inplace=True)
        )

        # fused mask head -> fixed 32x32 output
        self.mask_head = MaskHeadSetSize(fusion_channels, mid_ch=max(fusion_channels//2, 32), out_size=32)

        # fusion reconstruction head (1x1)
        self.fusion_reconstruct = nn.Conv2d(fusion_channels, fusion_recon_ch, kernel_size=1) if fusion_recon_ch > 0 else None

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(fusion_channels, num_classes)
        )

        # small projection for fusion mimic if desired
        self.projF = nn.Conv2d(fusion_channels, proj_dim, kernel_size=1, bias=False)

    def _to_tokens(self, feat):
        B, C, H, W = feat.shape
        pooled = F.adaptive_avg_pool2d(feat, self.token_pool)
        Hp, Wp = self.token_pool
        tokens = pooled.view(B, C, Hp * Wp).permute(0, 2, 1).contiguous()  # B,N,C
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
        residual = self.refine_conv(reduced)
        refined = self.refine_act(reduced + residual)

        # gating: use global pooled vectors from original projected deep features
        pvec_dwi = F.adaptive_avg_pool2d(p_dwi, (1,1)).view(p_dwi.size(0), -1)
        pvec_dce = F.adaptive_avg_pool2d(p_dce, (1,1)).view(p_dce.size(0), -1)
        gating_weights = self.gating(pvec_dwi, pvec_dce, dwi_mask=dwi_mask_pred, dce_mask=dce_mask_pred)  # (B,2)

        alpha_dwi = gating_weights[:, 0].view(-1,1,1,1)
        alpha_dce = gating_weights[:, 1].view(-1,1,1,1)

        # fuse by gating original projected deep features (not the reduced map)
        fused = alpha_dwi * p_dwi + alpha_dce * p_dce

        # optional cross-attention on tokens (small)
        attn_weights = None
        if self.use_cross_attention:
            t_dwi = self._to_tokens(p_dwi)   # B,N,C
            t_dce = self._to_tokens(p_dce)
            attn_out, attn_weights = self.cross_attn(t_dwi, t_dce, t_dce, need_weights=True)  # B,N,C
            attn_out = attn_out + self.attn_ffn(attn_out)
            B, N, C = attn_out.shape
            Hp, Wp = self.token_pool
            lowres = attn_out.permute(0,2,1).contiguous().view(B, C, Hp, Wp)
            up = F.interpolate(lowres, size=fused.shape[-2:], mode='bilinear', align_corners=False)
            fused = fused + up

        # refine fused
        fused_refined = self.fusion_refine(fused)

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

class MaskGuidedSpatialAttention(nn.Module):
    def __init__(self, in_channels_img, in_channels_mask):
        super().__init__()
        self.mask_processor = nn.Sequential(
            nn.Conv2d(in_channels_mask, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, img_features, mask_features):
        if mask_features.shape[-2:] != img_features.shape[-2:]:
            mask_up = F.interpolate(mask_features, size=img_features.shape[-2:], mode='bilinear', align_corners=False)
        else:
            mask_up = mask_features
        attention_map = self.mask_processor(mask_up)
        return img_features * attention_map

def init_parameter(model):
    if isinstance(model, nn.Linear):
        if model.weight is not None:
            init.kaiming_uniform_(model.weight.data)
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