import torch
import torch.nn as nn
import torchvision.models as models

import timm
import os
import re
from collections import defaultdict

# --------------------
# MONAI import helper
# ----------------------
def try_import_monai():
    try:
        import monai
        return monai
    except Exception:
        print('failed to import monai')
        return None


# ===========================================================
#  TIMM RESNET BACKBONE WITH OUTPUT_STRIDE=8, 2d only
# ===========================================================
def build_imagenet_backbone(
    name='resnet50d',
    pretrained=True,
    device='cuda',
    in_channels=3,
    output_stride=8
):
    """
    Build a feature-only ResNet backbone (resnet50 / resnet50d) with OS=8.
    Returns stages + output_dim automatically.
    """
    os.environ["TIMM_MODEL_CACHE"] = "/mnt/models/timm"

    # Create backbone with feature maps from all stages
    backbone = timm.create_model(
        name,
        pretrained=pretrained,
        features_only=False,
        in_chans=in_channels,
        output_stride=output_stride,       
    ).to(device)

    # ---- remove classifier head (TIMM version) ----
    if hasattr(backbone, "global_pool"):
      backbone.global_pool = nn.Identity()

    if hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    if hasattr(backbone, "head"):
        backbone.head = nn.Identity()


    backbone.output_dim = backbone.num_features
    backbone.expected_input = "B, C, H, W"

    return backbone




'''
#swapped out becuase it has too large stride for model, 32 vs 8 of 50d

# ===========================================================
#  IMAGENET RESNET50 (2D ONLY)
# ===========================================================
def build_imagenet_backbone(name='resnet50d', pretrained=True, device='cuda',
                            in_channels=3, is_3d=False):

    if is_3d:
        raise ValueError("ResNet is 2D-only. Use is_3d=False.")


    m = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)

    # Fix 1st conv for arbitrary input channels
    old_conv = m.conv1
    m.conv1 = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    # Initialize new channels
    if in_channels >= 3:
        m.conv1.weight.data[:, :3] = old_conv.weight
        if in_channels > 3:
            m.conv1.weight.data[:, 3:] = old_conv.weight.mean(1, keepdim=True)
    else:
        repeat = max(1, 3 // in_channels)
        m.conv1.weight.data = old_conv.weight[:, :in_channels].repeat(1, repeat, 1, 1)

    # Remove classifier (keep spatial feature maps)
    backbone = nn.Sequential(*list(m.children())[:-2])

    backbone.output_dim = 2048
    backbone.is_3d = False
    backbone.expected_input = "B, C, H, W"
    backbone.stages = split_backbone_into_stages(backbone)

    return backbone.to(device)
'''

# ===========================================================
#  SWIN-UNETR (2D + 3D)
# ===========================================================
def build_swin_unetr(in_channels=1, pretrained_path=None,
                     device='cuda', is_3d=False):
    MONAI = try_import_monai()

    if MONAI is None:
        raise RuntimeError("MONAI not installed. pip install monai")

    from monai.networks.nets import SwinUNETR

    if is_3d:
        model = SwinUNETR(
            img_size=(64, 128, 128),
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            feature_size=48,
        )
        model.is_3d = True
        model.expected_input = "B, 1, D, H, W"
        model.output_dim = 48

    else:
        model = SwinUNETR(
            img_size=(128, 128),
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            feature_size=48,
        )
        model.is_3d = False
        model.expected_input = "B, C, H, W"
        model.output_dim = 48

    # Load pretrained checkpoint
    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state, strict=False)

    # add stage grouping
    model.stages = split_backbone_into_stages(model)

    return model.to(device)


# ===========================================================
#  MODEL GENESIS (MONAI UNet encoder only)
# ===========================================================
def build_models_genesis(in_channels=1, pretrained_path=None,
                         device='cuda', is_3d=False):

    if MONAI is None:
        raise RuntimeError("MONAI not installed. pip install monai")

    from monai.networks.nets import UNet

    if is_3d:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
        )
        output_dim = 256
    else:
        model = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
        )
        output_dim = 512

    # Load checkpoint
    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state, strict=False)

    # Extract encoder only (MONAI UNet stores encoder as a list)
    encoder = nn.Sequential(*model.encoder) if hasattr(model, "encoder") else model

    encoder.output_dim = output_dim
    encoder.is_3d = is_3d
    encoder.expected_input = "B, 1, D, H, W" if is_3d else "B, C, H, W"

    # add stage grouping
    encoder.stages = split_backbone_into_stages(encoder)

    return encoder.to(device)


# ===========================================================
#  MAIN BUILDER returns backbone
# ===========================================================
def build_medical_backbone(parameters, device, method, in_channels, is_3d=False):
    """
    Entry point for building any backbone.
    """

    name = parameters[f"{method}_model_parameters"]["backbone_str"].lower()
    pretrained_path = parameters[f"{method}_model_parameters"].get("pretrained_path", None)

    #2d only resnet
    if name in["resnet50d"]:
        return build_imagenet_backbone(
            name=name,
            pretrained=True,
            device=device,
            in_channels=in_channels,
        )

    if name == "swin_unetr":
        return build_swin_unetr(
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            device=device,
            is_3d=is_3d,
        )

    if name == "model_genesis":
        return build_models_genesis(
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            device=device,
            is_3d=is_3d,
        )

    raise ValueError(f"Unknown backbone: {name}")

  

