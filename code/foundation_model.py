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
#  TIMM RESNET BACKBONE WITH OUTPUT_STRIDE=8, 2d only resnet50d option
# ===========================================================

def build_imagenet_backbone(
    name="resnet50d",
    pretrained=True,
    device="cuda",
    in_channels=6,
    output_stride=8,
):
    os.environ["TIMM_MODEL_CACHE"] = "/mnt/models/timm"


    
    backbone = timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=in_channels,
        features_only=True,           
        output_stride=output_stride,
        out_indices=(1, 2, 3, 4),    # C2–C5
    ).to(device)
    '''    
    if true_stride == 4:
      # --- Adjust stem for lower stride ---
      # TIMM ResNet stem: conv1 + bn1 + act + maxpool
      # Reduce conv1 stride to 1
      if hasattr(backbone, "conv1"):
          backbone.conv1.stride = (1, 1)
      # Remove maxpool stride
      if hasattr(backbone, "maxpool"):
          backbone.maxpool.stride = (1, 1)
          backbone.maxpool.kernel_size = 1
          backbone.maxpool.padding = 0
      '''
    backbone.feature_info = backbone.feature_info 
    backbone.output_dims = backbone.feature_info.channels()
    backbone.expected_input = "B, C, H, W"
    backbone.is_3d = False

    return backbone


# -----------------------
# vit dino 2d only, 2 options  vit_base_patch16_224, dino_vitbase16_pretrain
# ------------------------
def build_vit_dino_backbone(
    name="vit_base_patch16_224",
    pretrained=True,
    device="cuda",
    in_channels=6,
    out_indices=None,
    img_size = 256
    
):
    """
    Build a ViT/DINO TIMM backbone returning intermediate features for adapter chains.

    Args:
        name (str): TIMM model name (ViT or DINO).
        pretrained (bool): Load pretrained weights.
        device (str): 'cuda' or 'cpu'.
        in_channels (int): Number of input channels.
        out_indices (list[int], optional): List of transformer block indices to output.
            Defaults to all blocks for typical ViT base (0..11).

    Returns:
        backbone: TIMM backbone with .feature_info, .output_dims, .expected_input
    """

    os.environ["TIMM_MODEL_CACHE"] = "/mnt/models/timm"

    # Default: output all blocks
    if out_indices is None:
        # Try to infer number of blocks from model name if possible
        # Otherwise default to first 12 blocks (ViT-Base)
        out_indices = list(range(12))

    backbone = timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=in_channels,
        features_only=True,
        out_indices=out_indices,
        img_size = img_size
    ).to(device)

    # Add metadata
    backbone.feature_info = backbone.feature_info 
    backbone.output_dims = backbone.feature_info.channels()
    backbone.expected_input = "B, C, H, W"
    backbone.is_3d = False

    return backbone


# not implemented

# ===========================================================
#  SWIN-UNETR (2D   NOT IMPLEMENTED
# ===========================================================
def build_swin_backbone(
    name="swin_tiny_patch4_window7_224",
    pretrained=True,
    device="cuda",
    in_channels=6,
    img_size = 256
):
    import os
    os.environ["TIMM_MODEL_CACHE"] = "/mnt/models/timm"

    backbone = timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=in_channels,
        features_only=True,  # hierarchical features
        out_indices=(0, 1, 2, 3),
        img_size=img_size 
    ).to(device)

    backbone.output_dims = backbone.feature_info.channels()
    backbone.expected_input = "B, C, H, W"
    backbone.is_3d = False

    return backbone


# not implemented
# ===========================================================
#  MODEL GENESIS  
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
def build_medical_backbone(parameters, device, method, in_channels):
    """
    Entry point for building any backbone.
    """

    name = parameters[f"{method}_model_parameters"]["backbone_str"].lower()
    pretrained_path = parameters[f"{method}_model_parameters"].get("pretrained_path", None)
    output_stride = 8 # parameters[f"{method}_model_parameters"]["backbone_stride"]
    img_size = parameters[f"{method}_model_parameters"]["input_size"]

    #2d only resnet
    if name in["resnet50d"]:
        backbone =  build_imagenet_backbone(
            name=name,
            pretrained=True,
            device=device,
            in_channels=in_channels,
            output_stride=output_stride
        )
        
        #store the configuration 
        parameters[f"{method}_model_parameters"]["backbone_index_lists"] = [    
          [0],       # f1 = C2
          [1],       # f2 = C3
          [2, 3]     # f3 = C4 + C5 combined
        ]
        #required configuration to work:        
        #c2 downsamples 4x, c3 downsample 2x 
        parameters[f"{method}_model_parameters"]['downsample'] = (True, False, False)
        parameters[f"{method}_model_parameters"]['downsample_each_repeat'] = False

        print("backbone channels: ", backbone.output_dims)
        return backbone

    if name in ["vit_base_patch16_224", "dino_vitbase16_pretrain"]:
        parameters[f"{method}_model_parameters"]["backbone_index_lists"] = [
            [0,1,2],       # f1
            [3,4,5,6],     # f2
            [7,8,9,10,11]  # f3
        ]
        #required configuration to work:
        #'downsample': (False, False, False), all output stride 16 (before first input)
        parameters[f"{method}_model_parameters"]['downsample'] = (False, False, False)
        parameters[f"{method}_model_parameters"]['channels'] =(768, 768, 768) 
        parameters[f"{method}_model_parameters"]['transformer_backbone'] = True

        backbone = build_vit_dino_backbone(
            in_channels=in_channels,
            #pretrained_path=pretrained_path,
            device=device,
            out_indices= [0,1,2,3,4,5,6,7,8,9,10,11], # want all
            img_size= img_size
        )

        return backbone
    '''
    # not implemented
    if name == "swin_unetr":
        backbone =  build_swin_unetr(
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            device=device,
            is_3d=is_3d,
        )

        return backbone

    #not implemented -----

    if name == "model_genesis":
        backbone = build_models_genesis(
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            device=device,
            is_3d=is_3d,
        )

        return backbone
    '''
    raise ValueError(f"Unknown backbone: {name}")

  

