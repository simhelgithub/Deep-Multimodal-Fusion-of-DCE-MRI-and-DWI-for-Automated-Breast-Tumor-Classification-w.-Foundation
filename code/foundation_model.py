import torch
import torch.nn as nn
import torchvision.models as models

import timm
import os
import re
from collections import defaultdict


# ===========================================================
#  TIMM RESNET BACKBONE WITH OUTPUT_STRIDE=8, 2d only resnet50d option
# ===========================================================

def build_imagenet_backbone(
    name="resnet50d",
    pretrained=True,
    device="cuda",
    in_channels=6,
    output_stride=8,
    use_advanced_adapt= False,
    skip_adapt = True

):
    os.environ["TIMM_MODEL_CACHE"] = "/mnt/models/timm"

    # ---- Load pretrained RGB backbone ----
    backbone_rgb = timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=3,
        features_only=True,
        output_stride=output_stride,
        out_indices=(1, 2, 3, 4),
    )

    # ---- Adapt first convolution weights ----
    state_dict = backbone_rgb.state_dict()
    # Adapt conv1 to in_channels
    if not skip_adapt: 
        if use_advanced_adapt: 
          state_dict = advanced_adapt_first_conv(state_dict, in_channels) 
        else:
          state_dict = adapt_first_conv(state_dict, in_channels) 


        # ----  Rebuild backbone with correct input channels ----
        backbone = timm.create_model(
            name,
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            out_indices=(1, 2, 3, 4),
        )
        backbone.load_state_dict(state_dict, strict=False)
        backbone = backbone.to(device)

    else:
        backbone = backbone_rgb


    # ---- Attach metadata ----
    backbone.output_dims = backbone.feature_info.channels()
    backbone.expected_input = "B, C, H, W"
    backbone.is_3d = False

    return backbone
#---
# radimagenet
#--
from huggingface_hub import hf_hub_download

def download_radimagenet_weights(
    name: str,
    cache_dir: str = "/mnt/models/radimagenet",
):
    """
    Downloads RadImageNet weights from Hugging Face.
    """

    assert name in ["resnet50", "resnet101"]

    filename = {
        "resnet50": "ResNet50.pt",
        "resnet101": "ResNet101.pt",
    }[name]

    os.makedirs(cache_dir, exist_ok=True)

    path = hf_hub_download(
        repo_id="Lab-Rasool/RadImageNet",
        filename=filename,
        cache_dir=cache_dir,
    )

    return path

def adapt_first_conv(state_dict, in_channels):
    """
    Adapt the first conv to any number of input channels.
    """
    # Find conv1 key
    key_candidates = ["conv1.weight", "encoder.conv1.weight", "module.conv1.weight"]
    conv_key = None
    for k in key_candidates:
        if k in state_dict:
            conv_key = k
            break

    if conv_key is None:
        return state_dict

    w = state_dict[conv_key]  # [out, in_ch, k, k]

    if w.shape[1] == in_channels:
        return state_dict

    # Average existing channels
    w_mean = w.mean(dim=1, keepdim=True)  # [out,1,k,k]
    w_new = w_mean.repeat(1, in_channels, 1, 1)

    state_dict[conv_key] = w_new
    return state_dict


 
def advanced_adapt_first_conv(state_dict, in_channels, eps=0.05):
    """
    Adapt first convolution for multi-channel grayscale medical inputs
    (e.g. time series or b-values).

    Strategy:
    - Convert RGB filters to luminance-equivalent filters
    - Replicate for all channels
    - Apply small deterministic per-channel scaling to break symmetry
    """
    conv_keys = [k for k in state_dict if k.endswith(".weight") and state_dict[k].dim() == 4]
    if not conv_keys:
        return state_dict

    # Heuristic: first conv has smallest input channel count
    conv_key = min(conv_keys, key=lambda k: state_dict[k].shape[1])
    w = state_dict[conv_key]  # [out, in_ch, k, k]

    if w.shape[1] == in_channels:
        return state_dict

    out_c, old_in, kh, kw = w.shape
    device = w.device
    dtype = w.dtype

    with torch.no_grad():
        # Convert RGB → luminance if possible
        if old_in >= 3:
            # ITU-R BT.601 luminance coefficients
            lum = (
                0.2989 * w[:, 0:1]
                + 0.5870 * w[:, 1:2]
                + 0.1140 * w[:, 2:3]
            )
        else:
            lum = w.mean(dim=1, keepdim=True)

        # Base weights replicated across channels
        w_new = lum.repeat(1, in_channels, 1, 1)

        # Deterministic per-channel scaling (monotonic)
        scales = torch.linspace(
            1.0 - eps, 1.0 + eps, in_channels, device=device, dtype=dtype
        ).view(1, in_channels, 1, 1)

        w_new *= scales

    state_dict[conv_key] = w_new
    return state_dict



def map_rasool_to_timm_keys(rasool_state_dict):
    """
    Maps Rasool ResNet50 keys -> TIMM resnet50 keys
    """

    mapped = {}

    layer_map = {
        '4': 'layer1',
        '5': 'layer2',
        '6': 'layer3',
        '7': 'layer4',
    }

    for k, v in rasool_state_dict.items():
        # Remove any prefix
        new_key = k
        if k.startswith("backbone."):
            new_key = k[len("backbone."):]

        # Stem conv/bn
        if new_key == "0.weight":
            new_key = "conv1.weight"
        elif new_key.startswith("1."):
            # bn1.* in TIMM
            new_key = "bn1." + new_key[2:]
        # Residual layers
        elif new_key[0] in layer_map and new_key[1] == '.':
            timm_layer = layer_map[new_key[0]]
            rest = new_key[2:]  # e.g., '0.conv1.weight'
            new_key = f"{timm_layer}.{rest}"

        # Drop classifier
        if new_key.startswith("fc."):
            continue

        mapped[new_key] = v

    return mapped

def build_radimagenet_backbone(
    name="resnet50",                 
    device="cuda",
    in_channels=6,
    output_stride=8,
    out_indices=(1, 2, 3, 4),         # C2–C5
    use_advanced_adapt = True
):


    assert name in ["resnet50", "resnet101"], \
        "RadImageNet supports vanilla ResNet50 / ResNet101 only"

    os.environ["TIMM_MODEL_CACHE"] = "/mnt/models/timm"

    # -------------------------------------------------------
    # Download RadImageNet checkpoint
    # -------------------------------------------------------
    pretrained_path = download_radimagenet_weights(name)

    print(f"[RadImageNet] Using checkpoint: {pretrained_path}")

    ckpt = torch.load(pretrained_path, map_location="cpu")
    print(list(ckpt.keys())[:20])

    # -------------------------------------------------------
    # Extract state_dict from HF checkpoint
    # -------------------------------------------------------
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "encoder"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise RuntimeError("[RadImageNet] Invalid checkpoint format")

    # -------------------------------------------------------
    # Build TIMM backbone (NO pretrained)
    # -------------------------------------------------------
    backbone = timm.create_model(
        name,
        pretrained=False,
        in_chans=in_channels,
        features_only=True,
        output_stride=output_stride,
        out_indices=out_indices,
    ).to(device)

    model_state = backbone.state_dict()
    cleaned = {}

    # -------------------------------------------------------
    # Clean & adapt weights
    # -------------------------------------------------------
    ckpt = map_rasool_to_timm_keys(ckpt)

    # Adapt conv1 to in_channels
    if use_advanced_adapt: 
      ckpt = advanced_adapt_first_conv(ckpt, in_channels) 
    else:
      ckpt = adapt_first_conv(ckpt, in_channels) 

    # Load into backbone
    model_state = backbone.state_dict()
    cleaned = {}
    for k, v in ckpt.items():
        if k.startswith("fc."):  # skip classifier
            continue
        if k in model_state and model_state[k].shape == v.shape:
            cleaned[k] = v

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    print(f"[RadImageNet] Loaded {len(cleaned)} tensors | Missing: {len(missing)} | Unexpected: {len(unexpected)}")


    
    if len(cleaned) < 100:
        raise RuntimeError(
            "[RadImageNet] Too few weights loaded — "
            "likely wrong architecture or incompatible checkpoint"
        )

    # -------------------------------------------------------
    # Metadata (match ImageNet builder)
    # -------------------------------------------------------
    backbone.output_dims = backbone.feature_info.channels()
    backbone.expected_input = "B, C, H, W"
    backbone.is_3d = False
    backbone.foundation_model = True
    backbone.transformer_backbone = False

    return backbone




# -----------------------
# vit dino 2d only, 2 options  vit_base_patch16_224, dino_vitbase16_pretrain
# ------------------------

def adapt_vit_patch_embed(state_dict, in_channels, eps=0.02):
    """
    Adapt ViT / DINO patch embedding weights for multi-channel grayscale inputs.

    Strategy:
    - Preserve pretrained RGB projection
    - Initialize extra channels with mean projection
    - Apply small deterministic scaling to break symmetry
    """
    # Common ViT patch embed keys
    key_candidates = [
        "patch_embed.proj.weight",
        "patch_embed.backbone.proj.weight",  # some DINO variants
    ]

    pe_key = None
    for k in key_candidates:
        if k in state_dict:
            pe_key = k
            break

    if pe_key is None:
        return state_dict

    w = state_dict[pe_key]  # [embed_dim, in_ch, p, p]

    if w.shape[1] == in_channels:
        return state_dict

    embed_dim, old_in, ph, pw = w.shape
    device, dtype = w.device, w.dtype

    with torch.no_grad():
        # Mean projection across channels
        w_mean = w.mean(dim=1, keepdim=True)

        # Replicate for all channels
        w_new = w_mean.repeat(1, in_channels, 1, 1)

        # Small deterministic per-channel scaling
        scales = torch.linspace(
            1.0 - eps, 1.0 + eps, in_channels,
            device=device, dtype=dtype
        ).view(1, in_channels, 1, 1)

        w_new *= scales

    state_dict[pe_key] = w_new
    return state_dict

def build_vit_dino_backbone(
    name="vit_base_patch16_224",
    pretrained=True,
    device="cuda",
    in_channels=6,
    out_indices=None,
    img_size=256,
    use_advanced_adapt = False

):
    os.environ["TIMM_MODEL_CACHE"] = "/mnt/models/timm"

    if out_indices is None:
        out_indices = list(range(12))


    if use_advanced_adapt:
      vit_rgb = timm.create_model(
          name,
          pretrained=pretrained,
          in_chans=3,
          features_only=True,
          out_indices=out_indices,
          img_size=img_size,
      )
      state_dict = vit_rgb.state_dict()
      # Adapt patch embedding
      if in_channels != 3:
          state_dict = adapt_vit_patch_embed(state_dict, in_channels)

      # Rebuild with correct input channels
      vit = timm.create_model(
          name,
          pretrained=False,
          in_chans=in_channels,
          features_only=True,
          out_indices=out_indices,
          img_size=img_size,
      )
    else:
        # Load RGB pretrained ViT
        vit_rgb = timm.create_model(
            name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=out_indices,
            img_size=img_size,
        )
        state_dict = vit_rgb.state_dict()

        vit = vit_rgb

    vit.load_state_dict(state_dict, strict=False)
    vit = vit.to(device)

    vit.output_dims = vit.feature_info.channels()
    vit.expected_input = "B, C, H, W"
    vit.is_3d = False

    return vit


# -----------------------
# UNI2-h pathology foundation model (2D ViT) not implemented
# -----------------------
def build_uni2h_backbone(
    name="hf-hub:MahmoodLab/UNI2-h",
    pretrained=True,
    device="cuda",
    in_channels=3,
    out_indices=None,
    img_size=224,
):
    """
    Build UNI2-h backbone using timm, returning transformer block features.

    Args:
        name (str): HF hub timm-compatible model id
        pretrained (bool): Load pretrained weights
        device (str): 'cuda' or 'cpu'
        in_channels (int): Number of input channels (usually 3)
        out_indices (list[int]): Transformer block indices to return
        img_size (int): Input resolution
        freeze (bool): Freeze backbone weights

    Returns:
        backbone: timm model with metadata
    """

    os.environ["TIMM_MODEL_CACHE"] = "/mnt/models/timm"

    # UNI2-h has 24 transformer blocks
    if out_indices is None:
        out_indices = list(range(24))

    backbone = timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=in_channels,
        features_only=True,
        out_indices=out_indices,
        img_size=img_size,
    ).to(device)

    # Metadata (match your existing pattern)
    backbone.output_dims = backbone.feature_info.channels()
    backbone.expected_input = "B, C, H, W"
    backbone.is_3d = False
    backbone.transformer_backbone = True
    backbone.foundation_model = True

    return backbone



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
    use_advanced_adapt = parameters[f"{method}_model_parameters"]["use_advanced_adapt"]

    #2d only resnet
    if name in["resnet50d", "resnet50"]:
        backbone =  build_imagenet_backbone(
            name=name,
            pretrained=True,
            device=device,
            in_channels=in_channels,
            output_stride=output_stride,
            use_advanced_adapt=use_advanced_adapt,
            skip_adapt =  parameters[f"{method}_model_parameters"]["use_input_adapt"]
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
    elif name in ["vit_base_patch16_224", "dino_vitbase16_pretrain"]:
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
            img_size= img_size,
            use_advanced_adapt=use_advanced_adapt
        )
    elif name in ["radimagenet", "radimagenet_resnet50"]:

        backbone = build_radimagenet_backbone(
            name="resnet50",
            #pretrained_path=pretrained_path,
            device=device,
            in_channels=in_channels,
            output_stride=output_stride,
            out_indices=(1, 2, 3, 4),
            use_advanced_adapt=use_advanced_adapt
        )

        # Feature grouping (matches spatial resolutions)
        parameters[f"{method}_model_parameters"]["backbone_index_lists"] = [
            [0],      # C2
            [1],      # C3
            [2, 3],   # C4 + C5
        ]

        # Spatial behavior
        parameters[f"{method}_model_parameters"]["downsample"] = (True, False, False)
        parameters[f"{method}_model_parameters"]["downsample_each_repeat"] = False
        #parameters[f"{method}_model_parameters"]["channels"] = backbone.output_dims # [256, 512, 1024, 2048] default

        print("RadImageNet backbone channels:", backbone.output_dims)
        return backbone
 
    return backbone


  

