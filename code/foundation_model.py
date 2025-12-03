
import torch
import torch.nn as nn
import torchvision.models as models

# --------------------
# MONAI import helper
# ----------------------
def try_import_monai():
    try:
        import monai
        return monai
    except Exception:
        return None

MONAI = try_import_monai()


# =================
#  AUTO 3D INPUT WRAPPER
# =================
class Auto3DInput(nn.Module):
    """
    Allows 3D backbones to accept (B, C, H, W) like 2D models.
    Converts to (B, 1, D=C, H, W) internally.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        if x.ndim == 4:
            # Input is (B, slices, H, W) → (B, 1, D, H, W)
            x = x.unsqueeze(1)
        return self.backbone(x)


# =====================
#  IMAGENET RESNET (2D ONLY)
# =====================
def build_imagenet_backbone(name='resnet50', pretrained=True, device='cuda',
                            in_channels=3, is_3d=False):
    """
    Loads a 2D ResNet and returns a spatial feature extractor.
    """

    if is_3d:
        raise ValueError("ResNet is 2D-only. Use is_3d=False.")

    if name.lower() != "resnet50":
        raise ValueError(f"Unsupported imagenet backbone: {name}")

    m = models.resnet50(pretrained=pretrained)

    # Fix 1st conv for arbitrary in_channels
    old_conv = m.conv1
    m.conv1 = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    # Weight init logic
    if in_channels >= 3:
        m.conv1.weight.data[:, :3] = old_conv.weight
        if in_channels > 3:
            m.conv1.weight.data[:, 3:] = old_conv.weight.mean(1, keepdim=True)
    else:
        repeat = 3 // in_channels
        m.conv1.weight.data = old_conv.weight[:, :in_channels].repeat(1, repeat, 1, 1)

    # Remove classifier
    m = nn.Sequential(*list(m.children())[:-2])
    m.output_dim = 2048
    m.is_3d = False
    m.expected_input = "B, C, H, W"
    return m.to(device)


# ===================
#  SWIN-UNETR (2D + 3D)
# ===================
def build_swin_unetr(in_channels=1, pretrained_path=None,
                     device='cuda', is_3d=False):

    if MONAI is None:
        raise RuntimeError("MONAI not installed. pip install monai")

    from monai.networks.nets import SwinUNETR

    if is_3d:
        model = SwinUNETR(
            img_size=(64, 128, 128),    # Example depth/height/width
            spatial_dims=3,
            in_channels=1,              # Slices → depth, so always 1
            out_channels=1,
            feature_size=48,
        )
        model.is_3d = True
        model.expected_input = "B, 1, D, H, W"
        model.output_dim = 48

        # Wrap so it can accept (B, C, H, W)
        model = Auto3DInput(model)

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

    # Load checkpoint
    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu')
        model.backbone.load_state_dict(state, strict=False) if is_3d else model.load_state_dict(state, strict=False)

    return model.to(device)


# ======================
#  MODEL GENESIS (2D + 3D)
# ======================
def build_models_genesis(in_channels=1, pretrained_path=None,
                         device='cuda', is_3d=False):

    if MONAI is None:
        raise RuntimeError("MONAI not installed. pip install monai")

    from monai.networks.nets import UNet

    if is_3d:
        model = UNet(
            spatial_dims=3,
            in_channels=1,  # slices → depth
            out_channels=1,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
        )
        output_dim = 256
        is_wrapper_needed = True

    else:
        model = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
        )
        output_dim = 512
        is_wrapper_needed = False

    # Load checkpoint
    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state, strict=False)

    # Use only encoder stages if available
    encoder = nn.Sequential(*model.encoder) if hasattr(model, "encoder") else model

    encoder.output_dim = output_dim
    encoder.is_3d = is_3d
    encoder.expected_input = "B, 1, D, H, W" if is_3d else "B, C, H, W"

    if is_wrapper_needed:
        encoder = Auto3DInput(encoder)

    return encoder.to(device)


# ============
#  Builder to be called fro mthe outside
# ==========
def build_medical_backbone(parameters, device, method, in_channels, is_3d=False):
    """
    Main entrypoint.

    is_3d=True  → input is (B, slices, H, W) → internally converted to (B, 1, D, H, W)
    is_3d=False → input is (B, C=slices, H, W)
    """

    name = parameters[f"{method}_model_parameters"]["backbone_str"].lower()
    pretrained_path = parameters[f"{method}_model_parameters"].get("pretrained_path", None)

    # --------------------------
    # 2D IMAGENET BACKBONES
    # --------------------------
    if name == ["resnet50"]:
        return build_imagenet_backbone(
            name="resnet50",
            pretrained=True,
            device=device,
            in_channels=in_channels,
            is_3d=is_3d
        )

    # --------------------------
    # SWIN UNETR
    # --------------------------
    if name == ["swin_unetr"]:
        return build_swin_unetr(
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            device=device,
            is_3d=is_3d
        )

    # --------------------------
    # MODEL GENESIS (UNet encoder)
    # --------------------------
    if name == ["model_genesis"]:
        return build_models_genesis(
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            device=device,
            is_3d=is_3d
        )

    raise ValueError(f"Unknown backbone name: {name}")
