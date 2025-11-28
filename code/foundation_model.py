import torch
import torch.nn as nn
import torchvision.models as tvmodels
import importlib
import warnings
import os
import torchvision.models as models

def try_import_monai():
    try:
        import monai
        return monai
    except Exception:
        return None

MONAI = try_import_monai()

def build_imagenet_backbone(name='resnet50', pretrained=True, device='cuda'):
  pass


#todo implement
def build_swin_unetr(backbone_name='swin_unetr', pretrained_path=None, device='cuda'):
    if MONAI is None:
        raise RuntimeError("MONAI not installed. ")
    from monai.networks.nets import SwinUNETR
    '''
    model = SwinUNETR(
        img_size=(128,128,128) if False else (128,128), # choose 2D or 3D at your preference
        in_channels=1,  # override as needed
        out_channels=1,
        feature_size=48,
        depths=[2,2,2,2],
        num_heads=[3,6,12,24],
        use_checkpoint=False
    )
    '''
    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state, strict=False)
    return model.to(device)

def build_models_genesis(pretrained_path=None, backbone='unet', device='cuda'):

  pass

#builds medical bacbone current optiosn: resnet50
def build_medical_backbone(parameters, device, method, in_channels):
    name = parameters[f"{method}_model_parameters"]["backbone_str"]   

    if name == "resnet50":
        
        m = models.resnet50(pretrained=True)

        #it usually wants 3 input channels, change to mulit input

        # Fix first conv for multi-channel input
        old_conv = m.conv1

        m.conv1 = nn.Conv2d(
            in_channels,          # <--- 6 or 14
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # Initialize new channels from pretrained weights
        if in_channels > 3:
            # copy RGB weights
            m.conv1.weight.data[:, :3] = old_conv.weight.data
            # average for remaining channels
            m.conv1.weight.data[:, 3:] = old_conv.weight.data.mean(dim=1, keepdim=True)
        elif in_channels < 3:
            # duplicate channels if fewer than 3
            repeat = 3 // in_channels
            m.conv1.weight.data = old_conv.weight.data.repeat(1, repeat, 1, 1)

        # remove classifier but keep spatial output
        m = nn.Sequential(*list(m.children())[:-2])

        m.output_dim = 2048   # channels of last feature map

        return m.to(device)
