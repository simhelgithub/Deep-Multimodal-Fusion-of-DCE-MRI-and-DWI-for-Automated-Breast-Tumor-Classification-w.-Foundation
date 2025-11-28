import torch.utils.data 
import os
import torch
import copy
import torch.nn
from torch.utils.data._utils.collate import default_collate
from dataset import *
from model_module import *


#    Runs full fusion training using prepared models + dataloaders.

def prepare_fusion_model(dwi_key, dce_key,dce_backbone, dwi_backbone, fold, parameters, device, method = 'fusion'):

    # ------ 
    # basic setup
    # ----- 
    batch_size = parameters['batch_size']
    namelist = parameters['namelist']
    mask_parameters = parameters[f"{method}_model_parameters"]['mask_parameters'] 
    class_num = parameters['class_num']
    dataloaders_dict = {}
    image_datasets = {}
    dwi_data = [None, None, None] # will contain masks if any
    dce_data = [None, None, None] 

    #---
    # load in model data and put it in image_datasets
    # ---
    model_dict = torch.load(parameters['model_dict_path'])


    for idx, split in enumerate(namelist):
        '''
        data structure loaded

        [imgs, masks, labels]
        '''

        dwi_data[idx] = load_dataset_split(
            os.path.join(parameters['data_path'], f"dwi{fold}{split}data") #methodfoldsplitdata
        )

        dce_data[idx] = load_dataset_split(
            os.path.join(parameters['data_path'], f"dce{fold}{split}data")
        )


  
    # -----
    #  Build dataloaders
    # -----

    # prepare image datasets
    for idx, split in enumerate(namelist):
      current_masks = dwi_data[idx]['masks'] if (mask_parameters['mask'] and dwi_data[idx]['masks'] is not None) else None 
      image_datasets[split] = LoadedFusionDataset(
          dwi=dwi_data[idx]['imgs'], 
          dce=dce_data[idx]['imgs'],
          masks=current_masks,
          labels=dwi_data[idx]['labels']  
      )

    #dataset to dataloder
    for idx, split in enumerate(namelist):
        dataloaders_dict[split] = torch.utils.data.DataLoader(
          image_datasets[namelist[idx]],
          batch_size=batch_size,
          shuffle=([namelist[idx]]=='train'),
          num_workers= 0,drop_last=False,
          collate_fn=custom_double_input_collate_fn
        )


  
    # -----
    #  Build models
    # -----
    dwi_params = parameters["dwi_model_parameters"] 
    dce_params = parameters["dce_model_parameters"]


    dwi_model = initialize_model(
        ModelMaskHeadBackbone(parameters['dwi_channel_num'], class_num, dwi_params['channels'], dwi_params['proj_dim'], dwi_params['enable_modality_attention'], dwi_params['use_se'], dwi_backbone),
        requires_grad=True
    )
    dwi_model.load_state_dict(model_dict[dwi_key])
    
    dce_model = initialize_model(
        ModelMaskHeadBackbone(parameters['dce_channel_num'], class_num, dce_params['channels'], dce_params['proj_dim'], dce_params['enable_modality_attention'], dce_params['use_se'], dce_backbone),
        requires_grad=True
    )
    dce_model.load_state_dict(model_dict[dce_key])
    

    # -----------------------------
    # Build Fusion Model
    # -----------------------------

    # can't just load the params if using prebaked model
    dwi_model_output_channels = infer_f3_channels(dwi_model, parameters['dwi_channel_num'], input_size=128)
  
    dce_model_output_channels = infer_f3_channels(dce_model, parameters['dce_channel_num'], input_size=128)

    fusion_params = parameters["fusion_model_parameters"]

    fusion_model = FusionModel(
        fusion_channels= fusion_params["channels"],
        proj_dim=fusion_params["proj_dim"],
        token_pool=fusion_params["token_pool"],
        use_cross_attention=fusion_params["use_cross_attention"],
        use_mask_attention=fusion_params["use_mask_attention"],
        mha_heads=fusion_params["mha_heads"],
        num_classes=class_num,
        use_se_in_fusion=fusion_params["use_se"],
        encoder_out_channels=(dwi_model_output_channels, dce_model_output_channels)
    )

    return dataloaders_dict,dwi_model, dce_model, fusion_model

def load_dataset_split(load_path):
    data = torch.load(load_path)
    return data



# assume dwi_encoder, dce_encoder already created
def infer_f3_channels(encoder, in_channels, input_size=128):
    enc_cpu = copy.deepcopy(encoder).cpu().eval()
    with torch.no_grad():
        dummy = torch.randn(1, in_channels, input_size, input_size)
        _, aux, _ = enc_cpu(dummy, None)
        recon_feats = aux['recon_feats']
        raw_feats = aux['raw_feats']
        f3 = raw_feats[-1]
        ch = f3.shape[1]
    del enc_cpu, dummy, raw_feats, recon_feats, aux
    return ch



def custom_double_input_collate_fn(batch):
    # unify into lists
    dwi_list, dce_list, mask_list, label_list = [], [], [], []
    masks_present = False

    for item in batch:
        if len(item) == 4:
            dwi, dce, mask, label = item
            dwi_list.append(dwi); dce_list.append(dce); mask_list.append(mask); label_list.append(label)
            masks_present = True
        elif len(item) == 3:
            dwi, dce, label = item
            dwi_list.append(dwi); dce_list.append(dce); label_list.append(label)
        else:
            raise RuntimeError("Dataset item must be 3 or 4 elements")

    dwi_batch = default_collate(dwi_list)
    dce_batch = default_collate(dce_list)
    labels_batch = default_collate(label_list)

    if masks_present:
        masks_batch = default_collate(mask_list)
    else:
        masks_batch = None

    return dwi_batch, dce_batch, masks_batch, labels_batch  
