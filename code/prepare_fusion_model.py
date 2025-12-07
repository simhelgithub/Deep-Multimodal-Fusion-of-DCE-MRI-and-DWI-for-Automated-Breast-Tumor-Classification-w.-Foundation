import torch.utils.data 
import os
import torch
import copy
import torch.nn
from torch.utils.data._utils.collate import default_collate
from dataset import *
from model_module import *


#    Runs full fusion training using prepared models + dataloaders.

def prepare_fusion_model(dwi_results, dce_results,fold, parameters, device, method = 'fusion'):

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
    dwi_model = dwi_results["trained_model"]
    dce_model = dce_results["trained_model"]

    # -----------------------------
    # Build Fusion Model
    # -----------------------------
    fusion_model = FusionModel(
        parameters
    )
    return dataloaders_dict,dwi_model, dce_model, fusion_model

def load_dataset_split(load_path):
    data = torch.load(load_path)
    return data



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
