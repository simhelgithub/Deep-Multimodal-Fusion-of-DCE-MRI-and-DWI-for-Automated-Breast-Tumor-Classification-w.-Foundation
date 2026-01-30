import torch

parameters={}

# -- 
# 3d params
# ---

parameters['dim']  = 2

#---
# compile & optimization 
#--
parameters['compile'] = True
parameters['dataloader_num_workers'] = 11 #t4 optimal?

# ----
# Core Training parameters
# ---
parameters['debug_training'] = True
parameters['debug_val'] =parameters['debug_training']
parameters['backbone_debug'] = False
parameters['full_debug'] = False #controls more verbose debugs
parameters["debug_anomaly"] = False


#parameters['num_epochs']=1500
#parameters['finetune_num_epochs']=20
#parameters['batch_size']=32
parameters['num_epochs']= 900
#parameters['finetune_num_epochs']=20 #no longer needed
parameters['batch_size']=32 
#parameters['input_size']=32
parameters['segnum']=5
parameters['class_num']=4
parameters['methods'] = ["dwi", "dce"]

parameters['namelist'] = ['train','val','test']




#--
# main control
#--
parameters['control_metric'] =  'val_loss'
#parameters['control_metric'] =  'val_roc_auc'
parameters['early_stop_metric'] = 'val_roc_auc'

parameters['patience'] = 90

#controlled elsewhere
#parameters['version'] = 'experimental' #
#parameters['name'] = ''

parameters['save_dir'] = 'logs'

#----
# Single model specific parameters
#--
parameters['forced_mask_size'] = 32


parameters["dwi_model_parameters"] = {


    # --- base model info
    'input_size': 256, #128, #currently must be 128 for mask

    #--- tranfromer model 
    'use_hybrid_transformer': False, #selects to use hybrid dce/dwi transformer
    'transformer_heads': 4,
    'transformer_patch_size': 2, 
    'transformer_depth': 6,
    'transformer_embed_dim': 512, #must be same as mid channels
    # ---
    'dropout': 0.2,

    # ---- model structure (block1->block2->block3) ----
    #"channels": (32, 64, 128), #underfit
    #"channels": (64, 128, 256), #underfit
    "channels": (128, 256, 512),
    'repeat_blocks': (1,1,1), #repeat block count on different depths, block1,block2,block3 1 for off
    'downsample': (True, False, False),
    'downsample_each_repeat':False,
    'mid_squeeze': 2, #2 or 4 reasonable
    #"backbone_stride": 8,
    "backbone_index_lists": [], # set when creating the model automatically setup for each
    "backbone_out_channels": (), # set when creating the model automatically setup for each
    "proj_dim": 64,
    "use_se": True,
    'grad_clip': 5.0,
    'gradient_clip_algorithm':'norm', #alt, "norm" "value"
  
    # ---- model features
    "enable_modality_attention": True,
    "use_backbone": True,
    "use_input_adapt": False, #only for resnet and resnet50d
    "use_advanced_adapt": False, #use a more complex backbone channel adpation
    "transformer_backbone": False,
    "backbone_str": 'radimagenet', #options resnet50d, resnet50, vit_base_patch16_224, dino_vitbase16_pretrain, radimagenet
    # ---- label smoothing ----
    "label_smoothing_enabled": True,
    "label_smoothing_alpha": 0.1,

    # ---- mimic loss ----
    "mimic_enabled": True,
    "lambda_mimic": 0.2,

    # ---- reconstruction loss ----
    "recon_enabled": True,
    "reconstruction_loss_code": "mse",  
    "lambda_recon": 0.1,

    # ---- classification loss
    "classification_loss_parameters": {
      "classification_loss_code":"wfl",          # or 'fl'
      "gamma": 1.5,
      "alpha": None #will be calculated with wfl
    },
    # --- Mask parameters
    'mask_parameters' : {
      "mask": True,   #mask enabled for custom models 
      "mask_stage": "f2", #"f2", #f1,f2,f3 avaialbe
      "lambda_mask": 0.2,  
      "mask_loss_type": "dice", #dice_bce or dice
      "mask_target_size": (32, 32),
      "mask_fusion_attention": True,
      "dice_weight": 0.5, #dice_bce only
      "bce_weight": 0.5 #dice_bce only
    },
    # --- optimizer parameters
    'optimizer_parameters' : {
      'name':"adamW", #or adam
      "lr": 1e-4,
      "betas":(0.9, 0.999),
      "eps":1e-08,
      "amsgrad":False,
      "weight_decay": 4e-5, 
      "num_lr_groups": 3,
      "discriminative_lr": True,
      "lr_decay_factor": 1.2,
      "discrim_on": "all", #options: all, backbone, non_backbone
      "discriminative_reg": True,      
      "reg_decay_factor": 0.8,     
      "reg_base": 1e-4
    },
    'scheduler' : {
      'name' :"reduce_lr_on_plateau", #alts "cosine", "cosine_with_warmup", "reduce_lr_on_plateau"

      #for reduce_lr_on_plateau
      'factor': 0.5,
      'patience': int(5+ parameters['patience']/3), 
      'min_lr': 4e-7,
      'threshold': 0.0001,
      'monitor': parameters['control_metric'],

      #for cosine & cosine_with_warmup
      'T_max' : parameters['num_epochs'],
      'eta_min' : 0,
      'warmup_steps' : 500,
      'max_steps' : 10000

    },
    # regularization 
    "attn_reg_enabled": False,
    "lambda_attn_energy": 1e-4, 
    "lambda_feature_consistency": 1e-4,
    "feat_norm_reg_enabled": True,
    "lambda_feat_norm": 4e-5,
}      

#just the same for now
parameters["dce_model_parameters"] = parameters["dwi_model_parameters"] 




#----
# Fusion model parameters
#---

parameters["fusion_model_parameters"]  = parameters["dwi_model_parameters"] # for now
#extra parameters
parameters['fusion_model_parameters']['fusion_specific_parameters'] = {
    "mha_heads": 4,
    "use_cross_attention": True,
    "use_mask_attention": True,
    'token_pool': (4,4),
    'fusion_channels': 128,
    'dwi_out_channels': parameters["dwi_model_parameters"]['channels'][-1], # last dwi channel out
    'dce_out_channels': parameters["dce_model_parameters"]['channels'][-1],
    'fusion_recon_ch': 1 
}

#---
# Early stopping params
#--
parameters['early_stopping_parameters'] = {
    'metric': parameters['early_stop_metric'], #parameters['control_metric'],
    'mode': 'max',
    'patience': parameters['patience'], 
    'min_delta': 0.0001
}



#--
# AMP
#---
parameters['precision'] = "16-mixed" # "16-mixed" "bf16-mixed"
#---
# TTA and mc parameters
#-- 
parameters["test_mode"] = "tta_mc"   # or: "normal", "tta", "mc"
parameters['mc_passes'] = 10 

#--
# Gradual unfreezing
#--
parameters['backbone_freeze_on_start'] = True
parameters['backbone_num_groups'] = 3 #how many parts to split the backbone in for gradual unfreezing
parameters['unfreeze_timer'] = 40
parameters['foundation_model_unfreeze_timer'] = 40
parameters['backbone_unfreeze_lr'] = parameters["dwi_model_parameters"]['optimizer_parameters']['lr'] * 0.1
parameters['backbone_unfreeze_wd'] = parameters["dwi_model_parameters"]['optimizer_parameters']['reg_base'] * 0.1
parameters['foundation_model_unfreeze_lr'] = 1e-5
parameters['backbone_unfreeze_lr_factor']=  0.25

# Aux-Loss Scheduling
#---
parameters['use_simple_aux_loss_scheduling'] = True
parameters["aux_loss_weight_epoch_limit"] = max(100 , parameters['unfreeze_timer']*(parameters['backbone_num_groups'] +2))




#---
# input data specific params
#---
parameters['dwi_bvals_to_use'] = (0,1,2,3,4,5,6,7,8,9,10,11,12) #todo actual bval value mapping + implement 
parameters['dce_channels_to_use'] = (0,1,2,3,4,5) #todo time mapping + implement


#calculate input data channel params
parameters['dwi_add_adc_map'] = True #false not implemented
parameters['dwi_base_channel_num']= len(parameters['dwi_bvals_to_use'])
parameters['dwi_channel_num'] =parameters['dwi_base_channel_num']
if parameters['dwi_add_adc_map']: parameters['dwi_channel_num']+=1

parameters['dce_channel_num']= len(parameters['dce_channels_to_use'])

#min epochs
parameters['min_epochs'] =  parameters['patience']*3

if parameters['backbone_freeze_on_start']:
  parameters['min_epochs'] = max(parameters['min_epochs'], parameters['unfreeze_timer']*(parameters['backbone_num_groups'] +1))

if parameters['use_simple_aux_loss_scheduling']:
  parameters['min_epochs']  = max(parameters['min_epochs'], parameters["aux_loss_weight_epoch_limit"]+1)
parameters['min_epochs'] =  max(parameters['min_epochs'], parameters['num_epochs']/3) #min 1/3 of max epochs



# ----
# paths 
# ----
base_path = r'/content/drive/My Drive/master/DWI_DCE_CDFR-DNN_-main/archive/'

parameters['base_path'] = base_path
parameters['masks_path'] = base_path + 'masks/mask.pth' 

parameters['nyul_path'] = base_path + "nyul_landmarks.npy"
parameters['data_path'] = base_path + "data/" 

parameters['model_dict_path'] =  base_path + "model_dict/model_dict.pth"



parameters['dwi_tensordata']= base_path + 'dwi_tensordata/dwi_tensordata.pth'
parameters['dce_tensordata']= base_path + 'dce_tensordata/dce_tensordata.pth'
parameters['labels_tensordata']= base_path + 'labels_tensordata/labels_tensordata.pth'

parameters['dwi_test_tensordata']=base_path + 'dwi_test_tensordata/dwi_test_tensordata.pth'
parameters['dce_test_tensordata']= base_path + 'dce_test_tensordata/dce_test_tensordata.pth'
parameters['labels_test_tensordata']= base_path + 'labels_test_tensordata/labels_test_tensordata.pth'

parameters['fusion_model_dict'] = base_path + "fusion_model_dict/fusion_model_dict.pth"




#---
# basic setup 
#---
parameters["data_key_mod"] = "data"  # just added after keys to notify when its the data from dataloader

model_dict={}
torch.save(model_dict, parameters['model_dict_path'])
fusion_model_dict={}
torch.save(fusion_model_dict, parameters['fusion_model_dict'])

torch.save(parameters,base_path + "parameters/parameters.pth")

