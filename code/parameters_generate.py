import torch

parameters={}


# ----
# Core Training parameters
# ---
parameters['debug_training'] = True
parameters['debug_val'] = parameters['debug_training']
parameters['backbone_debug'] = False
parameters['full_debug'] = True #controls more verbose debugs



#parameters['num_epochs']=1500
#parameters['finetune_num_epochs']=20
#parameters['batch_size']=32
parameters['num_epochs']= 300
parameters['finetune_num_epochs']=2
parameters['batch_size']=32 
#parameters['input_size']=32
parameters['segnum']=5
parameters['class_num']=4
parameters['methods'] = ["dwi", "dce"]

parameters['namelist'] = ['train','val','test']




#--
# main control
#--
parameters['control_metric'] =  'val_roc_auc'
parameters['patience'] = 50

#controlled elsewhere
#parameters['version'] = 'experimental' #
#parameters['name'] = ''

parameters['save_dir'] = 'logs'

#----
# Single model specific parameters
#--


parameters["dwi_model_parameters"] = {


    # --- base model info
    'input_size':128,

    
    # ---- model structure ----
    "channels": (32, 64, 128),
    "proj_dim": 32,
    "use_se": False,
    'grad_clip': 3,
    'gradient_clip_algorithm':'norm', #alt, "norm" "value"
  
    # ---- model features
    "enable_modality_attention": True,
    "backbone_str": 'resnet50',
    
    'grad_clip':5.0,
    # ---- label smoothing ----
    "label_smoothing_enabled": True,
    "label_smoothing_alpha": 0.1,

    # ---- mimic loss ----
    "mimic_enabled": True,
    "lambda_mimic": 0.4,

    # ---- reconstruction loss ----
    "recon_enabled": True,
    "reconstruction_loss_code": "mse",  
    "lambda_recon": 0.2,

    # ---- classification loss
    "classification_loss_parameters": {
      "classification_loss_code":"wfl",          # or 'fl'
      "gamma": 2.0,
      "alpha": None #will be calculated with wfl
    },
    # --- Mask parameters
    'mask_parameters' : {
      "mask": True,   #mask enabled for custom models
      "lambda_mask": 0.4,
      "mask_loss_type": "dice",
      "mask_target_size": (32, 32),
      "mask_fusion_attention": True
    },
    # --- optimizer parameters
    'optimizer_parameters' : {
      'name':"adamW", #or adam
      "lr": 5e-4,
      #"lr":0.001,
      "betas":(0.9, 0.999),
      "eps":1e-08,
      #"weight_decay":0.0001,
      "amsgrad":False,
      "num_lr_groups": 3,
      "discriminative_lr": True,
      "lr_decay_factor": 2,
      "weight_decay": 0.001,
      "discrim_on": "all", #options: all, backbone, non_backbone
      "discriminative_reg": True,      
      "reg_decay_factor": 2,     
      "reg_base": 1e-4
    },
    'scheduler' : {
      'name' :"reduce_lr_on_plateau", #alts "cosine", "cosine_with_warmup", "reduce_lr_on_plateau"

      #for reduce_lr_on_plateau
      'factor': 0.1,
      'patience': parameters['patience'], 
      'min_lr': 1e-7,
      'threshold': 1e-4,
      'monitor': parameters['control_metric'],

      #for cosine & cosine_with_warmup
      'T_max' : parameters['num_epochs'],
      'eta_min' : 0,
      'warmup_steps' : 500,
      'max_steps' : 10000

    },
    # regularization 
    "attn_reg_enabled": True,
    "lambda_attn_sparsity": 1e-4,
    "lambda_attn_consistency": 1e-4,
    "feat_norm_reg_enabled": True,
    "lambda_feat_norm": 1e-4,


}      
'''
      "lr_multipliers": {
          "backbone_l1": 0.1,
          "backbone_l2": 0.25,
          "backbone_l3": 0.5,
          "heads": 1.0
      },
      "wd_multipliers": {
        "backbone_l1": 0.2,
        "backbone_l2": 0.5,
        "backbone_l3": 1.0,
        "heads": 1.0
      }
'''
#just the same for now
parameters["dce_model_parameters"] = parameters["dwi_model_parameters"] 




#----
# Fusion model parameters
#---

parameters["fusion_model_parameters"] = {
    
    # ---- model structure ----
    "channels": 128,
    "proj_dim": 32,
    "use_se": False,
    'fusion_channels':128,
    'token_pool': (4,4),
    "use_cross_attention":True,
    "use_mask_attention":True,
    "mha_heads": 4,
    'grad_clip': 3,
    'gradient_clip_algorithm':'norm', #alt, "norm" "value"

    # ---- label smoothing ----
    "label_smoothing_enabled": True,
    "label_smoothing_alpha": 0.1,
    # ---- reconstruction loss ----
    "recon_enabled": True,
    "reconstruction_loss_code": "mse",  
    "lambda_recon": 0.2,
    # ---- optimizer parameters ----
    'optimizer_parameters' : {
      'name':"adamW", #or adam
      "lr":0.001,
      "betas":(0.9, 0.999),
      "eps":1e-08,
      "weight_decay":0.0001,
      "amsgrad":False,
      "num_lr_groups": 3,
      "lr_decay_factor": 2,
    },

    # ---- classification loss
    "classification_loss_parameters": {
      "classification_loss_code":"wfl",          # or 'fl'
      "gamma": 2.0,
      "alpha": None #will be calculated with wfl
    },

    # --- mask parameters
    'mask_parameters' : {
      "mask": True,   #mask enabled for custom models
      "lambda_mask": 1.0,
      "mask_loss_type": "dice",
      "mask_target_size": (32, 32),
      "mask_fusion_attention": True
    },
    'scheduler' : {
      'name' :"reduce_lr_on_plateau", #alts "cosine", "cosine_with_warmup", "reduce_lr_on_plateau"

    },
    # --- do finetune?
    'finetune' : True
}


#---
# Early stopping params
#--
parameters['early_stopping_parameters'] = {
    'metric': parameters['control_metric'],
    'mode': 'max',
    'patience': parameters['patience'], 
    'min_delta': 0.001
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
parameters['unfreeze_timer'] = 30

parameters['backbone_unfreeze_lr'] = parameters["dwi_model_parameters"]['optimizer_parameters']['lr'] * 0.1
   
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
parameters['min_epochs'] =  max(parameters['patience']*3, parameters['num_epochs'])

if parameters['backbone_freeze_on_start']:
  parameters['min_epochs'] = max(parameters['min_epochs'], parameters['unfreeze_timer']*(parameters['backbone_num_groups'] +1))

if parameters['use_simple_aux_loss_scheduling']:
  parameters['min_epochs']  = max(parameters['min_epochs'], parameters["aux_loss_weight_epoch_limit"]+1)
parameters['min_epochs'] =  max(parameters['min_epochs'], parameters['num_epochs']/4) #min 1/4 of max epochs



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

