import torch

parameters={}


# ----
# Core Training parameters
# ---
parameters['debug_training'] = True


#parameters['num_epochs']=1500
#parameters['finetune_num_epochs']=20
#parameters['batch_size']=32
parameters['num_epochs']= 2
parameters['finetune_num_epochs']=2
parameters['batch_size']=64 
#parameters['input_size']=32
parameters['segnum']=5
parameters['class_num']=4
parameters['methods'] = ["dwi", "dce"]

parameters['namelist'] = ['train','val','test']

#----
# Single model specific parameters
#--


parameters["dwi_model_parameters"] = {


    # --- base model info
    'input_size':128,
 
    
    # ---- model structure ----
    "channels": (32, 64, 128),
    "proj_dim": 64,
    "use_se": False,
    # ---- model features
    "enable_modality_attention": True,
    "backbone_str": 'resnet50',
    
    'grad_clip':5.0,
    # ---- label smoothing ----
    "label_smoothing_enabled": True,
    "label_smoothing_alpha": 0.1,

    # ---- mimic loss ----
    "mimic_enabled": True,
    "lambda_mimic": 1.0,

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

    # --- optimizer parameters
    'optimizer_parameters' : {
      'name':"adamW", #or adam
      "lr":0.001,
      "betas":(0.9, 0.999),
      "eps":1e-08,
      "weight_decay":0.0001,
      "amsgrad":False
    },

    'mask_parameters' : {
      "mask": True,   #mask enabled for custom models
      "lambda_mask": 1.0,
      "mask_loss_type": "dice",
      "mask_target_size": (32, 32),
      "mask_fusion_attention": True
    }


}
#just the same for now
parameters["dce_model_parameters"] =parameters["dwi_model_parameters"] 




#----
# Fusion model parameters
#---


parameters["fusion_model_parameters"] = {
    
    # ---- model structure ----
    "channels": 128,
    "proj_dim": 64,
    "use_se": False,
    'fusion_channels':128,
    'token_pool': (4,4),
    "use_cross_attention":True,
    "use_mask_attention":True,
    "mha_heads":4,
    'grad_clip':5.0,
    # ---- label smoothing ----
    "label_smoothing_enabled": True,
    "label_smoothing_alpha": 0.1,
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

    # --- optimizer parameters
    'optimizer_parameters' : {
      'name':"adamW", #or adam
      "lr":0.001,
      "betas":(0.9, 0.999),
      "eps":1e-08,
      "weight_decay":0.0001,
      "amsgrad":False
    },
    # --- mask parameters
    'mask_parameters' : {
      "mask": True,   #mask enabled for custom models
      "lambda_mask": 1.0,
      "mask_loss_type": "dice",
      "mask_target_size": (32, 32),
      "mask_fusion_attention": True
    },
    # --- do finetune?
    'finetune' : True
}

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

