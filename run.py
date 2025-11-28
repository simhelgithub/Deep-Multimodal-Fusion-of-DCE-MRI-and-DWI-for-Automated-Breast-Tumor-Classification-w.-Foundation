# ------------------------------
#  Run all script
# ------------------------------
import torch
import torch.nn as nn

from parameters_generate import *
from prepare_single_model import *
from run_training import *
from prepare_fusion_model import *

from helper import *
from foundation_model import *
from loss import *


#torch defaults
torch.set_default_dtype(torch.float32)


# key=method+str(fold)  ( +data)

# ------------------------------
# PATHS
# ------------------------------
base_path = r'/content/drive/My Drive/master/DWI_DCE_CDFR-DNN_-main/archive/'
parameter_path =  "parameters/parameters.pth"
parameters = torch.load(base_path + parameter_path)

segnum = parameters['segnum']
class_num = parameters['class_num']
methods =parameters['methods']

num_epochs =parameters['num_epochs']

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Running on CPU SLOW WARNING !!")


# -----
# Playground
# -----






# ------------------------------
# TRAINING LOOP
# ------------------------------

for fold in range(segnum):
  dwi_key = None 
  dce_key = None 
  dwi_model = None
  dce_model = None
  dwi_backbone = None
  dce_backbone = None
  train_labels = None

  for method in methods:
      # ============================================================
      # 1. Prepare DWI + DCE Models (Single-Modality)
      # ============================================================
      print(f"\n==== Preparing for {method.upper()} for fold {fold+1}/{segnum} ====\n")
      local_model, dataloaders_dict, key, train_labels, backbone = prepare_single_custom_model(method, fold, parameters, device)

      # ============================================================
      # 2. Train DWI + DCE Models (Single-Modality)
      # ============================================================

      print(f"\n==== Training {method.upper()} model for fold {fold+1}/{segnum} ====\n")
      local_model,train_acc_history,train_loss_history,val_acc_history,val_loss_history = run_single_model(fold, parameters, device, local_model, dataloaders_dict, key, method, train_labels)

      #store fold models for fusion model
      if method == 'dwi':
        dwi_model = local_model
        dwi_key = key
        dwi_backbone = backbone
      elif method == 'dce':
        dce_model = local_model
        dce_key = key
        dce_backbone = backbone
      print(f"\n==== Finished training {method.upper()} model for fold {fold+1}/{segnum} ====\n")

  # ============================================================
  # 3. Prepare Fusion Model
  # ============================================================  
  print(f"\n==== Preparing FUSION model for fold {fold+1}/{segnum} ====\n")

  dataloaders_dict,dwi_model, dce_model, fusion_model = prepare_fusion_model(dwi_key, dce_key, dce_backbone, dwi_backbone, fold, parameters, device)


  # ============================================================
  # 4. Run Fusion Model
  # ============================================================
  print(f"\n==== Training FUSION model for fold {fold+1}/{segnum} ====\n")
  run_fusion_model(dwi_model, dce_model, fusion_model, dataloaders_dict, parameters, device, fold, train_labels)
  print(f"\n==== Finished training FUSION model for fold {fold+1}/{segnum} ====\n")
