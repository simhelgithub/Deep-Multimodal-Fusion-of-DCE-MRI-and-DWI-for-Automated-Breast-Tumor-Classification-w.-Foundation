
try:
    import pytorch_lightning
    print("pytorch_lightning is already installed.")
except ImportError:
    print("pytorch_lightning not found. Installing...")
    !pip install pytorch_lightning --quiet
    print("pytorch_lightning installed.")

try:
    import tensorboard
    print("tensorboard is already installed.")
except ImportError:
    print("tensorboard not found. Installing...")
    !pip install tensorboard --quiet
    print("tensorboard installed.")


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



# ------------------------------
# PATHS
# ------------------------------
base_path = r'/content/drive/My Drive/master/DWI_DCE_CDFR-DNN_-main/archive/'
parameter_path =  "parameters/parameters.pth"
parameters = torch.load(base_path + parameter_path)

segnum = parameters['segnum']
class_num = parameters['class_num']
methods =parameters['methods']


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Running on CPU SLOW WARNING !!")


# Detect GPU
name = torch.cuda.get_device_properties(0).name.lower()
print(f"[INFO] GPU detected: {name}")
# ---------------------------------------------
# Modern TF32 and AMP setup (NO legacy API USED)
# ---------------------------------------------
if "l4" in name:
    print("[INFO] NVIDIA L4 detected → Using TF32 + BF16 AMP")
    torch.set_float32_matmul_precision("high")    # old style
    #torch.backends.cuda.matmul.fp32_precision = "tf32" #crashes?
    #torch.backends.cudnn.conv.fp32_precision = "tf32" #crashes?
    parameters['precision'] = "bf16-mixed"       # L4 prefers BF16

elif "a100" in name:
    print("[INFO] NVIDIA A100 detected → Using TF32 + BF16 AMP")
    torch.set_float32_matmul_precision("high")   # old style
    #torch.backends.cuda.matmul.fp32_precision = "tf32" #crashes?
    #torch.backends.cudnn.conv.fp32_precision = "tf32" #crashes?
    parameters['precision'] = "bf16-mixed"       # A100 native BF16

else:
    print("[INFO] Unknown GPU → Using FP16 AMP")
    torch.set_float32_matmul_precision("medium")
    parameters['precision'] = "16-mixed"
# ---------------------------------------------
# Enable Flash / Memory Efficient Attention
# ---------------------------------------------
#torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_mem_efficient_sdp(True)
#torch.backends.cuda.enable_math_sdp(True)


# -----
# Playground
# -----
#dwi_name = ''
#dce_name = ''
#fusion_name = ''
segnum = 1
version = "base_highlr_lowdownsample"
#parameters['dwi_model_parameters']['use_backbone'] = False

parameters['dwi_model_parameters']['use_hybrid_transformer'] = False

#parameters['num_epochs'] = 2
#run quick
parameters['compile'] = True
#parameters['unfreeze_timer'] = 40
#parameters['foundation_model_unfreeze_timer'] = 1

parameters['debug_training'] = False


skip_testing = False #WARNING
#parameters['fusion_model_parameters']['recon_enabled'] = False  #currently breaks grad norm
#parameters['fusion_model_parameters']['mimic_enabled'] = False  #currently breaks grad norm

methods= ["dce", "dwi"] #change order

#parameters['min_epochs'] = 20 # is calculated automiatically if not set

load_pretrained = False
ver_nr=141

# ------------------------------
# TRAINING LOOP
# ------------------------------

for fold in range(segnum):
#for fold in range(3,3):

  train_labels = None
  dwi_results = None
  dce_results = None
  fusion_results = None #todo remove?

  for method in methods:
  #if False:
      name = f"{version}_{method}_{fold}"
      # ============================================================
      # 1. Prepare DWI + DCE Models (Single-Modality)
      # ============================================================
      print(f"\n= Preparing for {method.upper()} for fold {fold+1}/{segnum} =\n")
      local_model, dataloaders_dict, key, train_labels, backbone = prepare_single_model_dynamic(method, fold, parameters, device,load_pretrained=load_pretrained, ver_nr=ver_nr )
      # ============================================================
      # 2. Train DWI + DCE Models (Single-Modality)
      # ============================================================


      if load_pretrained:
        #store fold models for fusion model
        if method == 'dwi':
          d= {}
          d['trained_model'] = local_model
          dwi_results = d
        elif method == 'dce':
          d= {}
          d['trained_model'] = local_model
          dce_results = d

      else:
        print(f"\n==== Training {method.upper()} model for fold {fold+1}/{segnum} ====\n")
        results = run_single_model(fold, parameters, device, local_model, dataloaders_dict, method, train_labels, name = name, version = version)

        #store fold models for fusion model
        if method == 'dwi':
          dwi_results = results
        elif method == 'dce':
          dce_results = results

        print(f"\n==== Finished training {method.upper()} model for fold {fold+1}/{segnum} ====\n")
  '''
  # ============================================================
  # 3. Prepare Fusion Model
  # ============================================================
  print(f"\n= Preparing FUSION model for fold {fold+1}/{segnum} =\n")

  dataloaders_dict,dwi_model, dce_model, fusion_model = prepare_fusion_model(dwi_results, dce_results, fold, parameters, device)


  # ============================================================
  # 4. Run Fusion Model
  # ============================================================
  print(f"\n==== Training FUSION model for fold {fold+1}/{segnum} ====\n")
  name = f"fusion_{method}_{fold}"
  run_fusion_model(dwi_model, dce_model, fusion_model, dataloaders_dict, parameters, device, fold, train_labels, name = name, version = version)
  print(f"\n==== Finished training FUSION model for fold {fold+1}/{segnum} ====\n")
  '''
  #cleanup memory
  del dataloaders_dict
  del dwi_results
  del dce_results
  torch.cuda.empty_cache()