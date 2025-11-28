
import torch
import torch.nn as nn  
from torch.utils.data import dataloader
import torch.optim as optim 
from loss import *
from selector_helpers import *
from model_test import *
from train import *
from train_fusion import *

# Run single model, save the result
def run_single_model(fold, parameters, device, local_model, dataloaders_dict, key, method, train_labels):

    # ----
    # Get classifcation loss method
    # ----
    classification_loss_method = get_classification_loss(parameters, train_labels, method, device)
    # ----
    # Selectors for reconstruction loss
    # ----
    reconstruction_loss_method = get_recon_loss(parameters, method)

    # ----
    # Select optimizer
    # ----
    optimizer =  get_optimizer(local_model, parameters, method)

    # ----
    # Train the model
    # ----

    local_model = local_model.to(device)
    
   

    local_model,train_acc_history,train_loss_history,val_acc_history,val_loss_history= train_model(local_model, 
                                                                                            dataloaders_dict,
                                                                                            method,
                                                                                            classification_loss_method,
                                                                                            reconstruction_loss_method,
                                                                                            optimizer,
                                                                                            device, 
                                                                                            parameters)


    
    # ----
    # Test the model
    # ----
    single_model_test(local_model, dataloaders_dict,device, parameters)
  
    # ----
    # Store the model
    # ----
    model_dict_path = parameters['model_dict_path']

    model_dict=torch.load(model_dict_path)
    model_dict[key]=local_model.state_dict()
    torch.save(model_dict,model_dict_path)

    return  local_model,train_acc_history,train_loss_history,val_acc_history,val_loss_history




def run_fusion_model(dwi_model, dce_model, fusion_model, dataloaders, parameters, device, fold, train_labels, method = "fusion"):

    # Get classifcation loss method
    # ----
    classification_loss_method = get_classification_loss(parameters, train_labels, method, device)

    # ---- 

    # Selectors for reconstruction loss
    # ----
    reconstruction_loss_method = get_recon_loss(parameters, method)

    # ----
    # Select optimizer
    # ----
    optimizer =  get_optimizer(fusion_model, parameters, method)


    # ---
    # train phase
    # ---
    
    dwi_model, dce_model, fusion_model, history = train_fusion_model(
        dwi_model,
        dce_model,
        fusion_model,
        dataloaders,
        optimizer,
        classification_loss_method,
        device,
        parameters,
        finetune = False
    )
    # -------------
    # Save updated weights
    # -------------
    model_dict_path = parameters['model_dict_path']

    model_dict=torch.load(model_dict_path)
    model_dict[f'fusion_{fold}']=fusion_model.state_dict()
    torch.save(model_dict,model_dict_path)

    # -- 
    # finetune, if applicable
    # --


    if parameters["fusion_model_parameters"]["finetune"]: 
      dwi_model, dce_model, fusion_model, history = train_fusion_model(
          dwi_model,
          dce_model,
          fusion_model,
          dataloaders,
          optimizer,
          classification_loss_method,
          device,
          parameters,
          finetune = True
        )

    # --------------
    # Testing
    # ------
    fusion_model_test(
        dwi_model=dwi_model,
        dce_model=dce_model,
        fusion_model=fusion_model,
        dataloaders=dataloaders,
        device=device,
        mask_fusion=False  # we never have masks for testing
    )
    # -------------
    # Save final weights
    # ------------
    model_dict['final_dwi'] = dwi_model.state_dict()
    model_dict['final_dce'] = dce_model.state_dict()
    model_dict['final_fusion'] = fusion_model.state_dict()

    torch.save(model_dict, model_dict_path)

    #unsure if used
    return dwi_model, dce_model, fusion_model

