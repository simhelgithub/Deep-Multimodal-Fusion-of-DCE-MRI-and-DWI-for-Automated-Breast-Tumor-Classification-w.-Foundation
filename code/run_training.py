import os
import torch
import torch.nn as nn  
from torch.utils.data import dataloader
import torch.optim as optim 
from loss import *
from selector_helpers import *
from model_test import *
from train import *
from train_fusion import *

import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

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
    optimizer_fn = LightningOptimizerFactory(local_model, parameters, method).get_optimizer()
    # ----
    # Train the model
    # ----

    local_model = local_model.to(device)
      
    # recorder callback


    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    paths = prepare_output_paths(method, fold)
    logger = TensorBoardLogger(
        save_dir=paths["logs"], 
        name="",
        version='experiment' #temp disable 
    )

    lightning_model = LightningSingleModel(
        model=local_model,
        method=method,
        criterion_clf=classification_loss_method,
        criterion_recon=reconstruction_loss_method,
        optimizer_fn= optimizer_fn,
        device = device,
        parameters_dict=parameters,
        dataloaders= dataloaders_dict
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=paths["checkpoints"],
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best",
        accelerator="gpu"
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_cb],
        logger=logger,
        max_epochs=parameters["num_epochs"],
        precision="16-mixed",     # AMP
    )

    trainer.fit(lightning_model,
                dataloaders_dict["train"],
                dataloaders_dict["val"])

    #results
    best_model = LightningSingleModel.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        model=local_model,
        method=method,
        criterion_clf=classification_loss_method,
        criterion_recon=reconstruction_loss_method,
        optimizer_fn=optimizer_fn,
        device=device,
        parameters_dict=parameters,
        dataloaders=dataloaders_dict,
    )    


    # ----
    # Test the model
    # ----
    best_model.eval()
    best_model.to(device)
    test_results = trainer.test(model=best_model, dataloaders=dataloaders_dict["test"])
    # Save the metrics
    save_metrics(test_results, paths["metrics_json"])


    return {
    "model_name": key,
    "best_checkpoint": checkpoint_cb.best_model_path,
    "trained_model": best_model.cpu(),    # CPU to reduce GPU memory
    "train_metrics": trainer.callback_metrics,
    "test_metrics": test_results,   # from trainer.test()
  }
 


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
          finetune = True,
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
    model_dict[f'final_fusion_{fold}'] = dwi_model.state_dict()
    model_dict[f'final_dce_{fold}'] = dce_model.state_dict()
    model_dict[f'final_fusion_{fold}'] = fusion_model.state_dict()

    torch.save(model_dict, model_dict_path)

    #unsure if used
    return dwi_model, dce_model, fusion_model



#---
#save helpers
#---
def prepare_output_paths(method, fold, base_dir="results"):
    """Create and return the folder structure for saving results."""
    root = os.path.join(base_dir, method, f"fold_{fold}")
    paths = {
        "root": root,
        "checkpoints": os.path.join(root, "checkpoints"),
        "logs": os.path.join(root, "logs"),
        "metrics_json": os.path.join(root, "metrics.json"),
        "model_state": os.path.join(root, "model_state_dict.pth"),
    }

    for p in paths.values():
        if os.path.splitext(p)[1] == "":
            os.makedirs(p, exist_ok=True)

    return paths


def save_metrics(metrics_dict, path):
    """Save metrics to JSON."""
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=4)