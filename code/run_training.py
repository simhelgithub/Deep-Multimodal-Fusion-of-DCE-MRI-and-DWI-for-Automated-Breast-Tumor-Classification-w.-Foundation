import os
import torch
import copy
import torch.nn as nn  
from torch.utils.data import dataloader
import torch.optim as optim 
from loss import *
from selector_helpers import *
from train import *
from train_fusion import *
from debug_suite import *

import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Run single model, save the result
def run_single_model(fold, parameters, device, local_model, dataloaders_dict, method, train_labels, name, version, skip_testing = False):

    # ----
    # Get classifcation loss method
    # ----

    classification_loss_method = get_classification_loss(parameters, train_labels, method, device)



    # ---
    # Lightning model setup
    # ---
    local_model = local_model.to(device)
      
    # recorder callback
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    paths = prepare_output_paths(method, fold,parameters)
  
    logger = TensorBoardLogger(
        save_dir=paths["logs"], 
        name=name,
        version=version
    )


    # ---- EARLY STOPPING ----
    early_params = parameters['early_stopping_parameters']
    early_stop_cb = EarlyStopping(
        monitor=early_params['metric'],
        mode=early_params['mode'],
        patience=early_params['patience'] ,
        min_delta=early_params['min_delta'],
        verbose=True
    )

    # ---- GET OPTIMIZER + SCHEDULER FROM YOUR FACTORY ----
    factory = LightningOptimizerFactory(
        model=local_model,
        parameters=parameters,
        model_type=method
    )
    optimizer_fn = factory.optimizer_fn
    scheduler_fn = factory.scheduler_fn

    # ---- BUILD LIGHTNING MODEL ----
    lightning_model = LightningSingleModel(
        model=local_model,
        method=method,
        criterion_clf=classification_loss_method,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,   
        parameters_dict=parameters,
        paths = paths
    )

    #--- run debug
    if parameters['debug_training']:
    
      #Copy model for debug       
      model_cpu = copy.deepcopy(lightning_model)
      model_copy = model_cpu.to(device)

      # Delete the temporary model so CUDA memory is freed

      run_debug_suite_single(model_copy, method, parameters, device)
      del model_copy
      torch.cuda.empty_cache()

    # --- compile model
    if parameters['compile']:
       lightning_model = torch.compile(lightning_model, backend='inductor')
    # ---- CHECKPOINT ----
    checkpoint_cb = ModelCheckpoint(
        dirpath=paths["checkpoints"],
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best",
    )
 
    # ---- TRAINER ----
    trainer = pl.Trainer(
        callbacks=[
            checkpoint_cb,
            lr_monitor,
            early_stop_cb
        ],
        logger=logger,
        max_epochs=parameters["num_epochs"],
        min_epochs=parameters["min_epochs"],
        precision=parameters['precision']
        )

    trainer.fit(
        lightning_model,
        dataloaders_dict["train"],
        dataloaders_dict["val"]
    )
    #results
    
    best_model = LightningSingleModel.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        model=local_model,
        method=method,
        criterion_clf=classification_loss_method,
        optimizer_fn=optimizer_fn,
        parameters_dict=parameters,
        paths = paths,
    )    

    # ----
    # Test the model
    # ----
    skip_testing
    best_model.eval()
    best_model.to(device)
    
    #skip_testing only used for debug
    if skip_testing:
      test_results = None
    else:
      test_results = trainer.test(model=best_model, dataloaders=dataloaders_dict["test"])
      # Save the metrics
      save_metrics(test_results, paths["metrics_json"])
      
    return {
    "best_checkpoint": checkpoint_cb.best_model_path,
    "trained_model": best_model.cpu(),    # CPU to reduce GPU memory
    "train_metrics": trainer.callback_metrics,
    "test_metrics": test_results,   # from trainer.test()
  }
 
def run_fusion_model(dwi_model, dce_model, fusion_model, dataloaders_dict, parameters, device, fold, train_labels, name="fusion", version=None):
    method = "fusion"
    # ---- Losses ----
    classification_loss_method = get_classification_loss(parameters, train_labels, method, device)
    reconstruction_loss_method = get_recon_loss(parameters, method)

    # ---- Paths ----
    paths = prepare_output_paths(method, fold, parameters)

    # ---- Logger & Callbacks ----
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(save_dir=paths["logs"], name=name, version=version)

    early_params = parameters['early_stopping_parameters']
    early_stop_cb = EarlyStopping(
        monitor=early_params['metric'],
        mode=early_params['mode'],
        patience=early_params['patience'],
        min_delta=early_params['min_delta'],
        verbose=True
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=paths["checkpoints"],
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best",
    )

    # ---- Optimizer & Scheduler ----
    # Use unified factory for fusion
    fusion_opt_factory = LightningFusionOptimizerFactory(
        dwi_model= dwi_model,
        dce_model= dce_model,
        fusion_model= fusion_model,
        parameters=parameters
    )
    optimizer_fn = fusion_opt_factory.optimizer_fn
    scheduler_fn = fusion_opt_factory.scheduler_fn

    # ---- Build Lightning Model ----
    lightning_model = LightningFusionModel(
        dwi_model=dwi_model,
        dce_model=dce_model,
        fusion_model=fusion_model,
        parameters_dict=parameters,
        criterion_clf=classification_loss_method,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        paths=paths
    )

    # ---- Optional debug run ----
    if parameters['debug_training']:
        model_cpu = copy.deepcopy(lightning_model).to(device)
        run_debug_suite_fusion(model_cpu, method, parameters, device)
        del model_cpu
        torch.cuda.empty_cache()

    # ---- Compile ----
    if parameters.get('compile', False):
        lightning_model = torch.compile(lightning_model, backend='inductor')

    # ---- Trainer ----
    trainer = pl.Trainer(
        callbacks=[checkpoint_cb, lr_monitor, early_stop_cb],
        logger=logger,
        max_epochs=parameters["num_epochs"],
        min_epochs=parameters["min_epochs"],
        precision=parameters['precision'],
    )

    # ---- Fit ----
    trainer.fit(
        lightning_model,
        dataloaders_dict["train"],
        dataloaders_dict["val"]
    )

    # ---- Load best checkpoint ----
    best_model = LightningFusionModel.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        dwi_model=dwi_model,
        dce_model=dce_model,
        fusion_model=fusion_model,
        parameters_dict=parameters,
        criterion_clf=classification_loss_method,
        optimizer_fn=optimizer_fn,
        paths=paths
    )

    best_model.eval()
    best_model.to(device)

    # ---- Test ----
    test_results = trainer.test(model=best_model, dataloaders=dataloaders_dict["test"])

    save_metrics(test_results, paths["metrics_json"])

    # ---- Save model state dict ----
    model_dict_path = parameters.get('model_dict_path', f"{paths['root']}/fusion_model_dict.pth")
    if os.path.exists(model_dict_path):
        model_dict = torch.load(model_dict_path)
    else:
        model_dict = {}

    model_dict[f'fusion_{fold}'] = fusion_model.state_dict()
    model_dict[f'dwi_{fold}'] = dwi_model.state_dict()
    model_dict[f'dce_{fold}'] = dce_model.state_dict()
    torch.save(model_dict, model_dict_path)

    return {
        "best_checkpoint": checkpoint_cb.best_model_path,
        "trained_model": best_model.cpu(),  # CPU to reduce GPU memory
        "train_metrics": trainer.callback_metrics,
        "test_metrics": test_results,
    }

#---
#save helpers
#---
#todo make more flexible?
def prepare_output_paths(method, fold, parameters, base_dir="results"):
    """Create and return the folder structure for saving results."""

    root = os.path.join(base_dir, method, f"fold_{fold}")

    # Build paths
    paths = {
        "root": root,
        "checkpoints": os.path.join(root, "checkpoints"),
        "logs": os.path.join(root, parameters["save_dir"]),
        "metrics_json": os.path.join(root, "metrics.json"),
        "model_state": os.path.join(root, "model_state_dict.pth"),
    }

    # Create all directory paths
    dirs_to_create = [
        paths["root"],
        paths["checkpoints"],
        paths["logs"],
    ]

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    return paths

def save_metrics(metrics_dict, path):
    """Save metrics to JSON."""
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=4)