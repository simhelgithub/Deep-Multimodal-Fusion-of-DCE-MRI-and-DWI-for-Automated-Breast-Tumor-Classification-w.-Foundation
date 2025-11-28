import time
import copy
import gc
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler


from train import *
from loss import *
from selector_helpers import * 


def print_mem(tag: str = ""):
    if torch.cuda.is_available():
        print(f"{tag} GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB "
              f"reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def train_fusion_model(
    dwi_model,
    dce_model,
    fusion_model,
    dataloaders,
    optimizer,
    criterion_clf,
    device,
    parameters,
    finetune = False,
    method = "fusion"):


    # ----
    # basic config
    # ---
    model_params = parameters["fusion_model_parameters"]
    grad_clip = model_params["grad_clip"]
    if finetune:
      num_epochs = parameters["finetune_num_epochs"]
    else: 
      num_epochs = parameters["num_epochs"]



    # reconstruction
    recon_enabled = model_params["recon_enabled"]
    lambda_recon = model_params["lambda_recon"]

    

    # mask parameters (same structure as train_model)
    mask_params = model_params["mask_parameters"]
    mask_enabled = mask_params["mask"]
    lambda_mask = mask_params["lambda_mask"]
 

    # label smoothing
    label_smoothing_enabled = model_params["label_smoothing_enabled"]
    label_smoother = None
    if label_smoothing_enabled:
        alpha = model_params["label_smoothing_alpha"]
        class_num = parameters["class_num"]
        label_smoother = LabelSmoothing(class_num, alpha)

    # mask criterion 
    mask_criterion = mask_criterion_selector(parameters, method)
    if mask_criterion is None:
      print(mask_enabled, lambda_mask)


    #debug
    debug_training = parameters["debug_training"]
    ENABLE_MASK_VIZ = debug_training
    show_attention = debug_training
    debug_first = debug_training

    VIZ_FREQUENCY = parameters.get("viz_frequency", 10)

    # ----------
    # Prepare AMP 
    # -------
    use_amp = parameters.get("use_amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

  
    dwi_model.to(device)
    dce_model.to(device)
    fusion_model.to(device)

    # ----
    # Best model bookkeeping 
    # ------
    since = time.time()

    best_weights = {
        'dwi': copy.deepcopy(dwi_model.state_dict()),
        'dce': copy.deepcopy(dce_model.state_dict()),
        'fusion': copy.deepcopy(fusion_model.state_dict())
    }
    best_val = -float("inf")



    # histories
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }    
    # move models to device
    dwi_model.to(device)
    dce_model.to(device)
    fusion_model.to(device)


    # ----------------
    # training loop
    # ---------------
    for epoch in range(num_epochs):
      print(f"Epoch {epoch+1}/{num_epochs}")
      epoch_start = time.time()
      for phase in ["train", "val"]:
          is_train = phase == "train"
          if is_train:
              dwi_model.train()
              dce_model.train()
              fusion_model.train()
          else:
              dwi_model.eval()
              dce_model.eval()
              fusion_model.eval()

          running_loss = 0.0
          running_corrects = 0
          running_mask_dice = 0.0
          mask_count = 0
          outputs_dict = {}
          total_loss = 0
          dataset_len = len(dataloaders[phase].dataset) if hasattr(dataloaders[phase], 'dataset') else 0

          for batch in dataloaders[phase]:
              # batch can be (dwi, dce, masks, labels) or (dwi, dce, labels)
              if len(batch) == 4:
                  dwi_inputs, dce_inputs, masks_batch, labels = batch
              elif len(batch) == 3:
                  dwi_inputs, dce_inputs, labels = batch
                  masks_batch = None
              else:
                  raise ValueError("Dataloader must yield (dwi, dce, masks?, labels)")

              
              dwi_inputs = dwi_inputs.float().to(device, non_blocking=True)
              dce_inputs = dce_inputs.float().to(device, non_blocking=True)
              labels = labels.long().to(device, non_blocking=True)
              if masks_batch is not None:
                  masks_batch = masks_batch.float().to(device, non_blocking=True)

              optimizer.zero_grad(set_to_none=True)

              with torch.set_grad_enabled(is_train):
                  if use_amp:
                      autocast_ctx = autocast()
                  else:
                      class _noop:
                          def __enter__(self): return None
                          def __exit__(self, exc_type, exc, tb): return False
                      autocast_ctx = _noop()

                  with autocast_ctx:
                      # encoder forwards
                      _, dwi_aux, dwi_mask_pred = dwi_model(dwi_inputs)
                      _, dce_aux, dce_mask_pred = dce_model(dce_inputs)
 

                      # fusion forward  
                      logits, fused_mask_logits, aux = fusion_model(
                          dwi_aux.get("raw_feats", None),
                          dce_aux.get("raw_feats", None),
                          dwi_mask_pred,
                          dce_mask_pred
                      )

                      
                      # build unified outputs dict 
                  
                      outputs_dict = {
                          'logits': logits,
                          'fusion_mask': fused_mask_logits,
                          'recon': {
                              'dwi': dwi_aux.get('recon_feats', None),
                              'dce': dce_aux.get('recon_feats', None),
                              'fusion': aux.get('recon_fused', None) if aux is not None else None
                          },
                          'proj_pairs': {
                              'dwi': dwi_aux.get('proj_pairs', None),
                              'dce': dce_aux.get('proj_pairs', None),
                              'fusion': aux.get('proj_fused', None) if aux is not None else None
                          },
                          'modality_attn': aux.get('gating_weights', None) if aux is not None else None,
                          'aux': aux
                      }
                      

                      # classification loss (label smoothing if provided)
                      if label_smoother is not None and is_train:
                          smoothed = label_smoother(logits, labels)
                          cls_loss = criterion_clf(logits, smoothed) 
                      else:
                          cls_loss = criterion_clf(logits, labels)
                      total_loss = cls_loss

                      mask_loss_val = 0.0
                      if masks_batch is not None:
                          mask_loss_val = (
                              safe_mask_loss(dwi_mask_pred, masks_batch, mask_criterion) +
                              safe_mask_loss(dce_mask_pred, masks_batch, mask_criterion) +
                              safe_mask_loss(outputs_dict['fusion_mask'], masks_batch, mask_criterion), 
                          )
                        
                      # reconstruction loss (optional, only if finetuning encoder recon)
                      recon_loss_val = 0.0
                      if finetune:
                          if outputs_dict['recon']['dwi'] is not None:
                              recon_loss_val += compute_recon_list_loss(outputs_dict['recon']['dwi'], dwi_inputs)
                          if outputs_dict['recon']['dce'] is not None:
                              recon_loss_val += compute_recon_list_loss(outputs_dict['recon']['dce'], dce_inputs)


                      total_loss = cls_loss + lambda_recon * recon_loss_val 

              # backward + step
              if phase == 'train':
                  if use_amp:
                      scaler.scale(total_loss).backward()
                      if grad_clip is not None:
                          scaler.unscale_(optimizer)
                          torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), grad_clip)
                      scaler.step(optimizer)  
                      scaler.update()
                  else:
                      total_loss.backward()
                      if grad_clip is not None:
                          torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), grad_clip)
                      optimizer.step()
          
              # metrics (use outputs_dict fields)
              with torch.no_grad():
                  preds = torch.argmax(outputs_dict['logits'], dim=1)
                  running_corrects += torch.sum(preds == labels).item()

                  running_loss += total_loss.item() * labels.size(0)

              # safe deletions to free memory
              for varname in ['dwi_mask_pred','dce_mask_pred','dwi_aux','dce_aux','outputs_dict','logits','fused_mask_logits','aux','total_loss']:
                  if varname in locals():
                      try:
                          del locals()[varname]
                      except Exception:
                          pass
                          
              gc.collect()
              torch.cuda.empty_cache()

          # epoch metrics
          epoch_loss = running_loss / (dataset_len if dataset_len > 0 else 1)
          epoch_acc = running_corrects / (dataset_len if dataset_len > 0 else 1)
          epoch_mask_dice = running_mask_dice / mask_count if mask_count > 0 else 0.0

          print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f} mask_dice: {epoch_mask_dice:.4f}")

          if phase == 'train':
              history['train_loss'].append(epoch_loss); history['train_acc'].append(epoch_acc)
          else:
              history['val_loss'].append(epoch_loss); history['val_acc'].append(epoch_acc)
              if epoch_acc > best_val:
                  best_val = epoch_acc
                  best_weights['dwi'] = {k: v.detach().cpu().clone() for k, v in dwi_model.state_dict().items()}
                  best_weights['dce'] = {k: v.detach().cpu().clone() for k, v in dce_model.state_dict().items()}
                  best_weights['fusion'] = {k: v.detach().cpu().clone() for k, v in fusion_model.state_dict().items()}
          

      epoch_time = time.time() - epoch_start
      print(f"Epoch {epoch+1} completed in {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
      print_mem("after epoch")

    total_time = time.time() - since
    print(f"Training complete in {total_time//60:.0f}m {total_time%60:.0f}s")

    # load best weights
    dwi_model.load_state_dict({k: v.to(device) for k, v in best_weights['dwi'].items()})
    dce_model.load_state_dict({k: v.to(device) for k, v in best_weights['dce'].items()})
    fusion_model.load_state_dict({k: v.to(device) for k, v in best_weights['fusion'].items()})

    return dwi_model, dce_model, fusion_model, history 

# helper that pairs recon list and target scales
def compute_recon_list_loss(recon_list, input_img):
  loss = 0.0
  # targets: full, half, quarter,... depending on recon_list length
  scales = [1.0/(2**i) for i in range(0, len(recon_list))]  # e.g., [1, 1/2]
  for r, scale in zip(recon_list, scales):
      if r is None:
          continue
      target = input_img
      if scale != 1.0:
          target = F.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
      # ensure size match
      if target.shape[-2:] != r.shape[-2:]:
          target = F.interpolate(target, size=r.shape[-2:], mode='bilinear', align_corners=False)
      loss = loss + F.smooth_l1_loss(r, target)
  return loss

def safe_mask_loss(pred_logits, gt_mask, mask_criterion):
    if pred_logits.shape[-2:] != gt_mask.shape[-2:]:
        print("mask resized warning safe_mask_loss, loss.py",pred_logits.shape[-2:], gt_mask.shape[-2:]) 
        gt_resized = F.interpolate(gt_mask, size=pred_logits.shape[-2:], mode='nearest')
    else:
        gt_resized = gt_mask
    return mask_criterion(pred_logits, gt_resized)