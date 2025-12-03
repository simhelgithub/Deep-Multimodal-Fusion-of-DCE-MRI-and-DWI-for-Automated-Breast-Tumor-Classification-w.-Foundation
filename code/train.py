import time
import gc
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import MeanMetric
import pytorch_msssim

from loss import *
from selector_helpers import *
from torchmetrics.classification import MulticlassAUROC
VIZ_FREQUENCY = 10


class LightningSingleModel(pl.LightningModule):   
    def __init__(
        self,
        model,
        method,
        criterion_clf,
        #criterion_recon,
        optimizer_fn,
        scheduler_fn=None,      
        device=None,
        parameters_dict=None,
        dataloaders=None,
        paths = None,
    ):
        super().__init__()
        # store refs
        self.model = model
        self.method = method
        self.criterion_clf = criterion_clf
        #self.criterion_recon = criterion_recon
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.parameters_dict = parameters_dict
        self.dataloaders_ref = dataloaders   # only needed for viz
        self.paths = paths

        # unpack model params
        model_params = parameters_dict[f"{method}_model_parameters"]
        self.model_params = model_params
        self.recon_enabled = model_params["recon_enabled"]
        self.lambda_recon = model_params["lambda_recon"]
        self.mimic_enabled = model_params["mimic_enabled"]
        self.lambda_mimic = model_params["lambda_mimic"]
        self.enable_modality_attention = model_params["enable_modality_attention"]
        self.class_num = parameters_dict["class_num"]

        #regularization features
        self.attn_reg_enabled = model_params.get("attn_reg_enabled", False)
        self.lambda_attn_sparsity = model_params.get("lambda_attn_sparsity", 0.0)
        self.lambda_attn_consistency = model_params.get("lambda_attn_consistency", 0.0)

        self.feat_norm_reg_enabled = model_params.get("feat_norm_reg_enabled", False)
        self.lambda_feat_norm = model_params.get("lambda_feat_norm", 0.0)


        #unfreeze      
        self.unfreeze_timer = int(self.parameters_dict["unfreeze_timer"])
        self.backbone_freeze_on_start = self.parameters_dict['backbone_freeze_on_start'] 
        self.backbone_num_groups = self.parameters_dict['backbone_num_groups']
        self.layers_unfrozen = 0
        

        # mask params
        mask_params = model_params["mask_parameters"]
        self.mask_enabled = mask_params["mask"]
        self.lambda_mask = mask_params["lambda_mask"]
        self.mask_loss_type = mask_params["mask_loss_type"]

        # label smoothing
        label_smoothing_enabled = model_params["label_smoothing_enabled"]
        if label_smoothing_enabled:
            alpha = model_params["label_smoothing_alpha"]
            self.label_smoother = LabelSmoothing(self.class_num, alpha)
        else:
            self.label_smoother = None

        # mask loss
        self.mask_criterion = mask_criterion_selector(parameters_dict, method)

        # debug
        self.debug_training = parameters_dict["debug_training"]
        self.debug_first =parameters_dict["full_debug"]


        
        # book keeping for some non standard metrics
        self.best_val_acc = -1.0 #used to only save on best val acc
        self.latest_val_sample = None

        # used for visualization
        self._prepare_viz_batch(dataloaders)

        #transform helpers for tta
        self.transforms_list = self.transforms_list = [ 
                tta_id,
                tta_flip_lr,
                tta_flip_ud,
                tta_flip_lrud]
        #inverses
        #  [tta_id,inv_tta_id,tta_flip_lr,inv_tta_flip_lr,tta_flip_ud,inv_tta_flip_ud,tta_flip_lrud,inv_tta_flip_lrud]

        # metric objects (track averages across batches/epochs)
        self.train_mask_dice = MeanMetric()
        self.val_mask_dice = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.train_mimic_loss = MeanMetric()
        self.val_mimic_loss = MeanMetric()
        self.train_auc = MulticlassAUROC(num_classes=self.class_num)
        self.val_auc = MulticlassAUROC(num_classes=self.class_num)
        self.test_auc = MulticlassAUROC(num_classes=self.class_num)
        self.val_roc_auc = MulticlassAUROC(num_classes=self.class_num)

        self.opt_factory = LightningOptimizerFactory(
                model=self.model,
                parameters=parameters_dict,
                model_type=method,
            )
    # -------------------------
    # prep viz batch
    # ----------------
    def _prepare_viz_batch(self, dataloaders):
        self.viz_inputs_cpu = None
        self.viz_masks_cpu = None
        self.viz_labels_cpu = None
        self.num_viz_samples = 0

        if self.debug_training and "val" in dataloaders and len(dataloaders["val"]) > 0:
            example = next(iter(dataloaders["val"]))
            if len(example) == 3:
                vi, vm, vl = example
            else:
                vi, vl = example
                vm = None

            if vi is not None:
                N = min(4, vi.size(0))
                self.num_viz_samples = N
                self.viz_inputs_cpu = vi[:N].cpu()
                if vm is not None:
                    self.viz_masks_cpu = vm[:N].float().cpu()
                self.viz_labels_cpu = vl[:N].cpu()
    
    #---
    # get optimizer
    #---


    def configure_optimizers(self):
      optimizer = self.optimizer_fn(self.model.parameters())
      if self.scheduler_fn is None:
          print('train, no set scheduler')
          return optimizer

      sched_ret = self.scheduler_fn(optimizer)

      if isinstance(sched_ret, dict):
          # pack optimizer + scheduler dict in Lightning format
          return {
              "optimizer": optimizer,
              "lr_scheduler": sched_ret,
          }

      # if a scheduler object was returned, wrap it
      if hasattr(sched_ret, "step"):
          return {
              "optimizer": optimizer,
              "lr_scheduler": {
                  "scheduler": sched_ret,
                  "monitor": self.parameters_dict.get("control_metric", None),
                  "interval": "epoch",
              },
              "grad_clip_val": self.parameters_dict[f'{self.method}_model_parameters']['grad_clip'],
              "gradient_clip_algorithm": self.parameters_dict[f'{self.method}_model_parameters']['gradient_clip_algorithm']
          }
      #fallback
      print('no scheduler, returning optimizer in train')
      return optimizer


    # --- 
    # Gradual unfreezing
    # ---
    def on_train_epoch_start(self):

        if self.backbone_freeze_on_start and self.current_epoch <= (self.unfreeze_timer*self.backbone_num_groups+1):         
          if self.current_epoch % self.unfreeze_timer == 0 and self.current_epoch != 0:
            self.opt_factory.gradual_unfreeze(
                epoch=self.current_epoch,
                unfreeze_every_n_epochs=self.unfreeze_timer
            )
            self._sync_optimizer_new_params()
    #reset key metrics on epoch to be safe
    def on_validation_epoch_start(self):
        # reset validation metrics to ensure per-epoch accumulation
        try:
            self.val_auc.reset()
        except Exception:
            pass
        try:
            self.val_roc_auc.reset()
        except Exception:
            pass

    def on_test_epoch_start(self):
        try:
            self.test_auc.reset()
        except Exception:
            pass
    # --------------------------------
    # forward 
    # --------------------------------
    def forward(self, x, masks=None):
        return self.model(x, masks)


    # --------------------------------
    # shared step (train + val)
    # --------------------------------

    def _shared_step(self, batch, batch_idx, phase, return_preds: bool = False):
        is_train = phase == "train"

        # unpack
        if len(batch) == 3:
            inputs, masks, labels = batch
        else:
            inputs, labels = batch
            masks = None

        # aux loss weight scheduling, drops off towards epoch
        if self.parameters_dict.get('use_simple_aux_loss_scheduling', False):
            aux_w = max(0.0, 1 - self.current_epoch / self.parameters_dict["aux_loss_weight_epoch_limit"])
        else:
            aux_w = 1.0

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)

        # ensure label dtype for classification
        labels = labels.long()

        # DEBUG only first few
        if self.current_epoch == 0 and batch_idx < 5 and self.debug_first and is_train:
            print(f"[DEBUG] Input Stats: Min={inputs.min():.4f}, Max={inputs.max():.4f}, Mean={inputs.mean():.4f}, Std={inputs.std():.4f}")
            if masks is not None:
                print(f"[DEBUG] Mask Stats: Min={masks.min():.4f}, Max={masks.max():.4f}, Mean={masks.mean():.4f}")

        # If caller asked for predictions (validation), run forward under no_grad to be explicit/safe
        # forward step
        if return_preds and not is_train:
            with torch.no_grad():
                outputs, aux, mask_output = self(inputs, masks)
        else:
            outputs, aux, mask_output = self(inputs, masks)
        recon_feats = aux.get("recon_feats", []) if aux is not None else []
        proj_pairs = aux.get("proj_pairs", None) if aux is not None else None

        # classification loss (label smoothing only during training)
        if self.label_smoother is not None and is_train:
            smoothed = self.label_smoother(outputs, labels)
            clf_loss = self.criterion_clf(outputs, smoothed)
        else:
            # during validation we compute plain classification loss for logging (no gradient if no_grad)
            clf_loss = self.criterion_clf(outputs, labels)

        batch_loss = clf_loss
        # ---------------------------------------
        # REGULARIZATION BLOCK 
        # -------------------------------------
        
        attn_sparsity_loss = torch.tensor(0.0, device=self.device)
        attn_consistency_loss = torch.tensor(0.0, device=self.device)
        feat_norm_loss = torch.tensor(0.0, device=self.device)

        raw_feats = aux.get("raw_feats", None)

        # ----- ATTENTION SPARSITY -----
        if self.attn_reg_enabled and self.lambda_attn_sparsity > 0:
            attn_map = aux.get("mask_attn_map", None)

            if attn_map is not None:
                # L1 sparsity on attention map
                attn_sparsity_loss = attn_map.abs().mean()
            else:
                attn_sparsity_loss = torch.tensor(0.0, device=self.device)

            batch_loss += self.lambda_attn_sparsity * attn_sparsity_loss

        # ----- ATTENTION CONSISTENCY -----
        if self.attn_reg_enabled and self.lambda_attn_consistency > 0:
            # Use projected features (dimension = proj_dim)
            p1, p1_r, p2, p2_r = aux["proj_pairs"]

            # Upsample p2 to match p1 spatial size
            p2_up = F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)

            # Normalize both
            p1_norm = p1 / (p1.norm(dim=1, keepdim=True) + 1e-6)
            p2_norm = p2_up / (p2_up.norm(dim=1, keepdim=True) + 1e-6)

            # Consistency loss
            attn_consistency_loss = torch.tensor(0.0, device=self.device)

            attn_consistency_loss = F.mse_loss(p1_norm, p2_norm)

            batch_loss += self.lambda_attn_consistency * attn_consistency_loss

        # ----- FEATURE NORM REGULARIZATION -----
        if self.feat_norm_reg_enabled and self.lambda_feat_norm > 0 and raw_feats is not None:
            feat_norm_loss = sum([f.pow(2).mean() for f in raw_feats])
            batch_loss+=self.lambda_feat_norm * feat_norm_loss

        


        # ----------------------
        # mask losses
        # ----------------
        mask_out_resized = None
        if self.mask_enabled and mask_output is not None and masks is not None:
            if mask_output.shape[-2:] != masks.shape[-2:]:
                print('warning, mask rezised')
                mask_out_resized = F.interpolate(mask_output,
                                                 size=masks.shape[-2:],
                                                 mode="bilinear",
                                                 align_corners=False)
            else:
                mask_out_resized = mask_output

            mask_loss = self.mask_criterion(mask_out_resized, masks)
            # Add mask loss to optimization loss *only during training*
            if is_train:
                batch_loss = batch_loss + self.lambda_mask * mask_loss

            # dice metric (update irrespective of phase)
            with torch.no_grad():
                pred_bin = (torch.sigmoid(mask_out_resized) > 0.5).float()
                gt_bin = (masks > 0.5).float()
                inter = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
                union = pred_bin.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3))
                dice = ((2 * inter + 1e-6) / (union + 1e-6))  # per-sample dice

                if is_train:
                    self.train_mask_dice.update(dice.mean())
                else:
                    self.val_mask_dice.update(dice.mean())

        # ----------------
        # recon + mimic
        # -----------------
        recon_loss_val = torch.tensor(0.0, device=self.device)
        mimic_loss_val = torch.tensor(0.0, device=self.device)

        if self.recon_enabled and aux_w > 0.0:

            # ---------
            # Reconstruction
            # ---------
            for idx_r, pred_r in enumerate(recon_feats):
                if pred_r is None:
                    continue
                
                # --- Choose target WITHOUT scaling ---
                if idx_r == 0:
                    target = inputs
                else:
                    # Instead of scale_factor, directly use pred_r size
                    target = F.interpolate(
                        inputs, 
                        size=pred_r.shape[-2:], 
                        mode="bilinear", 
                        align_corners=False
                    )

                # --- Channel match (grayscale case) ---
                if pred_r.size(1) == 1 and target.size(1) > 1:
                    target = target.mean(dim=1, keepdim=True)

                # --- Ensure sizes match (failsafe only) ---
                if pred_r.shape[-2:] != target.shape[-2:]:
                    pred_r = F.interpolate(
                        pred_r,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                recon_loss_val += recon_image_loss(pred_r, target)


            # ---------
            # Mimic Loss
            # ---------
            if self.mimic_enabled and proj_pairs is not None and len(proj_pairs) >= 4:
                p1, p1_r, p2, p2_r = proj_pairs[:4]
                mimic_loss_val = (
                    mimic_feat_loss(p1, p1_r) +
                    mimic_feat_loss(p2, p2_r)
                )


            # ---------
            # Add auxiliary losses (training only)
            # ---------
            if is_train:
                batch_loss += (
                    self.lambda_recon * recon_loss_val * aux_w +
                    self.lambda_mimic * mimic_loss_val * aux_w
                )


        # ---------
        # Metrics Logging
        # ---------
        if is_train:
            self.train_recon_loss.update(recon_loss_val.detach())
            self.train_mimic_loss.update(mimic_loss_val.detach())
        else:
            self.val_recon_loss.update(recon_loss_val.detach())
            self.val_mimic_loss.update(mimic_loss_val.detach())

        # ------------------
        # metrics (accuracy counted without grads)
        # ---------------------
        with torch.no_grad():
            _, preds = outputs.max(dim=1)
            correct = (preds == labels).sum().item()
            batch_size = labels.size(0)

        # log (Lightning)
        self.log(f"{phase}_loss", batch_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_acc", correct / batch_size, prog_bar=True, on_step=False, on_epoch=True)

        # log metric objects (will be aggregated/reset by Lightning)
        if is_train:
            self.log("train_mask_dice", self.train_mask_dice, on_epoch=True, prog_bar=True)
            self.log("train_recon_loss", self.train_recon_loss, on_epoch=True, prog_bar=True)
            self.log("train_mimic_loss", self.train_mimic_loss, on_epoch=True, prog_bar=True)
        else:
            self.log("val_mask_dice", self.val_mask_dice, on_epoch=True, prog_bar=True)
            self.log("val_recon_loss", self.val_recon_loss, on_epoch=True, prog_bar=True)
            self.log("val_mimic_loss", self.val_mimic_loss, on_epoch=True, prog_bar=True)

        if return_preds:
            # detach outputs so caller doesn't accidentally keep computation graph
            return batch_loss.detach(), outputs.detach(), aux, mask_out_resized  
        return batch_loss





    # ----
    # Dropout (only during training)
    # ---
    def enable_dropout(self, model: torch.nn.Module):
      for m in model.modules():
          if isinstance(m, torch.nn.Dropout):
              m.train()
    
    
    # ---
    # MC dropout
    # ---
    #helper for setting batchnorm to eval
    def set_batchnorm_eval(self,model: torch.nn.Module):
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                              torch.nn.SyncBatchNorm)):
                m.eval()
    def _get_module_train_states(self, model: torch.nn.Module):
        # return dict of module -> was_training
        states = {}
        for m in model.modules():
            states[m] = m.training
        return states

    def _restore_module_train_states(self, model: torch.nn.Module, states):
        for m, was_training in states.items():
            try:
                m.train(was_training)
            except Exception:
                pass
    # Enable MC dropout
    def mc_enable(self, model: torch.nn.Module):
        self.enable_dropout(model)
        self.set_batchnorm_eval(model)
    # MC dropout prediction
  
    def predict_mc_dropout(self, model, x, passes=20):
      # save original training/eval states
      orig_states = self._get_module_train_states(model)
      # enable mc
      self.mc_enable(model)
      preds = []
      with torch.no_grad():
          for _ in range(passes):
              logits = model(x)[0]
              probs = torch.softmax(logits, dim=1)
              preds.append(probs)
      preds = torch.stack(preds, dim=0)
      # restore modes
      self._restore_module_train_states(model, orig_states)
      return preds.mean(0), preds.std(0)
      
 

    # ----
    # TTA
    #---- 

    
    #todo could be made to share function with tta_mc
    def predict_tta(self, model, x, masks, transforms=None):
        if transforms is None:
            transforms = self.transforms_list
        preds = []
        with torch.no_grad():
            for t in transforms:
                xt = t(x=x)
                logits = model(xt, masks)[0]
                probs = torch.softmax(logits, dim=1)
                preds.append(probs)

        preds = torch.stack(preds, dim=0)
        return preds.mean(0)

    # ----
    # both tta and mc
    # ---
    def predict_tta_mc(self, model, x, masks,  transforms=None, passes=10):
        # save original training/eval states
        orig_states = self._get_module_train_states(model)

        if transforms is None:
            transforms = self.transforms_list

        preds = []
        self.mc_enable(model)

        with torch.no_grad():
            for t in transforms:
                xt = t(x=x)                      
                for _ in range(passes):
                    logits = model(xt, masks)[0]
                    probs = torch.softmax(logits, dim=1)
                    preds.append(probs)
        # restore modes
        self._restore_module_train_states(model, orig_states)
        preds = torch.stack(preds, dim=0)
        return preds.mean(0), preds.std(0)
      
    #---
    # calls the chosen test mode
    #--  
    def predict_custom(self, batch, mode="normal", mc_passes=10):
      inputs = batch[0]
      labels = batch[-1]
      masks = batch[1] if len(batch) == 3 else None

      if mode == "normal":
          with torch.no_grad():
              outputs, aux, mask_out = self.model(inputs, masks)
          return outputs

      elif mode == "tta":
          return self.predict_tta(self.model, inputs, masks)

      elif mode == "mc":
          return self.predict_mc_dropout(self.model, inputs, passes=mc_passes)

      elif mode == "tta_mc":
          return self.predict_tta_mc(self.model, inputs, masks, passes=mc_passes)

      else:
          raise ValueError(f"Unknown predict mode: {mode}")
    # -----------------
    # Lightning training_step
    # -----------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")


    # ------
    # Lightning validation_step
    # -------

    def validation_step(self, batch, batch_idx):      
        loss, outputs, aux, mask_output = self._shared_step(batch, batch_idx, "val", return_preds=True)
     
        probs = torch.softmax(outputs, dim=-1)
        labels = batch[-1].long()
        self.val_roc_auc.update(probs, labels)
        
        if self.parameters_dict['debug_val'] and (self.enable_modality_attention or self.mask_enabled) and batch_idx == 0: 
          # Only store FIRST batch of validation
          self.latest_val_sample = {
              "input": batch[0][0].detach().cpu(),
              "pred_mask": mask_output[0].detach().cpu(),
              "gt_mask": batch[1][0].detach().cpu() if self.mask_enabled else None,
              "mod_attn": aux["mod_attn_map"].detach().cpu() if self.enable_modality_attention else None,
          }
              
      
        return loss
    
          
    def on_validation_epoch_end(self):
        val_acc = float(self.trainer.callback_metrics["val_acc"])

        #compute & update  & reset val_auc_roc
        val_auc_roc = self.val_roc_auc.compute()
        self.log("val_roc_auc", val_auc_roc, prog_bar=True)
        self.val_roc_auc.reset()
        
        # nothing to compare yet
        if self.latest_val_sample is None:
            return

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

            best = self.latest_val_sample
            save_dir = self.paths[self.parameters_dict["save_dir"]]

            # --------------------
            # Save masks
            # --------------------
            if self.mask_enabled:
                torch.save(best["pred_mask"], f"{save_dir}/best_pred_mask.pt")
                torch.save(best["input"],     f"{save_dir}/best_input.pt")

                if best["gt_mask"] is not None:
                    torch.save(best["gt_mask"], f"{save_dir}/best_gt_mask.pt")

                # Optional visualization
                if self.parameters_dict.get("debug_val", False):
                    visualize_single_mask_triplet(
                        input_img=best["input"],
                        gt_mask=best["gt_mask"],
                        pred_mask=best["pred_mask"],
                        title_prefix=f"Epoch {self.current_epoch}, best-so-far sample:",
                    )

            # --------------------
            # Save modality attention
            # --------------------
            if self.enable_modality_attention:
                mod = best.get("mod_attn")
                mod_cpu = mod.detach().to(torch.float32).cpu()

                # only get first sample will be same
                vec = mod_cpu[0].view(-1).tolist()
                if mod is not None:
                    torch.save(vec, f"{save_dir}/best_modality_attention.pt")

                if self.parameters_dict.get("debug_val", False):
                    self.print(f"Modality vector (sample 0): {vec}")
    # -----
    #  lightning test step
    # ---------

    def test_step(self, batch, batch_idx):

        mode = self.parameters_dict["test_mode"]
        mc_passes = self.parameters_dict["mc_passes"]

        # handle predictions
        pred_result = self.predict_custom(batch, mode=mode, mc_passes=mc_passes)
        # handle normal / tta result
        if isinstance(pred_result, torch.Tensor):
            outputs = pred_result
        # handle MC or TTA_MC which return (mean, var)
        elif isinstance(pred_result, tuple):
            outputs, variance = pred_result
            self.log("test_uncertainty", variance.mean().item(), prog_bar=False)
        else:
            raise RuntimeError("Unexpected predict_custom output.")

        # compute accuracy
        labels = batch[-1]
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_acc", acc, prog_bar=True) # todo add more test data

        return acc


    #update to train previously frozen
    def _sync_optimizer_new_params(self):
        try:
            if not hasattr(self, "trainer") or not getattr(self, "trainer", None):
                return
            if not self.trainer.optimizers:
                return
            opt = self.trainer.optimizers[0]
        except Exception:
            return

        existing_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        to_add = [p for p in self.model.parameters() if p.requires_grad and id(p) not in existing_ids]
        if not to_add:
            return

        base_lr = self.parameters_dict.get("backbone_unfreeze_lr", 1e-4)
        factor = self.parameters_dict.get("backbone_unfreeze_lr_factor", 0.25)

        # clean, safe LR schedule
        backbone_lr = base_lr * (factor ** self.layers_unfrozen)

        self.layers_unfrozen += 1

        base_wd = opt.param_groups[0].get("weight_decay", 0.0)
        opt.add_param_group({"params": to_add, "lr": backbone_lr, "weight_decay": base_wd})

        print(f"[INFO] Added {len(to_add)} newly-unfrozen params with lr={backbone_lr:.6g} (group={self.layers_unfrozen-1})")

    # ---
    # measure grad norm, pre clip
    # ---
    def on_after_backward(self):
        total_norm = torch.norm(
            torch.stack([p.grad.data.norm(2) for p in self.parameters() if p.grad is not None]), 2
        )
        self.log("grad_norm", total_norm, prog_bar=True)



# -------
# small helpers 
# -----------
#tta helpers
def tta_id(x): return x
def inv_tta_id(x): return x
def tta_flip_lr(x): return torch.flip(x, dims=[-1])
def inv_tta_flip_lr(x): return torch.flip(x, dims=[-1])
def tta_flip_ud(x): return torch.flip(x, dims=[-2])
def inv_tta_flip_ud(x): return torch.flip(x, dims=[-2])
def tta_flip_lrud(x): return torch.flip(torch.flip(x, dims=[-1]), dims=[-2])
def inv_tta_flip_lrud(x): return torch.flip(torch.flip(x, dims=[-1]), dims=[-2])


class GetWeights:
    def __init__(self, module: torch.nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, inp, output):
        try:
            self.features = output.detach().cpu()
        except RuntimeError:
            self.features = output.detach()

    def close(self):
        try:
            self.hook.remove()
        except Exception:
            pass
        self.features = None

#simple vizualizer with less perf impact
def visualize_single_mask_triplet(input_img, gt_mask, pred_mask, title_prefix=""):
    input_img = input_img[0]
    gt_mask = gt_mask.squeeze()
    pred_bin = (torch.sigmoid(pred_mask) > 0.5).squeeze().float().cpu().numpy() # same as used in mask loss calculation, important
    pred_mask= pred_mask.squeeze().float().cpu().numpy()
    
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(input_img, cmap="gray")
    plt.title(f"{title_prefix}Input")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("GT Mask")
    plt.axis("off")

  
    plt.subplot(1, 4, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Pred Mask")
    plt.axis("off")


    plt.subplot(1, 4, 4)
    plt.imshow(pred_bin, cmap="gray")
    plt.title("Pred Bin")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close("all")


def mimic_feat_loss(s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
    s = F.normalize(s_feat.flatten(1), dim=1)
    t = F.normalize(t_feat.flatten(1), dim=1)
    return 1.0 - (s * t).sum(dim=1).mean()


def recon_image_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    # per-image normalization (safe)
    t = target.view(target.size(0), -1)
    t_min = t.min(dim=1)[0].view(-1, 1, 1, 1)
    t_max = t.max(dim=1)[0].view(-1, 1, 1, 1)
    target = (target - t_min) / (t_max - t_min + eps)
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    l1 = F.l1_loss(pred, target)
    ssim_l = 1.0 - pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)
   
    return 0.5 * l1 + 0.5 * ssim_l
