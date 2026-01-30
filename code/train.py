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
from loss import *
from selector_helpers import *
from torchmetrics.classification import MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix

VIZ_FREQUENCY = 10

class LightningSingleModel(pl.LightningModule):   
    def __init__(
        self,
        model,
        method,
        criterion_clf,
        optimizer_fn,
        scheduler_fn=None,      
        parameters_dict=None,
        paths = None,
    ):
        super().__init__()
        # store refs
        self.model = model
        self.method = method
        self.criterion_clf = criterion_clf
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.parameters_dict = parameters_dict
        self.paths = paths

        # unpack model params .get
        model_params = parameters_dict[f"{method}_model_parameters"]
        self.model_params = model_params
        self.recon_enabled = model_params["recon_enabled"]
        self.lambda_recon = model_params["lambda_recon"]
        self.mimic_enabled = model_params["mimic_enabled"]
        self.lambda_mimic = model_params["lambda_mimic"]
        self.enable_modality_attention = model_params["enable_modality_attention"]
        self.class_num = parameters_dict["class_num"]

        #regularization features
        self.attn_reg_enabled = model_params["attn_reg_enabled"]
        self.lambda_attn_energy = model_params["lambda_attn_energy"]
        self.lambda_feature_consistency = model_params["lambda_feature_consistency"]
        
        self.feat_norm_reg_enabled = model_params["feat_norm_reg_enabled"]
        self.lambda_feat_norm = model_params["lambda_feat_norm"]


        #unfreeze      
        self.use_backbone = model_params['use_backbone']
        self.unfreeze_timer = int(self.parameters_dict["foundation_model_unfreeze_timer"])
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
        self.debug_anomaly =parameters_dict["debug_anomaly"]
        torch.autograd.set_detect_anomaly(self.debug_anomaly) 

        # aux loss weight scheduling, drops off towards epoch
        self.use_aux_loss_sched = parameters_dict['use_simple_aux_loss_scheduling'] 
        self.aux_loss_limit = parameters_dict["aux_loss_weight_epoch_limit"]

        
        # book keeping for some non standard metrics
        self.latest_val_sample = None
        #self.val_mod_attn = None
        self.test_mod_attn = []
        self.test_preds = []
        self.test_targets = []
        self.test_preds_array = []
        self.test_targets_array = []
        #transform helpers for tta
        self.transforms_list = self.transforms_list = [ 
                tta_id,
                tta_flip_lr,
                tta_flip_ud,
                tta_flip_lrud]
        #inverses
        #  [tta_id,inv_tta_id,tta_flip_lr,inv_tta_flip_lr,tta_flip_ud,inv_tta_flip_ud,tta_flip_lrud,inv_tta_flip_lrud]

        # metric objects (track averages across batches/epochs)
        # -------------------
        # TRAIN METRICS
        # -------------------
        self.train_mask_loss   = MeanMetric().cpu()
        self.train_recon_loss  = MeanMetric().cpu()
        self.train_mimic_loss  = MeanMetric().cpu()
        self.train_acc         = MeanMetric().cpu()
        self.train_f1          = MulticlassF1Score(num_classes=self.class_num).cpu()
        # -------------------
        # VALIDATION METRICS
        # -------------------
        self.val_mask_loss     = MeanMetric().cpu()
        self.val_recon_loss    = MeanMetric().cpu()
        self.val_mimic_loss    = MeanMetric().cpu()
        self.val_acc           = MeanMetric().cpu()
        self.val_roc_auc       = MulticlassAUROC(num_classes=self.class_num).cpu()
        self.val_f1            = MulticlassF1Score(num_classes=self.class_num).cpu()
        self.val_precision     = MulticlassPrecision(num_classes=self.class_num).cpu()
        self.val_recall        = MulticlassRecall(num_classes=self.class_num).cpu()
        self.val_confmat       = MulticlassConfusionMatrix(num_classes=self.class_num).cpu()

        self.val_epoch_preds = []
        self.val_epoch_probs = []
        self.val_epoch_labels = []

        # -------------------
        # TEST METRICS
        # -------------------
        self.test_auc          = MulticlassAUROC(num_classes=self.class_num).cpu()
        self.test_f1           = MulticlassF1Score(num_classes=self.class_num).cpu()
        self.test_precision    = MulticlassPrecision(num_classes=self.class_num).cpu()
        self.test_recall       = MulticlassRecall(num_classes=self.class_num).cpu()
        self.test_confmat      = MulticlassConfusionMatrix(num_classes=self.class_num).cpu()
        self.test_acc          = MeanMetric().cpu()

        self.test_acc_per_class = None  

        self.opt_factory = LightningOptimizerFactory(
                model=self.model,
                parameters=parameters_dict,
                model_type=method,
            )
    #---
    # get optimizer
    #---
    '''
    def configure_optimizers(self):
      self.optimizer = self.optimizer_fn(self.model.parameters())
      if self.scheduler_fn is None:
          print('train, no set scheduler')
          return self.optimizer

      sched_ret = self.scheduler_fn(self.optimizer)

      if isinstance(sched_ret, dict):
          # pack optimizer + scheduler dict in Lightning format
          return {
              "optimizer": self.optimizer,
              "lr_scheduler": sched_ret,
          }

      # if a scheduler object was returned, wrap it
      if hasattr(sched_ret, "step"):
          return {
              "optimizer": optimizer,
              "lr_scheduler": {
                  "scheduler": sched_ret,
                  "monitor": self.parameters_dict["control_metric"],
                  "interval": "epoch",
              },
              "grad_clip_val": self.parameters_dict[f'{self.method}_model_parameters']['grad_clip'],
              "gradient_clip_algorithm": self.parameters_dict[f'{self.method}_model_parameters']['gradient_clip_algorithm']
          }
      #fallback 
      print('no scheduler, returning optimizer in train')
      return optimizer
    '''
    def configure_optimizers(self):
        self.optimizer = self.optimizer_fn(self.model.parameters())

        if self.scheduler_fn is None:
            print("train, no set scheduler")
            self.scheduler = None
            return self.optimizer

        sched_ret = self.scheduler_fn(self.optimizer)

        # Store the raw scheduler object for debug/step
        if isinstance(sched_ret, dict) and "scheduler" in sched_ret:
            self.scheduler = sched_ret["scheduler"]
        elif hasattr(sched_ret, "step"):
            self.scheduler = sched_ret
        else:
            self.scheduler = None

        # Return in Lightning format
        if isinstance(sched_ret, dict):
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": sched_ret,
            }
        if hasattr(sched_ret, "step"):
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": sched_ret,
                    "monitor": self.parameters_dict["control_metric"],
                    "interval": "epoch",
                },
                "grad_clip_val": self.parameters_dict[f'{self.method}_model_parameters']['grad_clip'],
                "gradient_clip_algorithm": self.parameters_dict[f'{self.method}_model_parameters']['gradient_clip_algorithm']
            }

        return self.optimizer

    #debug    
    def on_before_optimizer_step(self, optimizer):
      if self.debug_anomaly and self.global_step % 100 == 0:   # print every 100 steps
          for name, p in self.named_parameters():
              if p.grad is not None:
                  print(f"{name} grad norm: {p.grad.norm().item()}")

    # --- 
    # Gradual unfreezing
    # ---
    def on_train_epoch_start(self):
        self.train_mask_loss.reset()
        self.train_recon_loss.reset()
        self.train_mimic_loss.reset()
        self.train_acc.reset()
        self.train_f1.reset()

        # Gradual unfreeze
        if self.backbone_freeze_on_start and self.current_epoch == self.unfreeze_timer: 
            new_params = self.opt_factory.unfreeze_backbone()
            if new_params:
                opt = self.trainer.optimizers[0]
                self.opt_factory.sync_unfrozen_params_to_optimizer(opt, new_params)


    #reset key metrics on epoch to be safe
    def on_validation_epoch_start(self):
        # reset validation metrics to ensure per-epoch accumulation
        try:
            self.val_mask_loss.reset()
            self.val_recon_loss.reset()
            self.val_mimic_loss.reset()
            self.val_precision.reset()
            self.val_recall.reset()
            self.val_roc_auc.reset()
            self.val_f1.reset()
            self.val_confmat.reset()
            self.val_epoch_preds.clear()
            self.val_epoch_probs.clear()
            self.val_epoch_labels.clear()
            self.val_acc.reset()

        except Exception:
            pass

    def on_test_epoch_start(self):
        try:
            self.test_auc.reset()
            self.test_f1.reset()
            self.test_precision.reset()
            self.test_recall.reset()
            self.test_confmat.reset()
            self.test_acc.reset()
        
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
    @torch._dynamo.disable
    def _shared_step(self, batch, batch_idx, phase, return_preds: bool = False):
        is_train = phase == "train"
        if batch_idx == 0 and self.current_epoch == 0:
            print("BATCH INPUT SHAPE:", batch[0].shape)

        # unpack
        #if len(batch) == 3:
        if self.mask_enabled:
            inputs, masks, labels = batch
        else:
            inputs, labels= batch
            masks = None

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        if self.mask_enabled:
            masks = masks.to(self.device)

        # ensure label dtype for classification
        labels = labels.long()

        # DEBUG first data points to see normalization
        if self.debug_first and self.current_epoch == 0 and batch_idx < 5 and is_train:
            print_debug_data(inputs = inputs, masks = masks if masks is not None else None)

       # aux loss weight scheduling, drops off towards epoch
        if self.use_aux_loss_sched:
            aux_w = max(0.0, 1 - self.current_epoch / self.aux_loss_limit)
        else:
            aux_w = 1.0

        # If caller asked for predictions (validation), run forward under no_grad to be explicit/safe
        # forward step
        outputs, aux, mask_output = self(inputs, masks)

        recon_feats = aux.get("recon_feats", []) if aux is not None else []
        proj_pairs = aux.get("proj_pairs", None) if aux is not None else None

        #----
        # Classification
        #----

        # classification loss (label smoothing only during training)
        if self.label_smoother:
            smoothed = self.label_smoother(outputs, labels)
        clf_loss = self.criterion_clf(outputs, smoothed) if is_train else self.criterion_clf(outputs, labels)

        


        batch_loss = clf_loss
        # ---------------------------------------
        # REGULARIZATION BLOCK 
        # -------------------------------------
        
        attn_energy_loss = torch.tensor(0.0, device=self.device)
        feature_consistency_loss = torch.tensor(0.0, device=self.device)
        feat_norm_loss = torch.tensor(0.0, device=self.device)

        raw_feats = aux.get("raw_feats", None)

        # ---- REGULARIZATION ----
        if self.attn_reg_enabled:
            attn_energy_loss = compute_attn_energy_loss(aux, self.lambda_attn_energy, self.device)
            feature_consistency_loss = compute_feature_consistency_loss(aux, self.lambda_feature_consistency, self.device)
            batch_loss+= attn_energy_loss * self.lambda_attn_energy + feature_consistency_loss* self.lambda_feature_consistency if is_train else 0.0
        if self.feat_norm_reg_enabled:
            feat_norm_loss = compute_feat_norm_loss(aux, self.device)
            batch_loss+= feat_norm_loss * self.lambda_feat_norm if is_train else 0.0


        # ----------------------
        # Mask loss
        # ----------------
        mask_out_resized = None
        mask_loss = 0.0
        if self.mask_enabled: #and mask_output is not None and masks is not None:
            if mask_output.shape[-2:] != masks.shape[-2:]:
                #print('warning, mask rezised')
                mask_out_resized = F.interpolate(mask_output,
                                                 size=masks.shape[-2:],
                                                 mode="bilinear",
                                                 align_corners=False)
            else:
                mask_out_resized = mask_output

            mask_loss = update_mask_metric(self,mask_output, masks)
            batch_loss += self.lambda_mask * mask_loss if is_train else 0.0
                    

        # ----------------
        # recon + mimic
        # -----------------
        recon_loss_val = torch.tensor(0.0, device=self.device)
        mimic_loss_val = torch.tensor(0.0, device=self.device)
        # ---- Auxiliary losses ----
        if self.recon_enabled and aux_w > 0.0:
          recon_loss_val, mimic_loss_val = self.compute_aux_losses(aux, inputs, aux.get("proj_pairs", None), aux_w, is_train)
          # Update metrics
          if is_train:
              self.train_recon_loss.update(recon_loss_val.detach())
              self.train_mimic_loss.update(mimic_loss_val.detach())
              batch_loss += (
                  self.lambda_recon * recon_loss_val * aux_w  + 
                  self.lambda_mimic * mimic_loss_val * aux_w
              )  if is_train else 0.0
          else:
              self.val_recon_loss.update(recon_loss_val.detach())
              self.val_mimic_loss.update(mimic_loss_val.detach())


        # -------------------
        # Compute batch accuracy safely
        # -------------------

        preds = outputs.argmax(dim=1)
        batch_acc = (preds == labels).float().mean()
        # -------------------
        # Update metrics
        # -------------------
        if is_train or phase == 'val':
          update_metrics(self,preds, outputs, labels, mask_loss, recon_loss_val, mimic_loss_val, phase=phase)

          # -------------------
          # Log aggregated metrics (MeanMetric objects)
          # -------------------
          log_losses(self, batch_loss, batch_acc, phase=phase)
        # -------------------
        # Optional: return detached outputs
        # -------------------
        if return_preds:
            return batch_loss.detach(), outputs.detach(), aux, mask_out_resized

        return batch_loss
        
    #---
    # Compuite aux lossees
    #--

    def compute_aux_losses(self, aux, inputs, proj_pairs, aux_w, is_train):
        """Compute reconstruction and mimic losses with optional weighting."""
        recon_feats = aux.get("recon_feats", []) if aux is not None else []
        proj_pairs = aux.get("proj_pairs", None) if aux is not None else None

        recon_loss_val = torch.tensor(0.0, device=self.device)
        mimic_loss_val = torch.tensor(0.0, device=self.device)

        if aux_w <= 0.0:
            return recon_loss_val, mimic_loss_val

        # ---- Reconstruction Loss ----
        for pred_r in recon_feats:
            if pred_r is None:
                continue
            target = inputs
            pred_r_upsampled = F.interpolate(pred_r, size=target.shape[-2:], mode="bilinear", align_corners=False)
            if pred_r_upsampled.size(1) == 1 and target.size(1) > 1:
                target = target.mean(dim=1, keepdim=True)

            recon_loss_val += recon_image_loss(pred_r_upsampled, target)

        # ---- Mimic Loss ----
        if self.mimic_enabled and proj_pairs is not None and len(proj_pairs) >= 4:
            p1, p1_r, p2, p2_r = proj_pairs[:4]
            mimic_loss_val = mimic_feat_loss(p1, p1_r) + mimic_feat_loss(p2, p2_r)

        # ---- Weight by aux_w (training only) ----
        if is_train:
            recon_loss_val = recon_loss_val * self.lambda_recon * aux_w
            mimic_loss_val = mimic_loss_val * self.lambda_mimic * aux_w

        return recon_loss_val, mimic_loss_val


    # ----
    # Dropout (only during training)
    # ---
    @torch._dynamo.disable
    def enable_dropout(self, model: torch.nn.Module):
      for m in model.modules():
          if isinstance(m, torch.nn.Dropout):
              m.train()
    
    
    # ---
    # MC dropout
    # ---
    #helper for setting batchnorm to eval
    @torch._dynamo.disable
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

    @torch._dynamo.disable
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

    # ---- MC dropout prediction ----
    @torch._dynamo.disable
    def predict_mc_dropout(self, model, x, masks=None, passes=20, return_aux=False):
        orig_states = self._get_module_train_states(model)
        self.mc_enable(model)

        preds = []
        aux_list = []
        with torch.no_grad():
            for _ in range(passes):
                if masks is not None:
                    logits, aux, _ = model(x, masks)
                else:
                    logits, aux, _ = model(x)
                probs = torch.softmax(logits, dim=1)
                preds.append(probs)
                if return_aux:
                    aux_list.append(aux)

        preds = torch.stack(preds, dim=0)
        mean_probs = preds.mean(0)
        std_probs = preds.std(0)

        self._restore_module_train_states(model, orig_states)

        if return_aux:
            # Return last aux dict (could also return mean over aux)
            return mean_probs, std_probs, aux_list[-1]
        return mean_probs, std_probs


    # ---- TTA prediction ----
    @torch._dynamo.disable
    def predict_tta(self, model, x, masks=None, transforms=None, return_aux=False):
        if transforms is None:
            transforms = self.transforms_list

        preds = []
        aux_list = []

        with torch.no_grad():
            for t in transforms:
                xt = t(x=x)
                if masks is not None:
                    logits, aux, _ = model(xt, masks)
                else:
                    logits, aux, _ = model(xt)
                probs = torch.softmax(logits, dim=1)
                preds.append(probs)
                if return_aux:
                    aux_list.append(aux)

        preds = torch.stack(preds, dim=0)
        mean_probs = preds.mean(0)

        if return_aux:
            return mean_probs,preds.std(0), aux_list[-1]
        return mean_probs, preds.std(0)


    # ---- TTA + MC dropout ----
    @torch._dynamo.disable
    def predict_tta_mc(self, model, x, masks=None, transforms=None, passes=10, return_aux=False):
        orig_states = self._get_module_train_states(model)
        self.mc_enable(model)
        if transforms is None:
            transforms = self.transforms_list

        preds = []
        aux_last = None

        with torch.no_grad():
            for t in transforms:
                xt = t(x=x)
                for _ in range(passes):
                    if masks is not None:
                        logits, aux, _ = model(xt, masks)
                    else:
                        logits, aux, _ = model(xt)
                    probs = torch.softmax(logits, dim=1)
                    preds.append(probs)
                    if return_aux:
                        aux_last = aux

        preds = torch.stack(preds, dim=0)
        mean_probs = preds.mean(0)
        std_probs = preds.std(0)

        self._restore_module_train_states(model, orig_states)

        if return_aux:
            return mean_probs, std_probs, aux_last
        return mean_probs, std_probs

      

    #---
    # calls the chosen test mode
    #--

    def predict_custom(self, batch, mode="normal", mc_passes=10, return_aux=False):
        inputs = batch[0]
        labels = batch[-1]
        masks = batch[1] if len(batch) == 3 else None

        if mode == "normal":
            outputs, aux, mask_out = self.model(inputs, masks)

            if return_aux:
                aux = detach_aux(aux)
                return outputs, aux
            return outputs

        elif mode == "tta":
            return self.predict_tta(self.model, inputs, masks, return_aux=return_aux)

        elif mode == "mc":
            return self.predict_mc_dropout(self.model, inputs, masks, passes=mc_passes, return_aux=return_aux)

        elif mode == "tta_mc":
            return self.predict_tta_mc(self.model, inputs, masks, passes=mc_passes, return_aux=return_aux)

        else:
            raise ValueError(f"Unknown predict mode: {mode}")

    # -----------------
    # Lightning training_step
    # -----------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    @torch._dynamo.disable
    def on_train_epoch_end(self):
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_mask_loss", self.train_mask_loss.compute())
        self.log("train_recon_loss", self.train_recon_loss.compute())
        self.log("train_mimic_loss", self.train_mimic_loss.compute())

        self.train_f1.reset()
        self.train_mask_loss.reset()
        self.train_recon_loss.reset()
        self.train_mimic_loss.reset()


    # ------
    # Lightning validation_step
    # -------
    def validation_step(self, batch, batch_idx):      
        loss, outputs, aux, mask_output = self._shared_step(batch, batch_idx, "val", return_preds=True)
        aux = detach_aux(aux)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        self.val_epoch_probs.append(probs.detach().cpu())
        self.val_epoch_preds.append(preds.detach().cpu())
        self.val_epoch_labels.append(batch[-1].detach().cpu())


        if self.parameters_dict['debug_val'] and (self.enable_modality_attention or self.mask_enabled) and batch_idx == 0: 
          # Only store FIRST batch of validation
          self.latest_val_sample = {
              "input": batch[0][0].detach().cpu(),
              "pred_mask": mask_output.detach().cpu() if self.mask_enabled else None,
              "gt_mask": batch[1][0].detach().cpu() if self.mask_enabled else None,
              "mod_attn": aux["mod_attn_map"].detach().cpu() if self.enable_modality_attention else None,
          }
              
        return loss
        
    @torch._dynamo.disable
    def on_validation_epoch_end(self):
        if len(self.val_epoch_labels) == 0:
            return

        probs = torch.cat(self.val_epoch_probs, dim=0).cpu()
        preds = torch.cat(self.val_epoch_preds, dim=0).cpu()
        labels = torch.cat(self.val_epoch_labels, dim=0).cpu()

        self.val_roc_auc.cpu()
        self.val_confmat.cpu()
        self.val_f1.cpu()

        self.val_roc_auc.update(probs, labels.to(dtype=torch.long))
        self.val_confmat.update(preds, labels)
        self.val_f1.update(preds, labels)


        self.log("val_roc_auc", self.val_roc_auc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_mask_loss", self.val_mask_loss.compute())
        self.log("val_recon_loss", self.val_recon_loss.compute())
        self.log("val_mimic_loss", self.val_mimic_loss.compute())
        # nothing to compare yet
        if self.latest_val_sample is None:
            return
        # --------------------
        # Visualize masks
        # --------------------
        if self.mask_enabled:
            # Optional visualization
            if self.parameters_dict["debug_val"] and self.current_epoch % VIZ_FREQUENCY == 0:
                visualize_single_mask_triplet(
                    input_img=self.latest_val_sample["input"],
                    gt_mask=self.latest_val_sample["gt_mask"],
                    pred_mask=self.latest_val_sample["pred_mask"],
                    title_prefix=f"Epoch {self.current_epoch}, sample:",
                )
            
        self.val_roc_auc.reset()
        self.val_confmat.reset()
        self.val_f1.reset()
        self.val_mask_loss.reset()
        self.val_recon_loss.reset()
        self.val_mimic_loss.reset()
        self.val_epoch_probs.clear()
        self.val_epoch_preds.clear()
        self.val_epoch_labels.clear()
        return super().on_validation_epoch_end()
            
    def on_test_epoch_start(self):
        torch.cuda.empty_cache()
        self.test_f1.cpu() 
        self.test_precision.cpu()
        self.test_recall.cpu()
        self.test_confmat.cpu()
        

    @torch._dynamo.disable 
    def test_step(self, batch, batch_idx):
        mode = self.parameters_dict["test_mode"]
        mc_passes = self.parameters_dict["mc_passes"]

        # ---- prediction ----
        if mode == "normal":
          outputs, aux = self.predict_custom(batch=batch, mode=mode, mc_passes=mc_passes,  return_aux=True)
        else:
          outputs, variance, aux = self.predict_custom(batch=batch, mode=mode, mc_passes=mc_passes, return_aux=True)

        aux = detach_aux(aux)

        # log MC/TTA uncertainty
        if mode in ['mc', 'tta', 'tta_mc']:
            self.log("test_uncertainty_mean", variance.mean(), prog_bar=False)

        labels = batch[-1].long().cpu()
        probs = torch.softmax(outputs, dim=1).cpu()   
        preds = probs.argmax(dim=1).cpu()

        # --- store raw probs for ROC ---
        self.test_preds.append(probs.detach().cpu())
        self.test_targets.append(labels.detach().cpu())

        # --- modality attention: take mean over batch ---
        if self.enable_modality_attention:
            mod = aux["mod_attn_map"]
            if mod is not None:
                # mod shape: (B, num_modalities) 
                # take mean over batch dimension
                mean_mod = mod.detach().cpu().mean(dim=0)  # shape: (num_modalities,)
                if not hasattr(self, "test_mod_attn"):
                    self.test_mod_attn = []
                self.test_mod_attn.append(mean_mod.float())

        batch_acc = (preds == labels).float().mean()

        # ---- update metrics  ----
        self.test_acc.update(batch_acc)  
        self.test_auc.update(probs, labels)
        self.test_f1.update(preds, labels)
        self.test_precision.update(preds, labels)
        self.test_recall.update(preds, labels)
        self.test_confmat.update(preds, labels)
        
        return preds

    def on_test_epoch_end(self):
        # compute main metrics
        self.log("test_acc", self.test_acc.compute())
        self.log("test_auc", self.test_auc.compute())
        self.log("test_f1", self.test_f1.compute())
        self.log("test_precision", self.test_precision.compute())
        self.log("test_recall", self.test_recall.compute())

        # compute confusion matrix and per-class accuracy
        confmat = self.test_confmat.compute()  # shape: [num_classes, num_classes]
        per_class_acc = confmat.diag() / confmat.sum(1).clamp(min=1)  # avoid div by 0


        for i, acc in enumerate(per_class_acc):
            self.log(f"test_acc_class_{i}", acc, prog_bar=True)
        self.test_acc_per_class = per_class_acc.cpu().numpy() 

        # reset metrics
        self.test_acc.reset()
        self.test_auc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_confmat.reset()

        
        # concatenate stored preds + labels
        self.test_preds_array.append(torch.cat(self.test_preds, dim=0).numpy())
        self.test_targets_array.append(torch.cat(self.test_targets, dim=0).numpy())



        # modality attention
        if self.enable_modality_attention and self.test_mod_attn:
            # stack per-batch mean vectors
            self.test_mod_attn = torch.stack(self.test_mod_attn, dim=0)  # shape: [num_batches, num_modalities]
            
        # clean up
        self.test_preds.clear()
        self.test_targets.clear()


    def on_after_backward(self):
        if self.global_step % 100 == 0:
            # ---- total grad norm ----
            total_norm = torch.norm(
                torch.stack([p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None]), 2
            )

            # ---- backbone grad norm ----
            backbone_params = list(self.model.model.backbone.parameters()) if self.use_backbone else []
            backbone_grads = [p.grad for p in backbone_params if p.grad is not None]

            if backbone_grads:
                backbone_norm = torch.norm(torch.stack([g.detach().norm(2) for g in backbone_grads]), 2)
            else:
                backbone_norm = torch.tensor(0.0, device=total_norm.device)

            # ---- check if backbone params are in optimizer and get LR ----
            #if self.debug_scheduler:
            if self.debug_anomaly:
              backbone_in_opt = []
              backbone_lrs = set()

              for group in self.optimizer.param_groups:
                  group_params = set(group['params'])
                  for p in backbone_params:
                      if p in group_params:
                          backbone_in_opt.append(True)
                          backbone_lrs.add(group['lr'])
                      else:
                          backbone_in_opt.append(False)
              # ---- summarize ----
              percent_in_opt = 100.0 * sum(backbone_in_opt) / len(backbone_in_opt) if len(backbone_in_opt) > 0 else 0
              lr_list = sorted(list(backbone_lrs))
              print(f"[Backbone] Any params in optimizer? {percent_in_opt}, LRs: {lr_list}")
            
            # ---- log ----
            self.log("grad_norm", total_norm, prog_bar=True)
            self.log("backbone_grad_norm", backbone_norm, prog_bar=True)
#--- 
# Deatach aux helper
# ---
def detach_aux(aux):
    if aux is None:
        return None
    return {k: (v.detach() if torch.is_tensor(v) else v)
            for k, v in aux.items()}
# -------------------
# Helper: Update metrics
# -------------------


@torch._dynamo.disable
def update_metrics(self, preds, logits, labels, mask_loss_val, recon_loss_val, mimic_loss_val, phase="train"):
    """
    Update all metrics for a given phase.
    Args:
        preds:  model predictions
        labels: ground truth class indices (same spatial shape as logits)
        phase: "train" | "val" | "test"
    """
    labels = labels.long()
    if phase == "train":
        self.train_f1.update(preds, labels)
        self.train_recon_loss.update(recon_loss_val.detach())
        self.train_mask_loss.update(mask_loss_val.detach()) if self.mask_enabled else self.train_mask_loss.update(0.0)
        self.train_mimic_loss.update(mimic_loss_val.detach())
    elif phase == "val":
              
        # Compute softmax probabilities for AUROC
        self.val_mask_loss.update(mask_loss_val.detach())
        self.val_mimic_loss.update(mimic_loss_val.detach())
        self.val_recon_loss.update(recon_loss_val.detach())
  
  
    #not currently used fix if I want to use it
    elif phase == "test":                  
        # Compute softmax probabilities for AUROC
        probs = torch.softmax(logits, dim=1)  # shape (B, C, H, W) or (B, C, D, H, W)
        self.test_f1.update(preds, labels)
        self.test_auc.update(probs, labels)
        
    else:
        raise ValueError(f"Unknown phase {phase}")

    


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

    # --- FIX: reduce 3D masks to 2D for visualization ---
    if pred_mask.ndim == 3:  # (1, H, W)
        pred_mask = pred_mask.squeeze(0)

    elif pred_mask.ndim == 4:  # (H, W, D)
        pred_mask = pred_mask[pred_mask.shape[-1] // 2]  # center slice

    elif pred_mask.ndim == 5:  # (1, H, W, D)
        pred_mask = pred_mask.squeeze(0)[pred_mask.shape[-1] // 2]

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
    #plt.imshow(pred_bin, cmap="Grays_r") #invert color scale 
    plt.imshow(pred_bin, cmap="gray")  
    plt.title("Pred Bin")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close("all")

def compute_attn_energy_loss(aux, device):
    """
    L1 energy consistency loss on attention maps
    """
    attn_map = aux.get("mask_attn_map", None)
    if attn_map is not None:
        loss = attn_map.abs().mean()
    else:
        loss = torch.tensor(0.0, device=device)
    return loss
def compute_feature_consistency_loss(aux, device):
    """
    Consistency loss between projected features from multiple views
    """
    if "proj_pairs" not in aux or aux["proj_pairs"] is None:
        return torch.tensor(0.0, device=device)

    p1, p1_r, p2, p2_r = aux["proj_pairs"]

    # Upsample p2 to match p1
    p2_up = F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)

    # Normalize features
    p1_norm = p1 / (p1.norm(dim=1, keepdim=True) + 1e-6)
    p2_norm = p2_up / (p2_up.norm(dim=1, keepdim=True) + 1e-6)

    loss = F.mse_loss(p1_norm, p2_norm)
    return loss

    
def compute_feat_norm_loss(aux, device):
    """
    L2 norm of intermediate features for regularization
    """
    raw_feats = aux.get("raw_feats", None)
    if raw_feats is not None:
        loss = sum([f.pow(2).mean() for f in raw_feats]) 
    else:
        loss = torch.tensor(0.0, device=device)
    return loss


def mimic_feat_loss(s_feat, t_feat, eps=1e-6):
    t_feat = t_feat.detach()  # detach teacher/fused
    s = F.normalize(s_feat.flatten(1), dim=1)
    t = F.normalize(t_feat.flatten(1), dim=1)
    cos = (s * t).sum(dim=1)
    return (1.0 - cos.clamp(-1+eps, 1-eps)).mean()

#charbonnier recon loss
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target)**2 + eps**2))
def recon_image_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred = pred.clamp(0,1)
    target = target.clamp(0,1)
    loss = charbonnier_loss(pred, target)
    return loss


    
@torch._dynamo.disable
def update_mask_metric(self, preds, masks):
  return self.mask_criterion(preds, masks)
# -------------------
# Log aggregated metrics (MeanMetric objects)
# -------------------
@torch._dynamo.disable
def log_losses(self, batch_loss, batch_acc, phase):    
  if phase=='train':
      self.log(f"train_loss", batch_loss, prog_bar=True, on_step=False, on_epoch=True)
      self.log(f"train_acc", batch_acc, prog_bar=True, on_step=False, on_epoch=True)

  elif phase == 'val':
      self.log(f"val_loss", batch_loss, prog_bar=True, on_step=False, on_epoch=True)
      self.log(f"val_acc", batch_acc, prog_bar=True, on_step=False, on_epoch=True)
      self.log("val_mask_loss", self.val_mask_loss, on_step=False, on_epoch=True, prog_bar=True)
      self.log("val_recon_loss", self.val_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
      self.log("val_mimic_loss", self.val_mimic_loss, on_step=False, on_epoch=True, prog_bar=True)

#---
# Debug for data, shows if input is normalized
#--
@torch._dynamo.disable
def print_debug_data(inputs, masks):
    print(f"[DEBUG] Input Stats: Min={inputs.min():.4f}, Max={inputs.max():.4f}, Mean={inputs.mean():.4f}, Std={inputs.std():.4f}")
    if masks is not None:
        print(f"[DEBUG] Mask Stats: Min={masks.min():.4f}, Max={masks.max():.4f}, Mean={masks.mean():.4f}")

