import time
import copy
import gc
from typing import Dict, Optional, Tuple, Any

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
from torchmetrics.classification import MulticlassAUROC
VIZ_FREQUENCY = 10


class LightningSingleModel(pl.LightningModule):   
    def __init__(
        self,
        model,
        method,
        criterion_clf,
        criterion_recon,
        optimizer_fn,
        scheduler_fn=None,      
        device=None,
        parameters_dict=None,
        dataloaders=None,
    ):
        super().__init__()
        # store refs
        self.model = model
        self.method = method
        self.criterion_clf = criterion_clf
        self.criterion_recon = criterion_recon
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.parameters_dict = parameters_dict
        self.dataloaders_ref = dataloaders   # only needed for viz

        # unpack model params
        model_params = parameters_dict[f"{method}_model_parameters"]
        self.model_params = model_params
        self.recon_enabled = model_params["recon_enabled"]
        self.lambda_recon = model_params["lambda_recon"]
        self.mimic_enabled = model_params["mimic_enabled"]
        self.lambda_mimic = model_params["lambda_mimic"]
        self.enable_modality_attention = model_params["enable_modality_attention"]
        self.class_num = parameters_dict["class_num"]



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
        self.debug_first = self.debug_training

        # attach attention hook if needed
        self.attention_hook = None
        if self.debug_training and self.enable_modality_attention and \
           hasattr(model, "modality_attention") and model.modality_attention is not None:
            self.attention_hook = GetWeights(self.model.modality_attention.fc)

        # used for visualization
        self._prepare_viz_batch(dataloaders)


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
          
      #no scheduler
      if self.scheduler_fn is None:
          return optimizer

      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, T_max=self.parameters_dict["num_epochs"]
      )

      return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": self.parameters_dict['control_metric']   
        },
      }



    # --- 
    # Gradual unfreezing
    # ---
    def on_train_epoch_start(self):
        unfreeze_timer = int(self.parameters_dict["unfreeze_timer"])
        if self.parameters_dict["backbone_freeze_on_start"]:
          self.opt_factory.gradual_unfreeze(
              epoch=self.current_epoch,
              unfreeze_every_n_epochs=unfreeze_timer
          )

    # --------------------------------
    # forward 
    # --------------------------------
    def forward(self, x, masks=None):
        return self.model(x, masks)


    # --------------------------------
    # shared step (train + val)
    # --------------------------------
    def _shared_step(self, batch, batch_idx, phase):
        is_train = phase == "train"

        # unpack
        if len(batch) == 3:
            inputs, masks, labels = batch
        else:
            inputs, labels = batch
            masks = None

        # aux loss weight scheduling, drops off towards epoch 
        if self.parameters_dict['use_simple_aux_loss_scheduling']:
          aux_w = max(0.0, 1 - self.current_epoch / self.parameters_dict["aux_loss_weight_epoch_limit"])
        

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        if masks is not None:
          masks = masks.to(self.device)

        #ensure label dtype for classification
        labels = labels.long()
        # DEBUG only first few
        if self.current_epoch == 0 and batch_idx < 5 and self.debug_first and is_train:
            print(f"[DEBUG] Input Stats: Min={inputs.min():.4f}, Max={inputs.max():.4f}, Mean={inputs.mean():.4f}, Std={inputs.std():.4f}")
            if masks is not None:
                print(f"[DEBUG] Mask Stats: Min={masks.min():.4f}, Max={masks.max():.4f}, Mean={masks.mean():.4f}")

        # forward
        outputs, aux, mask_output = self(inputs, masks)
        recon_feats = aux.get("recon_feats", []) if aux is not None else []
        proj_pairs = aux.get("proj_pairs", None) if aux is not None else None

        # classification loss (label smoothing)
        if self.label_smoother is not None and is_train:
            smoothed = self.label_smoother(outputs, labels)
            clf_loss = self.criterion_clf(outputs, smoothed)
        else:
            clf_loss = self.criterion_clf(outputs, labels)

        batch_loss = clf_loss

        # ----------------------
        # mask losses 
        # ----------------
        if self.mask_enabled and mask_output is not None and masks is not None:
            if mask_output.shape[-2:] != masks.shape[-2:]:
                mask_out_resized = F.interpolate(mask_output,
                                                 size=masks.shape[-2:],
                                                 mode="bilinear",
                                                 align_corners=False)
            else:
                mask_out_resized = mask_output

            mask_loss = self.mask_criterion(mask_out_resized, masks)
            batch_loss = batch_loss + self.lambda_mask * mask_loss

            # dice metric
            with torch.no_grad():
                pred_bin = (torch.sigmoid(mask_out_resized) > 0.5).float()
                gt_bin = (masks > 0.5).float()
                inter = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
                union = pred_bin.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3))
                dice = ((2 * inter + 1e-6) / (union + 1e-6))  # per-sample dice

                # update metrics
                if is_train:
                    # update train mask dice total and count
                    self.train_mask_dice.update(dice.mean())
                else:
                    self.val_mask_dice.update(dice.mean())
        # ----------------
        # recon + mimic
        # -----------------
        recon_loss_val = torch.tensor(0.0, device=self.device)
        mimic_loss_val = torch.tensor(0.0, device=self.device)

        if self.recon_enabled:
            for idx_r, r in enumerate(recon_feats):
                if r is None:
                    continue

                pred_r = r
                if idx_r == 0:
                    target = inputs
                else:
                    target = F.interpolate(inputs, scale_factor=1 / (2 ** idx_r),
                                           mode="bilinear", align_corners=False)

                if pred_r.size(1) == 1 and target.size(1) > 1:
                    target_used = target.mean(dim=1, keepdim=True)
                else:
                    target_used = target

                if pred_r.shape[-2:] != target_used.shape[-2:]:
                    pred_r = F.interpolate(pred_r,
                                           size=target_used.shape[-2:],
                                           mode="bilinear",
                                           align_corners=False)

                recon_loss_val = recon_loss_val + recon_image_loss(pred_r, target_used)

            # mimic
            if self.mimic_enabled and proj_pairs is not None and len(proj_pairs) >= 4:
                p1, p1_r, p2, p2_r = proj_pairs[:4]
                mimic_loss_val = mimic_feat_loss(p1, p1_r) + mimic_feat_loss(p2, p2_r)
                  
            if self.parameters_dict['use_simple_aux_loss_scheduling']:
                batch_loss = batch_loss + self.lambda_recon * recon_loss_val * aux_w +  self.lambda_mimic * mimic_loss_val * aux_w
            else: 
                batch_loss = batch_loss + self.lambda_recon * recon_loss_val +  self.lambda_mimic * mimic_loss_val

            # update recon/mimic metrics (recon_loss_val is mean-per-batch; weight by batch size)
            if is_train:
                self.train_recon_loss.update(recon_loss_val.detach())
                self.train_mimic_loss.update(mimic_loss_val.detach())
            else:
                self.val_recon_loss.update(recon_loss_val.detach())
                self.val_mimic_loss.update(mimic_loss_val.detach())
        # ------------------
        # metrics
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
            # log train epoch-level metrics
            self.log("train_mask_dice", self.train_mask_dice, on_epoch=True, prog_bar=True)
            self.log("train_recon_loss", self.train_recon_loss, on_epoch=True, prog_bar=True)
            self.log("train_mimic_loss", self.train_mimic_loss, on_epoch=True, prog_bar=True)
        else:
            self.log("val_mask_dice", self.val_mask_dice, on_epoch=True, prog_bar=True)
            self.log("val_recon_loss", self.val_recon_loss, on_epoch=True, prog_bar=True)
            self.log("val_mimic_loss", self.val_mimic_loss, on_epoch=True, prog_bar=True)

        if phase in ["val", "test"]:
            # convert to probabilities for auc, with softmax
            probs = torch.softmax(outputs, dim=1)
            auc = self.val_auc(probs, labels)

            self.log(
                f"{phase}_auc",
                auc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
    )

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

    # Enable MC dropout
    def mc_enable(self, model: torch.nn.Module):
        self.enable_dropout(model)
        self.set_batchnorm_eval(model)

    # MC dropout prediction
    def predict_mc_dropout(self, model, x, passes=20):
        self.mc_enable(model)
        preds = []
        with torch.no_grad():
            for _ in range(passes):
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                preds.append(probs)
        preds = torch.stack(preds, dim=0)
        return preds.mean(0), preds.std(0)

 

    # ----
    # TTA
    #---- 
    #transform helpers, just flips
    def tta_flip_lr(self, x: torch.Tensor) -> torch.Tensor: return torch.flip(x, dims=[-1]) 
    def tta_flip_ud(self, x: torch.Tensor) -> torch.Tensor: return torch.flip(x, dims=[-2]) 
    def inv_tta_flip_lr(self, x: torch.Tensor) -> torch.Tensor: return torch.flip(x, dims=[-1]) 
    def inv_tta_flip_ud(self, x: torch.Tensor) -> torch.Tensor: return torch.flip(x, dims=[-2])

    #todo could be made to share function with tta_mc
    def predict_tta(self, model, x, transforms=None):
        if transforms is None:
            transforms = [
                self.tta_flip_lr,
                self.tta_flip_ud,
                self.inv_tta_flip_lr,
                self.inv_tta_flip_ud,
            ]

        preds = []
        with torch.no_grad():
            for t in transforms:
                xt = t(x)
                logits = model(xt)[0]
                probs = torch.softmax(logits, dim=1)
                preds.append(probs)

        preds = torch.stack(preds, dim=0)
        return preds.mean(0)

    # ----
    # both tta and mc
    # ---
    def predict_tta_mc(self, model, x, transforms=None, passes=10):
        if transforms is None:
            transforms = [
                self.tta_flip_lr,
                self.tta_flip_ud,
                self.inv_tta_flip_lr,
                self.inv_tta_flip_ud,
            ]

        preds = []

        with torch.no_grad():
            for t in transforms:
                xt = t(x)                      
                self.mc_enable(model)
                for _ in range(passes):
                    logits = model(xt)[0]
                    probs = torch.softmax(logits, dim=1)
                    preds.append(probs)

        preds = torch.stack(preds, dim=0)
        return preds.mean(0), preds.std(0)

    # -----------------
    # Lightning training_step
    # -----------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")


    # ------
    # Lightning validation_step
    # -------
    '''    
        # visualize masks every N epochs
        if self.debug_training and \
           (self.current_epoch + 1) % VIZ_FREQUENCY == 0 and \
           self.viz_inputs_cpu is not None and \
           self.viz_masks_cpu is not None and \
           batch_idx == 0:
            visualize_masks_helper(
                self.model,
                self.viz_inputs_cpu,
                self.viz_masks_cpu,
                self.current_epoch,
                self.num_viz_samples,
                self.device,
            )
    '''  
    def validation_step(self, batch, batch_idx):
        # unpack
        if len(batch) == 3:
            inputs, masks, labels = batch
        else:
            inputs, labels = batch
            masks = None
            
        outputs, aux, mask_output = self(inputs, masks)
        loss = self._shared_step(batch, batch_idx, "val")

        # compute probs so roc-auc can use them
        probs = torch.softmax(outputs, dim=-1)
        # Update ROC-AUC 
        labels = labels.long()
        self.val_roc_auc.update(probs, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # log roc auc
        self.log("val_roc_auc", 
                self.val_roc_auc.compute(), 
                prog_bar=True, on_epoch=True, sync_dist=True)

        return loss
      
    # --- 
    # custom predict function to run differned modes
    # ---
    @torch.no_grad()
    def predict_custom(self, batch, mode="normal", mc_passes=10):
        # unpack batch
        inputs = batch[0]
        labels = batch[-1]
        masks = batch[1] if len(batch) == 3 else None

        if mode == "normal":
            outputs, aux, mask_out = self.model(inputs, masks)
            return outputs

        elif mode == "tta":
            return self.predict_tta(self.model, inputs)

        elif mode == "mc":
            self.mc_enable(self.model)
            preds = torch.stack([self.model(inputs, masks)[0] for _ in range(mc_passes)])
            return preds.mean(0), preds.var(0)

        elif mode == "tta_mc":
            return self.predict_tta_mc(self.model, inputs, passes=mc_passes)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    # --------
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
    # --------
    # teardown hook
    # ---------
    def on_fit_end(self):
        if self.attention_hook is not None:
            self.attention_hook.close()


# -------
# small helpers 
# -----------
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


def visualize_masks_helper(model: torch.nn.Module, inputs_cpu: torch.Tensor, masks_cpu: torch.Tensor,
                           epoch: int, num_samples: int, device: torch.device):
    model.eval()
    with torch.no_grad():
        inputs = inputs_cpu.to(device=device, dtype=torch.float32)
        masks = masks_cpu.to(device=device, dtype=torch.float32)
        _, _, pred_masks = model(inputs, masks)
        if pred_masks.shape[-2:] != masks.shape[-2:]:
            pred_masks = F.interpolate(pred_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        pred_np = (torch.sigmoid(pred_masks).cpu().numpy() > 0.5)
        inputs_np = inputs.cpu().numpy()
        masks_np = masks.cpu().numpy()

    plt.figure(figsize=(12, num_samples * 4))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(inputs_np[i, 0], cmap="gray"); plt.title(f"Input {i}"); plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(masks_np[i, 0], cmap="gray"); plt.title("GT mask"); plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred_np[i, 0], cmap="gray"); plt.title("Pred mask"); plt.axis("off")
    plt.tight_layout(); plt.show(); plt.close("all")


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
    try:
        import pytorch_msssim
        ssim_l = 1.0 - pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)
    except Exception:
        ssim_l = F.l1_loss(pred, target)  # fallback
    return 0.7 * l1 + 0.3 * ssim_l
