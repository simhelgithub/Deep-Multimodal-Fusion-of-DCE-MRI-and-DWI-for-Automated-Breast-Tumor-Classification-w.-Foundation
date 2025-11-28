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

VIZ_FREQUENCY = 10


class LightningSingleModel(pl.LightningModule):   
    def __init__(self, model, method, criterion_clf, criterion_recon,
                 optimizer_fn, device, parameters_dict, dataloaders):
        super().__init__()

        # store refs
        self.model = model
        self.method = method
        self.criterion_clf = criterion_clf
        self.criterion_recon = criterion_recon
        self.optimizer_fn = optimizer_fn
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

        # mask params
        mask_params = model_params["mask_parameters"]
        self.mask_enabled = mask_params["mask"]
        self.lambda_mask = mask_params["lambda_mask"]
        self.mask_loss_type = mask_params["mask_loss_type"]

        # label smoothing
        label_smoothing_enabled = model_params["label_smoothing_enabled"]
        if label_smoothing_enabled:
            alpha = model_params["label_smoothing_alpha"]
            class_num = parameters_dict["class_num"]
            self.label_smoother = LabelSmoothing(class_num, alpha)
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
      return self.optimizer_fn(self.model.parameters())

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

        #ensure label dtype for classification
        labels = labels.to(self.device)
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
                batch_loss = batch_loss + self.lambda_mimic * mimic_loss_val

            batch_loss = batch_loss + self.lambda_recon * recon_loss_val

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

        return batch_loss


    # -----------------
    # Lightning training_step
    # -----------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")


    # ------
    # Lightning validation_step
    # -------
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "val")
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

        return loss


    # --------
    #  lightning test step
    # ---------
    def test_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[-1]
        masks = batch[1] if len(batch) == 3 else None

        outputs, aux_dict, mask_output = self.model(inputs, masks)

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_acc", acc, prog_bar=True)
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
