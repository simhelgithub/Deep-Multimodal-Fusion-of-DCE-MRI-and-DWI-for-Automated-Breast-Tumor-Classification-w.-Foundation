import time
import copy
import gc
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from loss import *
from selector_helpers import *

# Visualize frequency (epochs)
VIZ_FREQUENCY = 10



def train_model(model, dataloaders, method, criterion_clf, criterion_recon, optimizer, device, parameters):
    # ----
    # setup
    # ---
    model_params = parameters[f"{method}_model_parameters"]
    num_epochs = parameters["num_epochs"]
    recon_enabled = model_params["recon_enabled"]
    lambda_recon = model_params["lambda_recon"]
    mimic_enabled = model_params["mimic_enabled"]
    lambda_mimic = model_params["lambda_mimic"]
    enable_modality_attention = model_params["enable_modality_attention"]

    # mask params
    mask_params = model_params["mask_parameters"]
    mask_enabled = mask_params["mask"] 
    lambda_mask = mask_params["lambda_mask"]
    mask_loss_type = mask_params["mask_loss_type"]

    # label smoothing
    label_smoothing_enabled = model_params["label_smoothing_enabled"]
    label_smoother = None
    if label_smoothing_enabled:
        alpha = model_params["label_smoothing_alpha"]
        class_num = parameters["class_num"]
        label_smoother = LabelSmoothing(class_num, alpha)

    # mask loss selection
    mask_criterion = mask_criterion_selector(parameters, method)

    # debug / viz flags
    debug_training = parameters["debug_training"]
    ENABLE_MASK_VIZ = debug_training
    show_attention = debug_training
    debug_first = debug_training
    # prepare AMP scaler local to this training call
    scaler = torch.cuda.amp.GradScaler()

    # best model bookkeeping
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = -float("inf")

    # hook to be able to visualize modality weights
    attention_hook = None
    if show_attention and enable_modality_attention and hasattr(model, "modality_attention") and model.modality_attention is not None:
        attention_hook = GetWeights(model.modality_attention.fc)

    # optional fixed batch for visualization
    viz_inputs_cpu = viz_masks_cpu = viz_labels_cpu = None
    num_viz_samples = 0
    if ENABLE_MASK_VIZ and "val" in dataloaders and len(dataloaders["val"]) > 0:
        example = next(iter(dataloaders["val"]))
        if len(example) == 3:
            viz_inputs_cpu, viz_masks_cpu, viz_labels_cpu = example
        elif len(example) == 2:
            viz_inputs_cpu, viz_labels_cpu = example
            viz_masks_cpu = None
        if viz_inputs_cpu is not None:
            num_viz_samples = min(4, viz_inputs_cpu.size(0))
            viz_inputs_cpu = viz_inputs_cpu[:num_viz_samples].cpu()
            if viz_masks_cpu is not None:
                viz_masks_cpu = viz_masks_cpu[:num_viz_samples].float().cpu()
            if viz_labels_cpu is not None:
                viz_labels_cpu = viz_labels_cpu[:num_viz_samples].cpu()

    # histories
    train_acc_history, train_loss_history, val_acc_history, val_loss_history = [], [], [], []

    model.to(device)


    #---------------------------------
    # ---------- main loop -----------
    #---------------------------------
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        for phase in ("train", "val"):
            is_train = phase == "train"
            if is_train:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_mask_dice = 0.0
            mask_count = 0
            per_class_correct = {}
            per_class_count = {}

            running_recon_loss = 0.0
            running_mimic_loss = 0.0

            # iterate batches
            for batch_idx, batch in enumerate(dataloaders[phase]):
                if len(batch) == 3:
                    inputs, masks, labels = batch
                elif len(batch) == 2:
                    inputs, labels = batch
                    masks = None
                else:
                    raise ValueError(f"Unexpected batch tuple length: {len(batch)}")

                # move to device & type safety
                inputs = inputs.float().to(device, non_blocking=True)
                labels = labels.long().to(device, non_blocking=True)
                if masks is not None:
                    masks = masks.float().to(device, non_blocking=True)

                # DEBUG: Print input stats for the first batch of the first epoch
                if epoch == 0 and batch_idx == 0 and phase == 'train' and debug_first:
                    print(f"[DEBUG] Input Stats: Min={inputs.min():.4f}, Max={inputs.max():.4f}, Mean={inputs.mean():.4f}, Std={inputs.std():.4f}")
                    if masks is not None:
                        print(f"[DEBUG] Mask Stats: Min={masks.min():.4f}, Max={masks.max():.4f}, Mean={masks.mean():.4f}")


                # zero grads only in train
                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                # forward
                with torch.set_grad_enabled(is_train):
                    with torch.amp.autocast("cuda"):
                        outputs, aux, mask_output = model(inputs, masks)
                        # aux may contain proj_pairs and recon_feats
                        recon_feats = aux.get("recon_feats", []) if aux is not None else []
                        proj_pairs = aux.get("proj_pairs", None) if aux is not None else None

                        # check shape match
                        if outputs.size(0) != labels.size(0):
                            print(f"Warning: output batch ({outputs.size(0)}) != labels ({labels.size(0)}) — skipping batch.")
                            continue

                        # classification loss (label smoothing if provided)
                        if label_smoother is not None and is_train:
                            smoothed = label_smoother(outputs, labels)
                            clf_loss = criterion_clf(outputs, smoothed) 
                        else:
                            clf_loss = criterion_clf(outputs, labels)

                        batch_loss = clf_loss

                        # mask losses & metrics
                        if mask_output is not None and mask_enabled:
                            # handle shape mismatch by resizing predicted mask to gt size
                            if mask_output is None:
                                print("Warning: mask enabled but model returned no mask_output")
                            else:
                                if mask_output.shape[-2:] != masks.shape[-2:]:
                                    mask_out_resized = F.interpolate(mask_output, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                                else:
                                    mask_out_resized = mask_output

                                mask_loss = mask_criterion(mask_out_resized, masks)
                                batch_loss = batch_loss + lambda_mask * mask_loss

                                # dice metric (hard threshold) aggregated
                                with torch.no_grad():
                                    pred_bin = (torch.sigmoid(mask_out_resized) > 0.5).float()
                                    gt_bin = (masks > 0.5).float()
                                    inter = (pred_bin * gt_bin).sum(dim=(1,2,3))
                                    union = pred_bin.sum(dim=(1,2,3)) + gt_bin.sum(dim=(1,2,3))
                                    per_sample_dice = ((2. * inter + 1e-6) / (union + 1e-6)).cpu()
                                    running_mask_dice += per_sample_dice.sum().item()
                                    mask_count += inputs.size(0)

                        # recon & mimic
                        recon_loss_val = torch.tensor(0.0, device=device)
                        mimic_loss_val = torch.tensor(0.0, device=device)
                        if recon_enabled:
                            # recon_feats expected to be list like [r1, r2]
                            # compute recon_loss by comparing decoded preds to inputs 
                            for idx_r, r in enumerate(recon_feats):
                                if r is None:
                                    continue
                                pred_r = r
                                # choose target: full-size or half-size depending on index
                                if idx_r == 0:
                                    target = inputs
                                else:
                                    target = F.interpolate(inputs, scale_factor=1 / (2 ** idx_r), mode="bilinear", align_corners=False)
                                # if pred channel == 1 but target has many channels, average
                                if pred_r.size(1) == 1 and target.size(1) > 1:
                                    target_used = target.mean(dim=1, keepdim=True)
                                else:
                                    target_used = target

                                if pred_r.shape[-2:] != target_used.shape[-2:]:
                                    pred_r = F.interpolate(pred_r, size=target_used.shape[-2:], mode="bilinear", align_corners=False)

                                
                                recon_loss_val = recon_loss_val + recon_image_loss(pred_r, target_used)

                            # mimic loss from proj_pairs (p1, p1_r, p2, p2_r)
                            if mimic_enabled and proj_pairs is not None and len(proj_pairs) >= 4:
                                p1, p1_r, p2, p2_r = proj_pairs[:4]
                                mimic_loss_val = mimic_feat_loss(p1, p1_r) + mimic_feat_loss(p2, p2_r)
                                batch_loss = batch_loss + lambda_mimic * mimic_loss_val

                            batch_loss = batch_loss + lambda_recon * recon_loss_val
                            running_recon_loss += recon_loss_val.item() * inputs.size(0)
                            running_mimic_loss += mimic_loss_val.item() * inputs.size(0)

                        # end with batch_loss (tensor)
                    # end autocast

                    # backward + step when training
                    if is_train:
                        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                            print("Warning: NaN or Inf loss encountered; skipping backward for this batch.")
                        else:
                            scaler.scale(batch_loss).backward()
                            # optional gradient clipping 
                            max_norm = float(model_params.get("grad_clip", 0.0))
                            if max_norm > 0:
                                scaler.unscale_(optimizer)  # required before clip_grad_norm_
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                            scaler.step(optimizer)
                            scaler.update()

                # compute predictions / metrics on CPU to avoid GPU memory holding
                with torch.no_grad():
                    _, preds = torch.max(outputs, dim=1)
                    running_corrects += torch.sum(preds == labels).item()

                    # per-class stats
                    labels_cpu = labels.cpu().numpy()
                    preds_cpu = preds.cpu().numpy()
                    for cls in np.unique(labels_cpu):
                        cls = int(cls)
                        per_class_count.setdefault(cls, 0)
                        per_class_correct.setdefault(cls, 0)
                    for l, p in zip(labels_cpu.tolist(), preds_cpu.tolist()):
                        per_class_count[l] = per_class_count.get(l, 0) + 1
                        if p == l:
                            per_class_correct[l] = per_class_correct.get(l, 0) + 1

                # accumulate loss (use batch_loss.item only after gradients / scaler handled)
                running_loss += (batch_loss.detach().item() * labels.size(0))

                # free batch-level heavy tensors
                del inputs, labels, masks, outputs, aux, mask_output
                torch.cuda.empty_cache()
            # end epoch loop over batches

            # finalize epoch metrics
            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / (dataset_size if dataset_size > 0 else 1.0)
            epoch_acc = running_corrects / (dataset_size if dataset_size > 0 else 1.0)
            epoch_mask_dice = (running_mask_dice / mask_count) if mask_count > 0 else 0.0

            # recon / mimic averaged
            if recon_enabled:
                epoch_recon = running_recon_loss / (dataset_size if dataset_size > 0 else 1.0)
                epoch_mimic = running_mimic_loss / (dataset_size if dataset_size > 0 else 1.0)
            else:
                epoch_recon = None
                epoch_mimic = None

            print(f"{phase} Acc:{epoch_acc:.4f} Loss:{epoch_loss:.4f}")
            # per-class print in deterministic order
            class_keys = sorted(per_class_count.keys())
            class_accs = [ (per_class_correct.get(k,0) / per_class_count[k]) if per_class_count[k]>0 else 0.0 for k in class_keys ]
            print(f"{phase} per-class acc: {class_accs}")
            print(f"{phase} mask Dice: {epoch_mask_dice:.4f}")
            if epoch_recon is not None:
                print(f"{phase} Recon Loss: {epoch_recon:.4f}")
            if epoch_mimic is not None:
                print(f"{phase} Mimic Loss: {epoch_mimic:.4f}")

            # bookkeeping best
            if phase == "val":
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                if epoch_acc >= best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

            # attention hook printing
            if phase == "train" and (epoch + 1) % 10 == 0 and attention_hook is not None and attention_hook.features is not None:
                weights = attention_hook.features.mean(dim=0).detach().cpu().numpy()
                print(f"Epoch {epoch+1} Modality Weights: {np.round(weights, 4).tolist()}")

        # visualization of masks (val only)
        if ENABLE_MASK_VIZ and (epoch + 1) % VIZ_FREQUENCY == 0 and viz_inputs_cpu is not None and viz_masks_cpu is not None:
            visualize_masks_helper(model, viz_inputs_cpu, viz_masks_cpu, epoch, num_viz_samples, device)

        # cleanup
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"Training complete in {int(total_time//60)}m {int(total_time%60)}s; Best val acc: {best_val_acc:.4f}")

    # load best weights
    model.load_state_dict(best_model_wts)

    if attention_hook is not None:
        attention_hook.close()

    return model, train_acc_history, train_loss_history, val_acc_history, val_loss_history


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
