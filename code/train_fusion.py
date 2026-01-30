import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAUROC, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
from loss import *
from selector_helpers import mask_criterion_selector
from train import * 
import numpy as np


class LightningFusionModel(pl.LightningModule):
    def __init__(
        self,
        dwi_model: nn.Module,
        dce_model: nn.Module,
        fusion_model: nn.Module,
        parameters_dict: dict,
        criterion_clf,
        optimizer_fn,
        scheduler_fn=None,
        paths=None,
    ):
        super().__init__()
        self.method = "fusion"
        self.dwi_model = dwi_model
        self.dce_model = dce_model
        self.fusion_model = fusion_model
        self.parameters_dict = parameters_dict
        self.criterion_clf = criterion_clf
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.paths = paths

        # Fusion model parameters
        fusion_params = parameters_dict["fusion_model_parameters"]
        self.recon_enabled = fusion_params["recon_enabled"]
        self.lambda_recon = fusion_params["lambda_recon"]
        self.mask_enabled = fusion_params["mask_parameters"]["mask"]
        self.lambda_mask = fusion_params["mask_parameters"]["lambda_mask"]
        self.class_num = parameters_dict["class_num"]
        self.mimic_enabled = fusion_params["mimic_enabled"]
        self.lambda_mimic = fusion_params["lambda_mimic"]

        if fusion_params["label_smoothing_enabled"]:
            alpha = fusion_params["label_smoothing_alpha"]
            self.label_smoother = LabelSmoothing(self.class_num, alpha)
        else:
            self.label_smoother = None

        # Mask criterion
        self.mask_criterion = mask_criterion_selector(parameters_dict, self.method )

        # Metrics
        self.train_loss = MeanMetric()
        self.train_acc = MeanMetric()
        self.train_mask_loss   = MeanMetric()
        self.train_recon_loss  = MeanMetric()
        self.train_mimic_loss  = MeanMetric()
        self.train_f1 =  MulticlassF1Score(num_classes=parameters_dict["class_num"])

        self.val_loss = MeanMetric()
        self.val_acc = MeanMetric()
        self.val_roc_auc = MulticlassAUROC(num_classes=parameters_dict["class_num"])
        self.val_f1 = MulticlassF1Score(num_classes=parameters_dict["class_num"])
        self.val_precision = MulticlassPrecision(num_classes=parameters_dict["class_num"])
        self.val_recall = MulticlassRecall(num_classes=parameters_dict["class_num"])
        self.val_confmat = MulticlassConfusionMatrix(num_classes=parameters_dict["class_num"])
        self.val_mask_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.val_mimic_loss = MeanMetric()

        self.test_auc = MulticlassAUROC(num_classes=parameters_dict["class_num"])
        self.test_f1 = MulticlassF1Score(num_classes=parameters_dict["class_num"])
        self.test_precision = MulticlassPrecision(num_classes=parameters_dict["class_num"])
        self.test_recall = MulticlassRecall(num_classes=parameters_dict["class_num"])
        self.test_confmat = MulticlassConfusionMatrix(num_classes=parameters_dict["class_num"])
        self.test_acc = MeanMetric()


        self.enable_modality_attention = fusion_params["enable_modality_attention"]

        # aux loss weight scheduling, drops off towards epoch
        self.use_aux_loss_sched = parameters_dict['use_simple_aux_loss_scheduling'] 
        self.aux_loss_limit = parameters_dict["aux_loss_weight_epoch_limit"]
        # label smoothing
        label_smoothing_enabled = fusion_params["label_smoothing_enabled"]
        if label_smoothing_enabled:
            alpha = fusion_params["label_smoothing_alpha"]
            self.label_smoother = LabelSmoothing(self.class_num, alpha)
        else:
            self.label_smoother = None


        #regularization features
        self.attn_reg_enabled = fusion_params["attn_reg_enabled"]
        self.lambda_attn_energy = fusion_params["lambda_attn_energy"]
        self.lambda_feature_consistency = fusion_params["lambda_feature_consistency"]

        self.feat_norm_reg_enabled = fusion_params["feat_norm_reg_enabled"]
        self.lambda_feat_norm = fusion_params["lambda_feat_norm"]
        #unfreeze      
        self.unfreeze_timer = int(self.parameters_dict["unfreeze_timer"])
        self.backbone_freeze_on_start = self.parameters_dict['backbone_freeze_on_start'] 
        self.backbone_num_groups = self.parameters_dict['backbone_num_groups']
        self.layers_unfrozen = 0

        # book keeping for some non standard metrics
        self.best_val_acc = -1.0 #used to only save on best val acc
        self.latest_val_sample = None
        self.compile_enabled = parameters_dict['compile']

        #transform helpers for tta
        self.transforms_list = [ 
                tta_id,
                tta_flip_lr,
                tta_flip_ud,
                tta_flip_lrud]


        self.opt_factory = LightningFusionOptimizerFactory(
            dwi_model=self.dwi_model,
            dce_model=self.dce_model,
            fusion_model=self.fusion_model,
            parameters=self.parameters_dict
        )

        # Use its optimizer_fn and scheduler_fn
        self.optimizer_fn = self.opt_factory.optimizer_fn
        self.scheduler_fn = self.opt_factory.scheduler_fn
        self.to(self.device)

        # book keeping for some non standard metrics
        self.test_mod_attn = []
        self.test_modality_attention_mean = None
        self.test_preds = []
        self.test_targets = []
        self.test_preds_array = []
        self.test_targets_array = []
    def configure_optimizers(self):
        optimizer = self.optimizer_fn(None)  # factory ignores input; uses grouped params internally
        if self.scheduler_fn is None:
            return optimizer

        sched = self.scheduler_fn(optimizer)
        if isinstance(sched, dict):
            return {"optimizer": optimizer, "lr_scheduler": sched}
        return optimizer


    # -------------------------
    # epoch-level hooks & gradual unfreezing
    # -------------------------
    def on_train_epoch_start(self):
        # Reset metrics
        self.train_mask_loss.reset()
        self.train_recon_loss.reset()
        self.train_mimic_loss.reset()
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_loss.reset()
        # Gradual unfreeze
        if self.backbone_freeze_on_start and self.current_epoch <= (self.unfreeze_timer * self.backbone_num_groups + 1):
            
            new_params = self.opt_factory.gradual_unfreeze(epoch=self.current_epoch, unfreeze_every_n_epochs=self.unfreeze_timer)
            if new_params:
                opt = self.trainer.optimizers[0]
                self.opt_factory.sync_unfrozen_params_to_optimizer(opt, new_params)


    def on_validation_epoch_start(self):
        # Reset all validation metrics for this epoch
        try:
            self.val_mask_loss.reset()
            self.val_recon_loss.reset()
            self.val_mimic_loss.reset()
            self.val_precision.reset()
            self.val_recall.reset()
            self.val_confmat.reset()
            self.val_roc_auc.reset()
            self.val_f1.reset()
            self.val_loss.reset()
            self.val_acc.reset()
        except Exception:
            pass

    def on_test_epoch_start(self):
        # Reset all test metrics for this epoch
        try:
            self.test_auc.reset()
            self.test_f1.reset()
            self.test_precision.reset()
            self.test_recall.reset()
            self.test_confmat.reset()
            self.test_acc.reset()
        except Exception:
            pass

    def forward(self, dwi_feats, dce_feats, dwi_mask=None, dce_mask=None):
        return self.fusion_model(dwi_feats, dce_feats, dwi_mask, dce_mask)

    @torch._dynamo.disable
    def _shared_step(self, batch, phase="train", return_preds = False):
        is_train = phase == "train"

        # unpack batch
        if self.mask_enabled:
            dwi_inputs, dce_inputs, masks_batch, labels = batch
        else:
            dwi_inputs, dce_inputs, labels = batch
            masks_batch = None

        dwi_inputs = dwi_inputs.to(self.device)
        dce_inputs = dce_inputs.to(self.device)
        labels = labels.long().to(self.device)
        if masks_batch is not None:
            masks_batch = masks_batch.to(self.device)

        # aux loss weight scheduling, drops off towards epoch
        if self.use_aux_loss_sched:
            aux_w = max(0.0, 1 - self.current_epoch / self.aux_loss_limit)
        else:
            aux_w = 1.0

        # --- Forward encoders ---
        dwi_outputs, dwi_aux, dwi_mask_pred = self.dwi_model(dwi_inputs)
        dce_outputs, dce_aux, dce_mask_pred = self.dce_model(dce_inputs)

        # --- Fusion forward ---
        logits, fused_mask_logits, aux = self.forward(
            dwi_aux["raw_feats"],
            dce_aux["raw_feats"],
            dwi_mask_pred,
            dce_mask_pred,
        )
    
        # Classification loss
        if self.label_smoother:
            smoothed = self.label_smoother(logits, labels)
        cls_loss = self.criterion_clf(logits, smoothed) if is_train else self.criterion_clf(logits, labels)
        total_loss = cls_loss

        # Mask loss
        mask_loss_val = 0.0
        if self.mask_enabled: #and masks_batch is not None:
            num_masks = 3
            mask_loss_val = (
                safe_mask_loss(dwi_mask_pred, masks_batch, self.mask_criterion) +
                safe_mask_loss(dce_mask_pred, masks_batch, self.mask_criterion) +
                safe_mask_loss(fused_mask_logits, masks_batch, self.mask_criterion)
            ) / num_masks

            total_loss += self.lambda_mask * mask_loss_val if is_train else 0.0

        # ----------------------
        # Regularization  
        # ----------------------
        
        if self.attn_reg_enabled:
            attn_energy_loss = compute_attn_energy_loss(aux, self.device)
            feature_consistency_loss = compute_feature_consistency_loss(aux, self.device)
            total_loss += attn_energy_loss * self.lambda_attn_energy + feature_consistency_loss * self.lambda_feature_consistency if is_train else 0.0

        if self.feat_norm_reg_enabled:
            feat_norm_loss = compute_feat_norm_loss(aux, self.device)
            total_loss += feat_norm_loss * self.lambda_feat_norm if is_train else 0.0
  

        # Reconstruction losses
        recon_loss_val = torch.tensor(0.0, device=self.device)        
        mimic_loss_val = torch.tensor(0.0, device=self.device)        

        if aux_w > 0.0 and self.recon_enabled and is_train:
        # ---- Reconstruction ----
            num_recon_terms = 3
            dwi_input_det = dwi_inputs.detach()
            dce_input_det = dce_inputs.detach()
            fused_input_det = torch.cat([dwi_input_det, dce_input_det], dim=1)

            recon_loss_val = (
                compute_recon_list_loss(dwi_aux["recon_feats"], dwi_input_det) +
                compute_recon_list_loss(dce_aux["recon_feats"], dce_input_det) +
                compute_recon_list_loss(aux["recon_fused"], fused_input_det)
            ) / num_recon_terms

            total_loss += self.lambda_recon * recon_loss_val * aux_w
            self.train_recon_loss.update(recon_loss_val.detach())

            # ---- Mimic ----
            proj_pairs = aux.get("proj_fused", None)
            if self.mimic_enabled and proj_pairs is not None and len(proj_pairs) >= 4:
                p1, p1_r, p2, p2_r = proj_pairs[:4]
                mimic_loss_val = (mimic_feat_loss(p1, p1_r) + mimic_feat_loss(p2, p2_r)) / 2
                total_loss += self.lambda_mimic * mimic_loss_val * aux_w
                self.train_mimic_loss.update(mimic_loss_val.detach())

              
        # --- Metrics ---
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # test step handles its metrics itself
        if phase in['train','val']:
            # -------------------
            # Update metrics
            # -------------------
            update_metrics(self,preds, logits, labels, mask_loss_val, recon_loss_val, mimic_loss_val, phase=phase)

            # -------------------
            # Log aggregated metrics 
            # -------------------
            log_losses(self, total_loss, acc, phase)
        # -------------------
        # Optional: return detached outputs
        # -------------------
        if return_preds:
            return total_loss.detach(), logits.detach(), aux, fused_mask_logits


        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")


    # ------
    # Lightning validation_step for fusion model
    # -------
    
    def validation_step(self, batch, batch_idx):
        # Fusion _shared_step expects batch = (dwi_inputs, dce_inputs, masks?, labels)
        loss, outputs, aux, mask_output = self._shared_step(batch, phase="val", return_preds=True)

        return loss


    # -----
    # Lightning test_step for fusion model (updated)
    # -----
    @torch._dynamo.disable
    def test_step(self, batch, batch_idx):
        mode = self.parameters_dict.get("test_mode", "normal")
        mc_passes = self.parameters_dict.get("mc_passes", 10)

        # ---- prediction ----
        if mode == "normal":
          logits, mask_pred, aux = self.predict_custom(batch=batch, mode=mode, mc_passes=mc_passes)
        else:
          logits, variance, aux = self.predict_custom(batch=batch, mode=mode, mc_passes=mc_passes)


        # ---- log MC/TTA uncertainty ----
        if mode in ["mc", "tta", "tta_mc"] and variance is not None:
            self.log("test_uncertainty_mean", variance.mean(), prog_bar=False)

        labels = batch[-1].long()

        # ---- single point of truth ----
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        # ---- store raw probs for ROC/analysis ----
        if not hasattr(self, "test_preds"):
            self.test_preds = []
            self.test_targets = []
        self.test_preds.append(probs.detach().cpu())
        self.test_targets.append(labels.detach().cpu())
        
        # --- modality attention: take mean over batch ---
        if self.enable_modality_attention:
            mod = aux.get("gating_weights", None)
            if mod is not None:
                # mean over batch
                mean_mod = mod.detach().cpu().mean(dim=0)  # shape: [num_modalities]
                self.test_mod_attn.append(mean_mod.float())  # append to list
        # ---- save modality attention ----

        # ---- update metrics ----
        batch_acc = (preds == labels).float().mean()
        self.test_acc.update(batch_acc)
        self.test_auc.update(probs, labels)
        self.test_f1.update(preds, labels)
        self.test_precision.update(preds, labels)
        self.test_recall.update(preds, labels)
        self.test_confmat.update(preds, labels)

        return preds


    def on_test_epoch_end(self):
        # ---- compute + log main metrics ----
        self.log("test_acc", self.test_acc.compute())
        self.log("test_auc", self.test_auc.compute())
        self.log("test_f1", self.test_f1.compute())
        self.log("test_precision", self.test_precision.compute())
        self.log("test_recall", self.test_recall.compute())

        # ---- compute confusion matrix + per-class accuracy ----
        confmat = self.test_confmat.compute()
        per_class_acc = confmat.diag() / confmat.sum(1).clamp(min=1)

        for i, acc in enumerate(per_class_acc):
            self.log(f"test_acc_class_{i}", acc, prog_bar=True)
        self.test_acc_per_class = per_class_acc.cpu().numpy()

        # ---- reset metrics ----
        self.test_acc.reset()
        self.test_auc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_confmat.reset()

        # ---- concatenate stored preds + labels ----
        preds = torch.cat(self.test_preds, dim=0).numpy()
        targets = torch.cat(self.test_targets, dim=0).numpy()

        if not hasattr(self, "test_preds_array"):
            self.test_preds_array = []
            self.test_targets_array = []
        self.test_preds_array.append(preds)
        self.test_targets_array.append(targets)

        # --- modality attention ---
        if self.enable_modality_attention and len(self.test_mod_attn) > 0:
            # stack list of tensors -> [num_batches, num_modalities]
            self.test_mod_attn = torch.stack(self.test_mod_attn, dim=0)
            # compute mean over batches
            self.test_modality_attention_mean = self.test_mod_attn.mean(dim=0).numpy()
        else:
            # fallback: empty tensor / zeros
            self.test_mod_attn = torch.empty((0, 0), dtype=torch.float32)
            self.test_modality_attention_mean = np.array([])

        # ---- clean up ----
        self.test_preds.clear()
        self.test_targets.clear()
      
    
    # ----
    # Dropout (only during training)
    # ---
    @torch._dynamo.disable
    def enable_dropout(self, model: torch.nn.Module):
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
                

    # ---
    # MC dropout: helper to set batchnorm to eval
    # ---
    @torch._dynamo.disable
    def set_batchnorm_eval(self, model: torch.nn.Module):
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

    # ---- MC dropout prediction for fusion model ----
    @torch._dynamo.disable
    def predict_mc_dropout(self, dwi_inputs, dce_inputs, masks=None, passes=20):
        orig_states_dwi = self._get_module_train_states(self.dwi_model)
        orig_states_dce = self._get_module_train_states(self.dce_model)

        self.mc_enable(self.dwi_model)
        self.mc_enable(self.dce_model)

        preds_list = []
        gating_list = []

        with torch.no_grad():
            for _ in range(passes):
                dwi_out, dwi_aux, dwi_mask = self.dwi_model(dwi_inputs)
                dce_out, dce_aux, dce_mask = self.dce_model(dce_inputs)

                logits, _, aux = (
                    self.forward_from_inputs(dwi_out, dce_out, masks)
                    if masks is not None
                    else self.forward(dwi_aux["raw_feats"], dce_aux["raw_feats"], dwi_mask, dce_mask)
                )
                gw = aux["gating_weights"]
                
                if gw is not None:
                    # collapse spatial dims only if they exist
                    # If gw has spatial dims
                    if gw.dim() == 5:        # 3D feature map
                        gw = gw.mean(dim=(2,3,4))
                    elif gw.dim() == 4:      # 2D feature map
                        gw = gw.mean(dim=(2,3))
                    # If gw is (B, C)
                    elif gw.dim() == 2:
                        pass  # already collapsed
                    else:
                        raise ValueError(f"Unexpected gating weight shape: {gw.shape}")
                    gw = gw.cpu()
                    gating_list.append(gw)

                preds_list.append(torch.softmax(logits, dim=1))

        # ---- aggregate predictions and gating ----
        preds_stack = torch.stack(preds_list, dim=0)
        mean_preds = preds_stack.mean(0)
        std_preds = preds_stack.std(0)
        mean_gating = torch.stack(gating_list, dim=0).mean(0) if gating_list else None

        self._restore_module_train_states(self.dwi_model, orig_states_dwi)
        self._restore_module_train_states(self.dce_model, orig_states_dce)

        aux_out = {
            "gating_weights": mean_gating,
            "dwi_aux": dwi_aux,
            "dce_aux": dce_aux
        }
        return mean_preds, std_preds, aux_out


    # ---- Test-time augmentation (TTA) ----
    @torch._dynamo.disable
    def predict_tta(self, dwi_inputs, dce_inputs, masks=None, transforms=None):
        if transforms is None:
            transforms = self.transforms_list

        preds_list = []
        gating_list = []
        last_dwi_aux = None
        last_dce_aux = None

        with torch.no_grad():
            for t in transforms:
                xt_dwi = t(x=dwi_inputs)
                xt_dce = t(x=dce_inputs)

                logits, _, aux = self.forward_from_inputs(xt_dwi, xt_dce, masks)
                preds_list.append(torch.softmax(logits, dim=1))

                gw = aux["gating_weights"]
                # If gw has spatial dims
                if gw.dim() == 5:        # 3D feature map
                    gw = gw.mean(dim=(2,3,4))
                elif gw.dim() == 4:      # 2D feature map
                    gw = gw.mean(dim=(2,3))
                # If gw is (B, C)
                elif gw.dim() == 2:
                    pass  # already collapsed
                else:
                    raise ValueError(f"Unexpected gating weight shape: {gw.shape}")
                gating_list.append(gw.cpu())

                last_dwi_aux = aux.get("dwi_aux")
                last_dce_aux = aux.get("dce_aux")

        mean_preds = torch.stack(preds_list, dim=0).mean(0)
        std_preds = torch.stack(preds_list, dim=0).std(0)
        mean_gating = torch.stack(gating_list, dim=0).mean(0) if gating_list else None

        aux_out = {
            "gating_weights": mean_gating,
            "dwi_aux": last_dwi_aux,
            "dce_aux": last_dce_aux
        }
        return mean_preds, std_preds, aux_out


    # ---- TTA + MC dropout ----
    @torch._dynamo.disable
    def predict_tta_mc(self, dwi_inputs, dce_inputs, masks=None, transforms=None, passes=10):
        if transforms is None:
            transforms = self.transforms_list

        all_preds = []
        all_gating = []
        last_dwi_aux = None
        last_dce_aux = None

        for t in transforms:
            xt_dwi = t(x=dwi_inputs)
            xt_dce = t(x=dce_inputs)

            mean_preds, std_preds, aux = self.predict_mc_dropout(xt_dwi, xt_dce, masks=masks, passes=passes)
            all_preds.append(mean_preds)

            gw = aux["gating_weights"]

            # If gw has spatial dims
            if gw.dim() == 5:        # 3D feature map
                gw = gw.mean(dim=(2,3,4))
            elif gw.dim() == 4:      # 2D feature map
                gw = gw.mean(dim=(2,3))
            # If gw is (B, C)
            elif gw.dim() == 2:
                pass  # already collapsed
            else:
                raise ValueError(f"Unexpected gating weight shape: {gw.shape}")
            all_gating.append(gw)

            last_dwi_aux = aux.get("dwi_aux")
            last_dce_aux = aux.get("dce_aux")

        mean_preds = torch.stack(all_preds, dim=0).mean(0)
        std_preds = torch.stack(all_preds, dim=0).std(0)
        mean_gating = torch.stack(all_gating, dim=0).mean(0) if all_gating else None

        aux_out = {
            "gating_weights": mean_gating,
            "dwi_aux": last_dwi_aux,
            "dce_aux": last_dce_aux
        }
        return mean_preds, std_preds, aux_out

    # ---
    # measure grad norm, per group, pre clip
    # ---
    def on_after_backward(self):

        if self.global_step % 100 == 0:
            # -------------------------------------
            # Compute per-group grad norms
            # -------------------------------------
            param_groups = {
                "dwi": list(self.dwi_model.parameters()),
                "dce": list(self.dce_model.parameters()),
                "fusion": list(self.fusion_model.parameters()),
            }
            for name, params in param_groups.items():
                grads = [p.grad.data.norm(2) for p in params if p.grad is not None]
                if grads:
                    group_norm = torch.norm(torch.stack(grads), 2)
                else:
                    group_norm = torch.tensor(0.0)
                self.log(f"grad_norm_{name}", group_norm, prog_bar=True)

            # -------------------------------------
            # Total grad norm
            # -------------------------------------
            all_params = [p for p in self.parameters() if p.grad is not None]
            if all_params:
                total_norm = torch.norm(torch.stack([p.grad.data.norm(2) for p in all_params]), 2)
            else:
                total_norm = torch.tensor(0.0)

            self.log("grad_norm_total", total_norm, prog_bar=True)

    # ---
    # Helper to call the correct forward for TTA
    # ---
    def forward_from_inputs(self, dwi_inputs, dce_inputs, masks=None):
        # Encoder forward
        dwi_outputs, dwi_aux, dwi_mask_pred = self.dwi_model(dwi_inputs)
        dce_outputs, dce_aux, dce_mask_pred = self.dce_model(dce_inputs)
        # Fusion forward
        return self.forward(
            dwi_aux["raw_feats"], dce_aux["raw_feats"], dwi_mask_pred, dce_mask_pred
        )

    # ---
    # Unified predict interface for fusion model
    # ---
    def predict_custom(self, batch, mode="normal", mc_passes=10):
        dwi_inputs = batch[0].to(self.device)
        dce_inputs = batch[1].to(self.device)
        labels = batch[-1].to(self.device)
        masks = batch[2] if len(batch) == 4 else None

        if mode == "normal":
            return self.forward_from_inputs(dwi_inputs, dce_inputs, masks)
            
        elif mode == "tta":
            return self.predict_tta(dwi_inputs, dce_inputs, masks)

        elif mode == "mc":
            return self.predict_mc_dropout(dwi_inputs, dce_inputs, passes=mc_passes)

        elif mode == "tta_mc":
            return self.predict_tta_mc(dwi_inputs, dce_inputs, masks, passes=mc_passes)

        else:
            raise ValueError(f"Unknown predict mode: {mode}")

   
# -----
# Fusion unique helpers
# ----
# helper that pairs recon list and target scales
# Multi-scale / multi-source reconstruction loss
def compute_recon_list_loss(recon_list, input_img):
    """
    Computes reconstruction loss over a list of reconstructions.
    Supports multiple scales, handles channel mismatch, and normalizes by number of valid reconstructions.
    Uses Charbonnier loss.
    """
    if recon_list is None:
        return torch.tensor(0.0, device=input_img.device, dtype=input_img.dtype)

    dim = 3 if input_img.dim() == 5 else 2
    mode = 'trilinear' if dim == 3 else 'bilinear'

    if isinstance(recon_list, torch.Tensor):
        recon_list = [recon_list]

    valid_recons = [r for r in recon_list if r is not None]
    if len(valid_recons) == 0:
        return torch.tensor(0.0, device=input_img.device, dtype=input_img.dtype)

    total_loss = torch.zeros((), device=input_img.device, dtype=input_img.dtype)

    for r in valid_recons:
        # Upsample to target resolution
        r_up = F.interpolate(r, size=input_img.shape[-dim:], mode=mode, align_corners=False)
        
        # Handle channel mismatch
        if r_up.size(1) != input_img.size(1):
            r_up = r_up.mean(dim=1, keepdim=True)
            target = input_img.mean(dim=1, keepdim=True)
        else:
            target = input_img

        total_loss += recon_image_loss(r_up, target)

    # Normalize by number of valid reconstructions (scales)
    return total_loss / len(valid_recons)

    
@torch._dynamo.disable
def safe_mask_loss(pred_logits, gt_mask, mask_criterion):
    if pred_logits is None:
        raise ValueError("pred_logits is None in safe_mask_loss")
    if gt_mask is None:
        raise ValueError("gt_mask is None in safe_mask_loss")
    
    if pred_logits.shape[-2:] != gt_mask.shape[-2:]:
        print("mask resized warning safe_mask_loss", pred_logits.shape[-2:], gt_mask.shape[-2:])
        gt_resized = F.interpolate(gt_mask, size=pred_logits.shape[-2:], mode='nearest')
    else:
        gt_resized = gt_mask

    return mask_criterion(pred_logits, gt_mask)
