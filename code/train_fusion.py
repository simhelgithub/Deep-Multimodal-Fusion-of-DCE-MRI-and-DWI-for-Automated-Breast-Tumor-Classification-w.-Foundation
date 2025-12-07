import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAUROC, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
from loss import *
from selector_helpers import mask_criterion_selector
from train import * 


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

        # label smoothing
        label_smoothing_enabled = fusion_params["label_smoothing_enabled"]
        if label_smoothing_enabled:
            alpha = fusion_params["label_smoothing_alpha"]
            self.label_smoother = LabelSmoothing(self.class_num, alpha)
        else:
            self.label_smoother = None


        #regularization features
        self.attn_reg_enabled = fusion_params["attn_reg_enabled"]
        self.lambda_attn_sparsity = fusion_params["lambda_attn_sparsity"]
        self.lambda_attn_consistency = fusion_params["lambda_attn_consistency"]

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

    def configure_optimizers(self):
        # Let the factory handle all DWI/DCE/fusion params
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
            if self.current_epoch % self.unfreeze_timer == 0 and self.current_epoch != 0:
                self.opt_factory.gradual_unfreeze(
                    epoch=self.current_epoch,
                    unfreeze_every_n_epochs=self.unfreeze_timer
                )
                self._sync_optimizer_new_params()


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

    def _shared_step(self, batch, phase="train", return_preds = False):
        is_train = phase == "train"

        # unpack batch
        if len(batch) == 4:
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
        if self.parameters_dict['use_simple_aux_loss_scheduling']:
            aux_w = max(0.0, 1 - self.current_epoch / self.parameters_dict["aux_loss_weight_epoch_limit"])
        else:
            aux_w = 1.0

        # --- Forward encoders ---
        with torch.set_grad_enabled(is_train):
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
            if self.label_smoother is not None and is_train:
                smoothed = self.label_smoother(logits, labels)
                clf_loss = self.criterion_clf(logits, smoothed)
            else:
                clf_loss = self.criterion_clf(logits, labels)

            total_loss = clf_loss



            # Mask loss
            if self.mask_enabled and masks_batch is not None:
                mask_loss_val = (
                    safe_mask_loss(dwi_mask_pred, masks_batch, self.mask_criterion) +
                    safe_mask_loss(dce_mask_pred, masks_batch, self.mask_criterion) +
                    safe_mask_loss(fused_mask_logits, masks_batch, self.mask_criterion)
                )

                if is_train:
                    total_loss += self.lambda_mask * mask_loss_val
                    self.train_mask_loss.update(mask_loss_val)
                else:
                    self.val_mask_loss.update(mask_loss_val)
                    
            # ----------------------
            # Regularization (attention + feature norm)
            # ----------------------
            
            if self.attn_reg_enabled:
                attn_sparsity_loss = compute_attn_sparsity_loss(aux, self.lambda_attn_sparsity, self.device)
                attn_consistency_loss = compute_attn_consistency_loss(aux, self.lambda_attn_consistency, self.device)
                total_loss += attn_sparsity_loss * self.lambda_attn_sparsity + attn_consistency_loss * self.lambda_attn_consistency
  
            if self.feat_norm_reg_enabled:
                feat_norm_loss = compute_feat_norm_loss(aux, self.lambda_feat_norm, self.device)
                total_loss += feat_norm_loss * self.lambda_feat_norm


            # Reconstruction losses
            recon_loss_val = 0.0
            if aux_w > 0.0 and self.recon_enabled:
                recon_loss_val += compute_recon_list_loss(dwi_aux["recon_feats"], dwi_inputs)
                recon_loss_val += compute_recon_list_loss(dce_aux["recon_feats"], dce_inputs) 
                recon_loss_val += compute_recon_list_loss(aux["recon_fused"], torch.cat([dwi_inputs, dce_inputs], dim=1))
                
                if is_train:  
                  total_loss +=  self.lambda_recon * recon_loss_val * aux_w 
                  self.train_recon_loss.update(recon_loss_val.detach())
                else:
                  self.val_recon_loss.update(recon_loss_val.detach())



                # ---- Mimic Loss ----
                proj_pairs = aux["proj_fused"]
                if self.mimic_enabled and proj_pairs is not None and len(proj_pairs) >= 4:
                    p1, p1_r, p2, p2_r = proj_pairs[:4]
                    mimic_loss_val = mimic_feat_loss(p1, p1_r) + mimic_feat_loss(p2, p2_r)

                if is_train:
                    self.train_mimic_loss.update(mimic_loss_val.detach())
                    total_loss += self.lambda_mimic * mimic_loss_val * aux_w
                else:
                    self.val_mimic_loss.update(mimic_loss_val.detach())

                  

              
        # --- Metrics ---
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # -------------------
        # Update metrics
        # -------------------
        self.update_metrics(preds, logits, labels, phase=phase)

        # -------------------
        # Log aggregated metrics (MeanMetric objects)
        # -------------------
        if is_train:
            self.log(f"train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_mask_loss", self.train_mask_loss.compute(),  on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_recon_loss", self.train_recon_loss,  on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_mimic_loss", self.train_mimic_loss,  on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log(f"val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_mask_loss", self.val_mask_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_recon_loss", self.val_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_mimic_loss", self.val_mimic_loss, on_step=False, on_epoch=True, prog_bar=True)


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

        # Store first sample for debugging
        if self.parameters_dict.get('debug_val', False) and (self.enable_modality_attention or self.mask_enabled) and batch_idx == 0:
            self.latest_val_sample = {
                "dwi_input": batch[0][0].detach().cpu(),
                "dce_input": batch[1][0].detach().cpu(),
                "pred_mask": mask_output.detach().cpu() if self.mask_enabled else None,
                "gt_mask": batch[2][0].detach().cpu() if self.mask_enabled else None,
                #"mod_attn": aux["mod_attn_map"].detach().cpu() if self.enable_modality_attention else None,
            }

        return loss


    # -----
    # Validation epoch end
    # -----
    def on_validation_epoch_end(self):
        val_acc = float(self.trainer.callback_metrics["val_acc"])
        # Compute & log ROC AUC
        val_roc_auc = self.val_roc_auc.compute()
        self.log("val_roc_auc", val_roc_auc, prog_bar=True)
        self.val_roc_auc.reset()

        if self.latest_val_sample is None:
            return

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            best = self.latest_val_sample
            save_dir = self.paths[self.parameters_dict["save_dir"]]

            # ---- Save masks ----
            if self.mask_enabled:
                torch.save(best["pred_mask"], f"{save_dir}/best_pred_mask.pt")
                torch.save(best["dwi_input"], f"{save_dir}/best_dwi_input.pt")
                torch.save(best["dce_input"], f"{save_dir}/best_dce_input.pt")

                if best["gt_mask"] is not None:
                    torch.save(best["gt_mask"], f"{save_dir}/best_gt_mask.pt")

                if self.parameters_dict.get("debug_val", False):
                    visualize_single_mask_triplet(
                        input_img=torch.cat([best["dwi_input"], best["dce_input"]], dim=0),
                        gt_mask=best["gt_mask"],
                        pred_mask=best["pred_mask"],
                        title_prefix=f"Epoch {self.current_epoch}, best-so-far sample:",
                    )

            # ---- Save modality attention ----
            '''
            if self.enable_modality_attention:
                mod = best.get("mod_attn")
                if mod is not None:
                    mod_cpu = mod.to(torch.float32).cpu()
                    vec = mod_cpu[0].view(-1).tolist()
                    torch.save(vec, f"{save_dir}/best_modality_attention.pt")

                    if self.parameters_dict.get("debug_val", False):
                        self.print(f"Modality vector (sample 0): {vec}")
            '''

    # -----
    # Lightning test_step for fusion model
    # -----
    def test_step(self, batch, batch_idx):
        mode = self.parameters_dict.get("test_mode", "normal")
        mc_passes = self.parameters_dict.get("mc_passes", 10)

        # ---- prediction ----
        pred_result = self.predict_custom(batch, mode=mode, mc_passes=mc_passes)
        if isinstance(pred_result, torch.Tensor):
            outputs = pred_result
            variance = None
        elif isinstance(pred_result, tuple):
            outputs, variance = pred_result
            # log MC/TTA uncertainty
            self.log("test_uncertainty_mean", variance.mean(), prog_bar=False)
        else:
            raise RuntimeError("Unexpected predict_custom output.")

        labels = batch[-1].long()
        preds = outputs.argmax(dim=1)

        # ---- update test metrics ----
        self.test_acc.update((preds == labels).float().mean())
        self.test_auc.update(outputs, labels)
        self.test_f1.update(outputs, labels)
        self.test_precision.update(outputs, labels)
        self.test_recall.update(outputs, labels)
        self.test_confmat.update(preds, labels)

        # ---- log aggregated metrics ----
        self.log("test_acc", self.test_acc, prog_bar=True, on_epoch=True)
        self.log("test_auc", self.test_auc, prog_bar=True, on_epoch=True)
        self.log("test_f1", self.test_f1, prog_bar=True, on_epoch=True)
        self.log("test_precision", self.test_precision, prog_bar=True, on_epoch=True)
        self.log("test_recall", self.test_recall, prog_bar=True, on_epoch=True)

        return preds

    # -------------------
    # Helper: Update metrics
    # -------------------


    @torch._dynamo.disable
    def update_metrics(self, preds, logits, labels, phase="train"):
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
        elif phase == "val":
                  
            # Compute softmax probabilities for AUROC
            probs = torch.softmax(logits, dim=1)  # shape (B, C, H, W) or (B, C, D, H, W)
            self.val_f1.update(preds, labels)
            self.val_roc_auc.update(probs, labels)
            self.val_confmat.update(preds, labels)
        elif phase == "test":                  
            # Compute softmax probabilities for AUROC
            probs = torch.softmax(logits, dim=1)  # shape (B, C, H, W) or (B, C, D, H, W)
            self.test_f1.update(preds, labels)
            self.test_auc.update(probs, labels)
        else:
            raise ValueError(f"Unknown phase {phase}")


    # ----
    # Dropout (only during training)
    # ---
    def enable_dropout(self, model: torch.nn.Module):
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()


    # ---
    # MC dropout: helper to set batchnorm to eval
    # ---
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


    # MC dropout prediction for fusion model
    def predict_mc_dropout(self, dwi_inputs, dce_inputs, passes=20):
        orig_states_dwi = self._get_module_train_states(self.dwi_model)
        orig_states_dce = self._get_module_train_states(self.dce_model)

        self.mc_enable(self.dwi_model)
        self.mc_enable(self.dce_model)

        preds = []
        with torch.no_grad():
            for _ in range(passes):
                dwi_outputs, dwi_aux, dwi_mask_pred = self.dwi_model(dwi_inputs)
                dce_outputs, dce_aux, dce_mask_pred = self.dce_model(dce_inputs)
                logits, _, _ = self.forward(
                    dwi_aux["raw_feats"],
                    dce_aux["raw_feats"],
                    dwi_mask_pred,
                    dce_mask_pred
                )
                probs = torch.softmax(logits, dim=1)
                preds.append(probs)

        preds = torch.stack(preds, dim=0)

        self._restore_module_train_states(self.dwi_model, orig_states_dwi)
        self._restore_module_train_states(self.dce_model, orig_states_dce)

        return preds.mean(0), preds.std(0)


    # Test-time augmentation (TTA)
    def predict_tta(self, dwi_inputs, dce_inputs, masks=None, transforms=None):
        if transforms is None:
            transforms = self.transforms_list

        preds = []
        with torch.no_grad():
            for t in transforms:
                xt_dwi = t(x=dwi_inputs)
                xt_dce = t(x=dce_inputs)
                logits, _, _ = self.forward_from_inputs(xt_dwi, xt_dce, masks)
                probs = torch.softmax(logits, dim=1)
                preds.append(probs)

        preds = torch.stack(preds, dim=0)
        return preds.mean(0)


    # Both TTA + MC dropout
    def predict_tta_mc(self, dwi_inputs, dce_inputs, masks=None, transforms=None, passes=10):
        orig_states_dwi = self._get_module_train_states(self.dwi_model)
        orig_states_dce = self._get_module_train_states(self.dce_model)

        if transforms is None:
            transforms = self.transforms_list

        self.mc_enable(self.dwi_model)
        self.mc_enable(self.dce_model)

        preds = []
        with torch.no_grad():
            for t in transforms:
                xt_dwi = t(x=dwi_inputs)
                xt_dce = t(x=dce_inputs)
                for _ in range(passes):
                    logits, _, _ = self.forward_from_inputs(xt_dwi, xt_dce, masks)
                    probs = torch.softmax(logits, dim=1)
                    preds.append(probs)

        self._restore_module_train_states(self.dwi_model, orig_states_dwi)
        self._restore_module_train_states(self.dce_model, orig_states_dce)

        preds = torch.stack(preds, dim=0)
        return preds.mean(0), preds.std(0)

    def _sync_optimizer_new_params(self):
        if not hasattr(self, "trainer") or not getattr(self, "trainer", None):
            return
        if not self.trainer.optimizers:
            return

        opt = self.trainer.optimizers[0]
        existing_ids = {id(p) for g in opt.param_groups for p in g["params"]}

        # Collect all trainable params from DWI/DCE/fusion groups
        to_add = []
        for group in self.opt_factory.dwi_groups + self.opt_factory.dce_groups + [self.opt_factory.fusion_params]:
            for p in group:
                if p.requires_grad and id(p) not in existing_ids:
                    to_add.append(p)

        if not to_add:
            return

        # Learning rate scaling
        base_lr = self.parameters_dict.get("backbone_unfreeze_lr", 1e-4)
        factor = self.parameters_dict.get("backbone_unfreeze_lr_factor", 1.0)
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



    # ---
    # Helper to call the correct forward for TTA
    # ---
    def forward_from_inputs(self, dwi_inputs, dce_inputs, masks=None):
        # Encoder forward
        dwi_outputs, dwi_aux, dwi_mask_pred = self.dwi_model(dwi_inputs)
        dce_outputs, dce_aux, dce_mask_pred = self.dce_model(dce_inputs)
        # Fusion forward
        logits, fused_mask_logits, aux = self.forward(
            dwi_aux["raw_feats"], dce_aux["raw_feats"], dwi_mask_pred, dce_mask_pred
        )
        return logits, aux, fused_mask_logits


    # ---
    # Unified predict interface for fusion model
    # ---
    def predict_custom(self, batch, mode="normal", mc_passes=10):
        dwi_inputs = batch[0].to(self.device)
        dce_inputs = batch[1].to(self.device)
        labels = batch[-1].to(self.device)
        masks = batch[2] if len(batch) == 4 else None

        if mode == "normal":
            with torch.no_grad():
                logits, aux, mask_out = self.forward_from_inputs(dwi_inputs, dce_inputs, masks)
            return logits

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

def compute_recon_list_loss(recon_list, input_img):
    if recon_list is None:
        return torch.tensor(0.0, device=input_img.device)

    dim = 3 if input_img.dim() == 5 else 2
    mode = 'trilinear' if dim == 3 else 'bilinear'

    loss = torch.tensor(0.0, device=input_img.device)
    
    # If recon_list is a single tensor (not a list), wrap it
    if isinstance(recon_list, torch.Tensor):
        recon_list = [recon_list]

    for r in recon_list:
        if r is None:
            continue
        r_up = F.interpolate(r, size=input_img.shape[-dim:], mode=mode, align_corners=False)
        target = input_img
        if r_up.size(1) != target.size(1):
            target = target.mean(dim=1, keepdim=True)
        loss += F.smooth_l1_loss(r_up, target)
    
    return loss


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

    # Convert logits to probabilities and one-hot for dice loss (2 classes)
    mask_probs = torch.sigmoid(pred_logits)
    mask_pred_onehot = torch.cat([1 - mask_probs, mask_probs], dim=1)  # (N, 2, H, W)
    mask_target_onehot = torch.cat([1 - gt_resized, gt_resized], dim=1)

    return mask_criterion(mask_pred_onehot, mask_target_onehot)
