import torch
import torch.nn.functional as F
from train import LightningSingleModel



def run_debug_suite_single(model_module, method, parameters, device):
    """
    -runs the usual forward / shared step / MC / TTA checks
    -and also computes the three regularizers
    """
    print("\n========== DEBUG SUITE (device={}) ==========".format(device))

    # ----- synthetic batch -----
    params = parameters[f"{method}_model_parameters"]
    B = 2  # small batch
    C = parameters[f"{method}_channel_num"]
    H = params['input_size']
    W = params['input_size']

    x = torch.randn(B, C, H, W, device=device)
    masks = (torch.rand(B, 1, 32, 32, device=device) > 0.5).float()
    labels = torch.randint(0, model_module.class_num, (B,), device=device)
    batch = (x, masks, labels)

    print("Created synthetic batch.")
    print("  x device:", x.device, " mask device:", masks.device, " labels device:", labels.device)

    # ----- forward -----
    out, aux, mask_pred = model_module(x, masks)
    print("Forward pass OK.")
    print("  logits:", out.shape, " mask_pred:", None if mask_pred is None else mask_pred.shape)
    print(f"[DEBUG] Input Stats: Min={x.min():.4f}, Max={x.max():.4f}, Mean={x.mean():.4f}, Std={x.std():.4f}")
    print(f"[DEBUG] Mask Stats: Min={masks.min():.4f}, Max={masks.max():.4f}, Mean={masks.mean():.4f}")

    # ----- shared step -----
    loss = model_module._shared_step(batch, 0, "train")
    print("Shared step OK. Loss:", float(loss.item()))

    # ----- compute/inspect regularizers (robust) -----
    print("\n--- Regularizers debug ---")

    # Attention sparsity: prioritize explicit modality attention (aux['mod_attn']),
    #    then mask attention map (aux['mask_attn_map']) if present, else skip.
    attn_sparsity_val = None
    if aux is not None:
        if "mod_attn" in aux and aux["mod_attn"] is not None:
            mod_attn = aux["mod_attn"]
            # compute mean absolute activation as sparsity estimate
            attn_sparsity_val = mod_attn.abs().mean().item()
            print(f"Attention sparsity (mod_attn) = {attn_sparsity_val:.6g}  shape={tuple(mod_attn.shape)}")
        elif "mask_attn_map" in aux and aux["mask_attn_map"] is not None:
            mam = aux["mask_attn_map"]
            attn_sparsity_val = mam.abs().mean().item()
            print(f"Attention sparsity (mask_attn_map) = {attn_sparsity_val:.6g}  shape={tuple(mam.shape)}")
        else:
            print("Attention sparsity: no modality attention or mask-attn-map present -> skipped")
    else:
        print("Attention sparsity: aux is None -> skipped")

    # Quick sanity thresholds for sparsity
    if attn_sparsity_val is not None:
        if attn_sparsity_val > 1.0:
            print("! Attention sparsity unusually large (>1.0) — may dominate loss.")
        elif attn_sparsity_val < 1e-5:
            print("! Attention sparsity essentially zero (<1e-5) — attention may be collapsed/suppressed.")

    #Attention consistency: compare shallow (f1) vs deeper (f2) features in a robust pooled way
    attn_consistency_val = None
    if aux is not None and "raw_feats" in aux and aux["raw_feats"] is not None:
        raw = aux["raw_feats"]
        try:
            f1, f2, f3 = raw[:3]  # expect list-like
            # pool spatial dims to get per-channel vectors
            f1_vec = F.adaptive_avg_pool2d(f1, (1, 1)).view(f1.size(0), -1)  # [B, C1]
            f2_vec = F.adaptive_avg_pool2d(f2, (1, 1)).view(f2.size(0), -1)  # [B, C2]

            # make sizes compatible by slicing to min channels (safe, deterministic)
            minC = min(f1_vec.size(1), f2_vec.size(1))
            f1_slice = f1_vec[:, :minC]
            f2_slice = f2_vec[:, :minC]

            # normalize per-sample
            f1n = f1_slice / (f1_slice.norm(dim=1, keepdim=True) + 1e-6)
            f2n = f2_slice / (f2_slice.norm(dim=1, keepdim=True) + 1e-6)

            attn_consistency_val = F.mse_loss(f1n, f2n).item()
            print(f"Attention consistency (pooled MSE f1 vs f2) = {attn_consistency_val:.6g}  (pooled dims: {minC})")
        except Exception as e:
            print("Attention consistency: failed to compute (bad shapes) ->", e)
    else:
        print("Attention consistency: raw_feats not present in aux -> skipped")

    if attn_consistency_val is not None:
        if attn_consistency_val > 1.0:
            print("! Attention consistency loss large (>1.0) — features may be very different or exploding.")
        elif attn_consistency_val < 1e-6:
            print("! Attention consistency nearly zero (<1e-6) — features may be identical/collapsed.")

    # 3) Feature-norm regularization: mean L2 / norm of each stage
    feat_norm_val = None
    if aux is not None and "raw_feats" in aux and aux["raw_feats"] is not None:
        raw = aux["raw_feats"]
        try:
            vals = []
            for i, f in enumerate(raw):
                if f is None:
                    continue
                # compute per-feature average norm (per-sample)
                # use: mean of L2 norms across channels/spatial -> scalar per-batch
                per_sample_norm = f.pow(2).sum(dim=1).sqrt().view(f.size(0), -1).mean(dim=1)  # [B]
                vals.append(per_sample_norm.mean().item())
            if vals:
                feat_norm_val = float(sum(vals) / len(vals))
                print(f"Feature norm (mean L2 across stages) = {feat_norm_val:.6g}  per-stage={vals}")
            else:
                print("Feature norm: no raw_feats entries -> skipped")
        except Exception as e:
            print("Feature norm: failed to compute ->", e)
    else:
        print("Feature norm: raw_feats not present -> skipped")

    if feat_norm_val is not None:
        if feat_norm_val > 1e3:
            print("! Feature norms very large (>1e3) — may lead to instability.")
        elif feat_norm_val < 1e-6:
            print("! Feature norms essentially zero (<1e-6) — possible collapse.")

    # For further testing return dict 
    reg_summary = {
        "attn_sparsity": attn_sparsity_val,
        "attn_consistency": attn_consistency_val,
        "feat_norm": feat_norm_val,
    }


    # ===================================================================
    # MC DROPOUT — VALIDATE VARIATION
    # ===================================================================
    mc_mean, mc_std = model_module.predict_mc_dropout(model_module.model, x, passes=6)
    print("\nMC dropout OK. mean shape:", mc_mean.shape, " std shape:", mc_std.shape)

    # Compute a small expected threshold based on batch size and number of classes
    B, num_classes = x.size(0), mc_mean.size(1)
    expected_var_threshold = 1e-5 * B * num_classes  # scale with batch size and classes

    mc_std_mean = float(mc_std.mean())
    if mc_std_mean < expected_var_threshold:
        print(f"! WARNING: MC dropout variance extremely small ({mc_std_mean:.6g}) — dropout may NOT be active!")
    else:
        print(f"+ MC variance looks reasonable: {mc_std_mean:.6g}")

    print("MC mean var:", mc_std_mean)
    # ===================================================================
    # TTA — CHECK TRANSFORM EFFECT
    # ===================================================================
    base_pred = torch.softmax(out, dim=1)
    tta_pred, _  = model_module.predict_tta(model_module.model, x, masks, return_aux=False)

    diff = (tta_pred - base_pred).abs().mean()
    print("\nTTA OK. out shape:", tta_pred.shape)

    if diff < 1e-6:
        print("! WARNING: TTA had almost no effect — transforms may not be applied!")
    else:
        print("+ TTA modifies predictions. Mean diff:", float(diff.detach()))

    # ===================================================================
    # TTA-MC — CHECK COMBINED VARIATION
    # ===================================================================
    tta_mc_mean, tta_mc_std = model_module.predict_tta_mc(model_module.model, x, masks, passes=4)
    print("\nTTA-MC OK. mean shape:", tta_mc_mean.shape)

    if tta_mc_std.mean() < mc_std.mean():
        print("! WARNING: TTA-MC std < MC std — unexpected (may indicate broken TTA loop)!")
    else:
        print("+ TTA-MC variance > MC variance as expected.")

    # ===================================================================
    # predict_custom CROSS-CHECK
    # ===================================================================
    pm,_ = model_module.predict_custom(batch, mode="mc", mc_passes=3)
    pt,_ = model_module.predict_custom(batch=batch, mode="tta")
    ptm, _ = model_module.predict_custom(batch, mode="tta_mc", mc_passes=3)

    print("\npredict_custom(mc) OK")
    print("predict_custom(tta) OK")
    print("predict_custom(tta_mc) OK")


    # Check consistency
    diff = (pt - tta_pred).abs().mean().item()

    if diff < 1e-3:
        print(f"+ predict_custom(tta) matches predict_tta()   diff={diff:.4g}")
    elif diff < 1e-2:
        print(f"(!) Slight mismatch (acceptable): diff={diff:.4g}")
    else:
        print(f"X Large mismatch: diff={diff:.4g}")



    # ===================================================================
    # Mask loss metric check
    # ===================================================================
    try:
        model_module.train_mask_loss.compute()
        model_module.val_mask_loss.compute()
        print("\nMetric objects operational (compute() invoked).")
    except Exception as e:
        print("\n! Metric computation failed:", e)

    print("========== END DEBUG SUITE ==========\n")

    return reg_summary

#todo implement
def run_debug_suite_fusion(model_module, method, parameters, device):
  pass