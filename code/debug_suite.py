import torch
from train import *


#wip

def run_debug_suite_single(model_module: LightningSingleModel, method, parameters):
    device = next(model_module.model.parameters()).device

    print("\n========== DEBUG SUITE ==========")

    # --- 1) Synthetic batch ---
    model_params = parameters[f"{method}_model_parameters"]

    B = 2
    C = parameters[f"{method}_channel_num"]
    H = model_params['input_size']
    W = model_params['input_size']

    x = torch.randn(B, C, H, W, device=device)
    mask = (torch.rand(B, 1, H, W, device=device) > 0.5).float()
    labels = torch.randint(0, model_module.class_num, (B,), device=device)

    batch = (x, mask, labels)
    print("Created synthetic batch.")

    # --- 2) Forward pass ---
    logits, aux, mask_pred = model_module.model(x, mask)
    print("Forward pass OK.")
    print("  logits:", logits.shape)
    print("  mask_pred:", None if mask_pred is None else mask_pred.shape)

    # --- 3) Shared step (train) ---
    loss = model_module._shared_step(batch, 0, "train")
    print("Shared step OK. Loss:", loss.item())

    # --- 4) MC Dropout ---
    mean_mc, std_mc = model_module.predict_mc_dropout(model_module.model, x)
    print("MC dropout OK. mean:", mean_mc.shape, "std:", std_mc.shape)

    # --- 5) TTA ---
    tta_pred = model_module.predict_tta(model_module.model, x)
    print("TTA OK. out:", tta_pred.shape)

    # --- 6) TTA + MC ---
    tta_mc_mean, tta_mc_std = model_module.predict_tta_mc(model_module.model, x, passes=3)
    print("TTA-MC OK. mean:", tta_mc_mean.shape)

    # --- 7) Predict custom ---
    pm = model_module.predict_custom(batch, mode="mc", mc_passes=3)
    print("predict_custom(mc) OK")

    pt = model_module.predict_custom(batch, mode="tta")
    print("predict_custom(tta) OK")

    ptm = model_module.predict_custom(batch, mode="tta_mc", mc_passes=3)
    print("predict_custom(tta_mc) OK")

    # --- 8) Debug attention hook ---
    if model_module.enable_modality_attention and model_module.attention_hook:
        logits2, aux2, _ = model_module.model(x, mask)
        feats = model_module.attention_hook.features
        if feats is not None:
            print("Attention hook OK. Features:", feats.shape)
        else:
            print("Attention hook did NOT capture features!")

    # --- 9) Check metric objects ---
    model_module.train_mask_dice.compute()
    model_module.val_mask_dice.compute()
    print("Metric objects operational.")

    print("========== END DEBUG SUITE ==========\n")
