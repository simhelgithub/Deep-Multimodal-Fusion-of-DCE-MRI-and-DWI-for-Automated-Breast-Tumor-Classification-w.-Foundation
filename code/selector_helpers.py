import torch
import torch.nn as nn
import torch.optim as optim

from loss import *
import pytorch_lightning as pl

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR



def get_classification_loss(parameters, train_labels, model_type, device):
    classification_loss_parameters = parameters[f"{model_type}_model_parameters"]['classification_loss_parameters']
    classification_loss_code = classification_loss_parameters["classification_loss_code"]

    if classification_loss_code == 'fl': #focal loss
      classification_alpha = classification_loss_parameters['alpha']  
      classification_gamma = classification_loss_parameters['gamma']
      if classification_alpha == None: classification_alpha = 0.25
      if classification_gamma == None: classification_gamma = 2
      return SoftFocalLoss(classification_alpha,classification_gamma)  

    elif classification_loss_code == 'wfl': #weighted focal loss
      classification_gamma = classification_loss_parameters['gamma']
      if classification_gamma == None: classification_gamma = 2
      #classification alpha is calculated
      
      # Calculate counts per class
      class_counts = torch.bincount(train_labels.long())
      total_samples = train_labels.size(0)
      num_classes = len(class_counts)
      
      # Calculate Inverse Class Frequency Weights
      class_weights = total_samples / (num_classes * (class_counts.float() + 1e-6))

      print(f"Train Class Counts: {class_counts.tolist()}")
      print(f"Calculated Class Weights: {class_weights.cpu().numpy()}")
      class_weights = class_weights.to(device)
      return SoftWeightedFocalLoss(classification_gamma, class_weights)
    else:
      raise ValueError(
          f"Invalid classification_loss_code '{classification_loss_code}'. "
          f"Valid options: ['cel', 'fl', 'wfl']"
      )


# Reconstruction loss selector, not used in updated flow

def get_recon_loss(parameters, model_type):
    
    reconstruction_loss_code = parameters[f"{model_type}_model_parameters"]["reconstruction_loss_code"]
    recon_enabled = parameters[f"{model_type}_model_parameters"]["recon_enabled"]

    if not recon_enabled:
        return None

    if reconstruction_loss_code == "mse":
        return nn.MSELoss()

    raise ValueError(
        f"Invalid {model_type} reconstruction_loss_code '{reconstruction_loss_code}'. Only 'mse' supported."
    )



# Classification optimizer selector / generator old 
def get_optimizer(model, parameters, model_type):
    optimizer_parameters =  parameters[f"{model_type}_model_parameters"]['optimizer_parameters']
    optimizer_type = optimizer_parameters['name'].lower()
    if optimizer_type == "adamw":
        return optim.AdamW(
          model.parameters(),
          lr=optimizer_parameters['lr'],
          betas=optimizer_parameters['betas'],
          eps=optimizer_parameters['eps'],
          weight_decay=optimizer_parameters['weight_decay'],
          amsgrad=optimizer_parameters['amsgrad']
        )
    elif optimizer_type == "adam":
        return optim.Adam(
          model.parameters(),
          lr=optimizer_parameters['lr'],
          betas=optimizer_parameters['betas'],
          eps=optimizer_parameters['eps'],
          weight_decay=optimizer_parameters['weight_decay'],
          amsgrad=optimizer_parameters['amsgrad']
        )
    else:
        raise ValueError(
            f"Invalid optimizer_type name '{optimizer_type}'. Valid options: ['adamW', 'adam']"
        )
# Mask loss classication criterion selector
def mask_criterion_selector(parameters, model_type): 
  mask_parameters = parameters[f"{model_type}_model_parameters"]['mask_parameters']
  mask_enabled = mask_parameters["mask"]
  mask_loss_type = mask_parameters["mask_loss_type"] 
  mask_criterion = None
  if mask_enabled:
    if mask_loss_type == "dice":
        mask_criterion = DiceLoss()
    elif mask_loss_type == "BCE":
        mask_criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Invalid mask loss: {mask_loss_type}")

  return mask_criterion


class LightningOptimizerFactory:
    """
    - Splits parameters into backbone vs non-backbone (by name prefix "backbone").
    - Builds backbone depth groups (for freezing/unfreezing).
    - Optionally builds discriminative LR groups **from non-backbone params only**.
    - Provides clear debug printing of named groups.
    """

    def __init__(self, model: torch.nn.Module, parameters: dict, model_type: str):
        self.model = model
        self.parameters = parameters
        self.model_type = model_type
        self.config = parameters[f"{model_type}_model_parameters"]

        # config knobs
        self.num_backbone_groups = parameters.get("backbone_num_groups", 3)
        self.backbone_freeze_on_start = parameters.get("backbone_freeze_on_start", True)

        # ----- collect named params -----
        # list of (name, param)
        named = list(self.model.named_parameters())

        # split backbone vs non-backbone by prefix "backbone"
        self.backbone_named = [(n, p) for (n, p) in named if n.startswith("backbone")]
        self.non_backbone_named = [(n, p) for (n, p) in named if not n.startswith("backbone")]

        # build backbone groups (named) by depth
        self.backbone_named_groups = self.group_params_by_depth_named(
            self.backbone_named, num_groups=self.num_backbone_groups
        )

        # also build non-backbone (head) depth groups (named)
        # helpful for discriminative LR if you want it on head
        self.non_backbone_named_groups = self.group_params_by_depth_named(
            self.non_backbone_named, num_groups=self.num_backbone_groups
        )

        # prepare simple param lists for optimizer usage (un-named)
        self.backbone_groups = [[p for (_, p) in g] for g in self.backbone_named_groups]
        self.non_backbone_groups = [[p for (_, p) in g] for g in self.non_backbone_named_groups]

        # debug print
        if self.parameters["backbone_debug"]:
          self.print_grouping_debug(self.backbone_named_groups, tag="BACKBONE GROUPS")
          self.print_grouping_debug(self.non_backbone_named_groups, tag="NON-BACKBONE (HEAD) GROUPS")

        # Freeze backbone initially if requested
        if self.backbone_freeze_on_start:
            self._backbone_freeze_all()

        # Build optimizer + scheduler factories
        self.optimizer_fn = self._build_optimizer()
        self.scheduler_fn = self._build_scheduler()

    # ---------------------
    # Helper: group named params by "depth"
    # ---------------------
    def group_params_by_depth_named(self, named_params, num_groups=3):
        """
        named_params: list of (name, param)
        Returns: list of groups, each group is list of (name, param)
        Grouping is done by counting '.' in the name (depth) and slicing sorted list.
        """
        if not named_params:
            return []

        # compute depth
        items = [(name.count("."), name, p) for (name, p) in named_params]
        items.sort(key=lambda x: x[0])  # shallow -> deep

        group_size = max(1, len(items) // num_groups)
        groups = []
        for i in range(num_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < num_groups - 1 else len(items)
            slice_items = items[start:end]
            groups.append([(name, p) for (_, name, p) in slice_items])
        return groups

    # ---------------------
    # Freeze/unfreeze helpers
    # ---------------------
    def _backbone_freeze_all(self):
        for group in self.backbone_groups:
            for p in group:
                p.requires_grad = False

    def _unfreeze_group(self, group_index):
        cnt = 0
        for p in self.backbone_groups[group_index]:
            if not p.requires_grad:
                p.requires_grad = True
                cnt += 1
        print(f"[DEBUG] Unfroze backbone group {group_index}: {cnt} params now trainable")
        

    def gradual_unfreeze(self, epoch, unfreeze_every_n_epochs=2):
        num_groups = len(self.backbone_groups)
        if num_groups == 0:
            return

        # which group index to unfreeze at this epoch?
        group_to_unfreeze = epoch // unfreeze_every_n_epochs

        # group indices: [0,1,2] shallow→deep  
        # but we want to unfreeze from deep→shallow
        group_to_unfreeze = num_groups - 1 - group_to_unfreeze

        if 0 <= group_to_unfreeze < num_groups:
            self._unfreeze_group(group_to_unfreeze)
    def _filter_trainable(self, group):
        return [p for p in group if p.requires_grad]

    # ---------------------
    # base optimizer builder
    # ---------------------
    def _get_base_optimizer(self, params, cfg):
        opt_name = cfg["optimizer_parameters"]["name"].lower()
        p = cfg["optimizer_parameters"]
        if opt_name == "adamw":
            return torch.optim.AdamW(params, lr=p["lr"], eps=p["eps"], betas=p["betas"],
                                     amsgrad=p.get("amsgrad", False), weight_decay=p["weight_decay"])
        elif opt_name == "adam":
            return torch.optim.Adam(params, lr=p["lr"], eps=p["eps"], betas=p["betas"],
                                     amsgrad=p.get("amsgrad", False), weight_decay=p["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    # ---------------------
    # Build optimizer (clean)
    # ---------------------
    def _build_optimizer(self):
        cfg = self.config
        p = cfg["optimizer_parameters"]
        use_discriminative_lr = p.get("discriminative_lr", False)

        # If no discriminative LR single param group: all trainable params
        if not use_discriminative_lr:
            # return factory: lightning expects a lambda or function taking params arg
            return lambda params: self._get_base_optimizer(params, cfg)

        # If discriminative LR: we build param groups from non-backbone (head) or from backbone
        num_groups = p.get("num_lr_groups", self.num_backbone_groups)
        lr_decay_factor = p.get("lr_decay_factor", 2.0)
        weight_decay = p.get("weight_decay", p.get("weight_decay", 0.0))
        base_lr = p.get("lr", 1e-3)

        use_discriminative_reg = p.get("discriminative_reg", False)
        reg_base = p.get("reg_base", weight_decay)        # default to normal WD if not set
        reg_decay_factor = p.get("reg_decay_factor", 2.0) # deeper groups get reg_base * factor^k

        # Choose where to apply discriminative LR. Use config flag 'discrim_on' if present.
        discrim_on = cfg.get("discrim_on", "all")  # options: "non_backbone" or "backbone" or "all"
        if discrim_on == "backbone":
            named_groups_for_discrim = self.backbone_named_groups
        elif discrim_on == "all":
            # combine backbone + non-backbone named lists into one sequence
            named_groups_for_discrim = self.group_params_by_depth_named(
                self.backbone_named + self.non_backbone_named, num_groups=num_groups
            )
        else:
            # default: non-backbone
            named_groups_for_discrim = self.non_backbone_named_groups

        # Build param groups (only include trainable params)
        param_groups = []
        # deepest group (last) should get highest LR; we'll iterate shallow->deep but compute LR accordingly
        groups = named_groups_for_discrim
        n_groups = max(1, len(groups))
        for i, named_group in enumerate(groups):
            params_list = [p for (n, p) in named_group if p.requires_grad]
            if not params_list:
                continue
            # -----------------------------
            # LR: deeper → larger LR
            # -----------------------------
            lr = base_lr / (lr_decay_factor ** (n_groups - 1 - i))
            # -----------------------------
            # Discriminative REG: deeper → larger WD
            # -----------------------------
            if use_discriminative_reg:
                # shallow group gets reg_base, deep gets reg_base * factor^(n_groups-1)
                wd = reg_base * (reg_decay_factor ** (n_groups - 1 - i))
            else:
                wd = weight_decay

            param_groups.append({
                "params": params_list,
                "lr": lr,
                "weight_decay": wd,
            })

        # As a safety: if param_groups is empty, fall back to all trainable params
        if not param_groups:
            all_trainable = [p for (n, p) in (self.backbone_named + self.non_backbone_named) if p.requires_grad]
            param_groups = [{"params": all_trainable, "lr": base_lr, "weight_decay": weight_decay}]

        # Debug print of final discriminative grouping
        self.print_param_group_summary(param_groups, tag="FINAL DISCRIM PARAM GROUPS")

        # Return factory that builds the optimizer from the param_groups
        return lambda _: self._get_base_optimizer(param_groups, cfg)

    # ---------------------
    # Build LR scheduler (unchanged semantics)
    # ---------------------
    def _build_scheduler(self):
        sch_cfg = self.config.get("scheduler", None)
        if sch_cfg is None:
            return None

        name = sch_cfg["name"].lower()

        if name == "reduce_lr_on_plateau":
            def _scheduler(optimizer):
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=sch_cfg["factor"],
                    patience=sch_cfg["patience"],
                    min_lr=sch_cfg["min_lr"],
                    threshold=sch_cfg["threshold"],
                )
                return {"scheduler": scheduler, "monitor": sch_cfg["monitor"], "interval": "epoch"}
            return _scheduler

        if name == "cosine":
            def _scheduler(optimizer):
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=sch_cfg["T_max"], eta_min=sch_cfg["eta_min"])
                return {"scheduler": scheduler, "interval": "epoch"}
            return _scheduler

        if name == "cosine_with_warmup":
            warmup_steps = sch_cfg.get("warmup_steps", 500)
            max_steps = sch_cfg.get("max_steps", 10000)
            def _scheduler(optimizer):
                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(warmup_steps)
                    progress = (step - warmup_steps) / float(max_steps - warmup_steps)
                    return 0.5 * (1 + torch.cos(torch.pi * progress))
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                return {"scheduler": scheduler, "interval": "step"}
            return _scheduler

        raise ValueError(f"Unknown scheduler: {name}")


    # ---------------------
    # Debug print helpers
    # ---------------------
    def print_grouping_debug(self, named_groups, tag=""):
        print("\n" + "="*80)
        print(f"[DEBUG] Parameter Grouping ({tag})")
        print("="*80)
        for i, group in enumerate(named_groups):
            print(f"\n--- Group {i} ({len(group)} params) ---")
            for name, p in group:
                print(f"  depth={name.count('.'):2d} | grad={p.requires_grad} | {name}")

    def print_param_group_summary(self, param_groups, tag=""):
        print("\n" + "-"*60)
        print(f"[DEBUG] {tag}")
        for i, g in enumerate(param_groups):
            cnt = len(g.get("params", []))
            lr = g.get("lr", None)
            wd = g.get("weight_decay", None)
            print(f"  ParamGroup {i}: count={cnt}  lr={lr}  wd={wd}")
        print("-"*60 + "\n")
