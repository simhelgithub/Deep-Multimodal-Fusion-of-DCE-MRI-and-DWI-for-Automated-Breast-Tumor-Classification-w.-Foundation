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


# Reconstruction loss selector
  # only mse loss supported but other loss type actually used, fix
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

    def __init__(self, model, parameters, model_type):
        self.model = model
        self.parameters = parameters
        self.model_type = model_type
        self.config = parameters[f"{model_type}_model_parameters"]

        # ---- Backbone depth groups (with names maintained)
        self.num_backbone_groups = parameters["backbone_num_groups"]
        self.backbone_named_groups = self.group_params_by_depth(
            self.model,
            num_groups=self.num_backbone_groups,
            filter_by_prefix="backbone",
            return_names=True,
        )

        # Extract only the params for freezing/LR groups
        self.backbone_groups = [
            [p for (_, p) in group] for group in self.backbone_named_groups
        ]

        # Freeze everything initially if desired
        if parameters["backbone_freeze_on_start"]:
            self._backbone_freeze_all()

        # Build optimizer + scheduler factories
        self.optimizer_fn = self._build_optimizer()
        self.scheduler_fn = self._build_scheduler()

    # --------------------- 
    # GROUP PARAMS BY MODEL DEPTH 
    # ----------------------- 
    def group_params_by_depth(
        self,
        module,
        num_groups=3,
        filter_by_prefix=None,
        return_names=False,
    ):
        """
        Model-agnostic grouping by depth (dot-count).
        """

        layers = []
        for name, p in module.named_parameters():
            if filter_by_prefix and not name.startswith(filter_by_prefix):
                continue
            depth = name.count(".")
            layers.append((depth, name, p))

        if len(layers) == 0:
            return []

        layers.sort(key=lambda x: x[0])  # shallow → deep

        group_size = max(1, len(layers) // num_groups)
        groups = []

        for i in range(num_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < num_groups - 1 else len(layers)
            slice_layers = layers[start:end]

            if return_names:
                groups.append([(name, p) for (_, name, p) in slice_layers])
            else:
                groups.append([p for (_, _, p) in slice_layers])

        return groups

    # ----------------------------------------------------
    # BACKBONE FREEZE / UNFREEZE
    # ----------------------------------------------------
    def _backbone_freeze_all(self):
        for group in self.backbone_groups:
            for p in group:
                p.requires_grad = False

    def _unfreeze_group(self, group_index):
        for p in self.backbone_groups[group_index]:
            p.requires_grad = True

    def gradual_unfreeze(self, epoch, unfreeze_every_n_epochs=2):
        """
        Gradually unfreeze from deepest → shallowest.
        """
        num_groups = len(self.backbone_groups)

        groups_to_unfreeze = min(
            (epoch // unfreeze_every_n_epochs) + 1,
            num_groups,
        )

        # deepest groups: last groups
        start = num_groups - groups_to_unfreeze
        for i in range(start, num_groups):
            self._unfreeze_group(i)

    # ----------------------------------------------------
    # BASE OPTIMIZER
    # ----------------------------------------------------
    def _get_base_optimizer(self, params, cfg):
        opt_name = cfg["optimizer_parameters"]["name"].lower()
        p = cfg["optimizer_parameters"]

        if opt_name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=p["lr"],
                eps=p["eps"],
                betas=p["betas"],
                amsgrad=p["amsgrad"],
                weight_decay=p["weight_decay"],
            )

        elif opt_name == "adam":
            return torch.optim.Adam(
                params,
                lr=p["lr"],
                eps=p["eps"],
                betas=p["betas"],
                amsgrad=p["amsgrad"],
                weight_decay=p["weight_decay"],
            )

        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    # ----------------------------------------------------
    # OPTIMIZER WITH DISCRIMINATIVE LR FOR WHOLE MODEL
    # ----------------------------------------------------
    def _build_optimizer(self):
        cfg = self.config
        p = cfg["optimizer_parameters"]

        use_discriminative_lr =  p['discriminative_lr'] 

        if not use_discriminative_lr:
            # Lightning expects a factory: lambda params: optimizer
            return lambda params: self._get_base_optimizer(params, cfg)

        # ------------- discriminative LR version ----------------
        num_groups = p["num_lr_groups"]
        lr_decay_factor = p["lr_decay_factor"]
        weight_decay = p["weight_decay"]

        # Group whole model by depth (NOT only backbone)
        depth_groups = self.group_params_by_depth(self.model, num_groups=num_groups)

        base_lr = p["lr"]
        param_groups = []

        for i, group in enumerate(depth_groups):
            # Deepest = highest LR
            lr = base_lr / (lr_decay_factor ** (len(depth_groups) - 1 - i))

            param_groups.append({
                "params": group,
                "lr": lr,
                "weight_decay": weight_decay,
            })

        return lambda _: self._get_base_optimizer(param_groups, cfg)

    # ----------------------------------------------------
    # LR SCHEDULERS
    # ----------------------------------------------------
    def _build_scheduler(self):
        sch_cfg = self.config["scheduler"]
        if sch_cfg is None:
            return None

        name = sch_cfg["name"].lower()

        # --- schedules picker
        if name == "reduce_lr_on_plateau":

            def _scheduler(optimizer):
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=sch_cfg["factor"],
                    patience=sch_cfg["patience"],
                    min_lr=sch_cfg["min_lr"],
                    threshold=sch_cfg["threshold"]
                )
                return {
                    "scheduler": scheduler,
                    "monitor": sch_cfg['monitor'],
                    "interval": "epoch",
                }

            return _scheduler

        # --- Cosine Annealing ---
        if name == "cosine":

            def _scheduler(optimizer):
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max= sch_cfg['T_max'],
                    eta_min=sch_cfg["eta_min"],
                )
                return {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }

            return _scheduler

        # --- Cosine w/ Warmup ---
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
                return {
                    "scheduler": scheduler,
                    "interval": "step",
                }

            return _scheduler

        raise ValueError(f"Unknown scheduler: {name}")