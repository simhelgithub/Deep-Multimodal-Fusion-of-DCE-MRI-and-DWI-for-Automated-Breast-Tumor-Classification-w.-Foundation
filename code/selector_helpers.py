import torch
from torch._dynamo.decorators import skip
import torch.nn as nn
import torch.optim as optim
from torchmetrics.segmentation import DiceScore
from loss import *
from model_module import *
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
  mask_dce_weight = mask_parameters['bce_weight']
  mask_dice_weight = mask_parameters['dice_weight']
  mask_criterion = None
  if mask_enabled:
    if mask_loss_type == "dice":
        mask_criterion = SoftDiceLoss()
    elif mask_loss_type == "dice_bce":
        mask_criterion = DiceBCELoss(bce_weight=1.0, dice_weight=1.0)

    else:
        raise ValueError(f"Invalid mask loss: {mask_loss_type}")
    ''' #no longer supported
    elif mask_loss_type == "BCE":
        mask_criterion = nn.BCEWithLogitsLoss()
    '''
  return mask_criterion


# todo make all after this less ugly
@torch._dynamo.disable
class LightningOptimizerFactory:
    """
    Single-model optimizer factor:
    - Groups: [backbone, block1+block2, block3+other]
    - backbone freezing
    - Gradual unfreeze (deep → shallow)
    - Discriminative LR 
    """

    def __init__(self, model: torch.nn.Module, parameters: dict, model_type: str):
        self.model = model
        self.parameters = parameters
        self.model_type = model_type
        self.config = parameters[f"{model_type}_model_parameters"]

        #self.num_backbone_groups = parameters.get("backbone_num_groups", 3)
        self.backbone_freeze_on_start = parameters.get("backbone_freeze_on_start", True)
        self.layers_unfrozen = 0
        self.use_backbone = self.config["use_backbone"]

        # --- canonical grouping ---
        self.named_groups = self.group_model_with_backbone_params(
            model,
            use_backbone=self.use_backbone
        )

        # freeze if requested
        if self.backbone_freeze_on_start and self.use_backbone:
            self.backbone = self.named_groups[0]
            assert(self.backbone is not None) 
            self.freeze_backbone()
        self.optimizer_fn = self._build_optimizer()
        self.scheduler_fn = self._build_scheduler()

    # ------------------------------------------------------------------
    # Grouping  
    # ------------------------------------------------------------------
    @staticmethod
    def group_model_with_backbone_params(model, use_backbone=True):
        backbone, block1, block2, block3, other = [], [], [], [], []

        for name, p in model.named_parameters():
            if "classification_head" in name:
                continue
            if use_backbone and ("backbone" in name or "backbone_neck" in name):
                backbone.append((name, p))
            elif "block1" in name:
                block1.append((name, p))
            elif "block2" in name:
                block2.append((name, p))
            elif "block3" in name:
                block3.append((name, p))
            else:
                other.append((name, p))

        if use_backbone:
            block2 = block1 + block2
            block1 = backbone

        groups = [block1, block2, block3 + other]

        print(f"[DEBUG] Single-model group sizes: {[len(g) for g in groups]}")
        return groups

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------
    def freeze_backbone(self):
        if not self.use_backbone:
            return
        for _, p in self.named_groups[0]:  
            p.requires_grad = False
        print(f"[INFO] Backbone fully frozen ({len(list(self.named_groups[0]))} params)")

    def unfreeze_backbone(self):
        if not self.use_backbone:
            return
        newly = []
        cnt = 0
        for name, p in self.named_groups[0]:
            if not p.requires_grad: 
              p.requires_grad = True
              cnt += 1
              newly.append((name,p))
        print(f"[INFO] Backbone fully unfrozen ({cnt} params)")
        return newly

    
    def sync_unfrozen_params_to_optimizer(self, optimizer, newly_unfrozen):
        if not newly_unfrozen:
            return
        existing = {id(p) for g in optimizer.param_groups for p in g["params"]}
        new_params = [p for (name, p) in newly_unfrozen if id(p) not in existing]

        lr = self.parameters.get("foundation_model_unfreeze_lr", 1e-6)
        wd =  self.parameters.get("fondation_model_unfreeze_wd", 0)

        optimizer.add_param_group({"params": new_params, "lr": lr, "weight_decay": wd})
        print(f"[INFO] sync: Added {len(new_params)} newly-unfrozen params to optimizer with lr={lr:.6g}, wd={wd:.6g}")

    # ------------------------------------------------------------------
    # Optimizer + discriminative LR
    # ------------------------------------------------------------------
    def _get_base_optimizer(self, params, cfg):
        p = cfg["optimizer_parameters"]
        name = p["name"].lower()
        if name == "adamw":
            return torch.optim.AdamW(
                params, lr=p["lr"], betas=p["betas"], eps=p["eps"],
                weight_decay=p["weight_decay"], amsgrad=p.get("amsgrad", False)
            )
        if name == "adam":
            return torch.optim.Adam(
                params, lr=p["lr"], betas=p["betas"], eps=p["eps"],
                weight_decay=p["weight_decay"], amsgrad=p.get("amsgrad", False)
            )
        raise ValueError(name)

    def _build_optimizer(self):
        """
        Build optimizer with optional discriminative LR and discriminative weight decay.
        Backbone freezing/unfreezing is handled as all-or-nothing.
        """
        cfg = self.config
        p = cfg["optimizer_parameters"]
        base_lr = p.get("lr", 1e-4)
        weight_decay = p.get("weight_decay", 0.0)
        lr_decay_factor = p.get("lr_decay_factor", 2.0)
        use_discriminative_lr = p.get("discriminative_lr", False)
        use_discriminative_reg = p.get("discriminative_reg", False)
        reg_base = p.get("reg_base", weight_decay)
        reg_decay_factor = p.get("reg_decay_factor", 2.0)

        named_groups = self.named_groups


        param_groups = []
        n_groups = max(1, len(named_groups))
        for i, group in enumerate(named_groups):
            params_list = [p for (n, p) in group if p.requires_grad]
            if not params_list:
                continue

            # Learning rate: deeper group → higher LR
            if use_discriminative_lr:
              lr = base_lr / (lr_decay_factor ** (n_groups - 1 - i))
            else:
              lr = base_lr
            # Weight decay: discriminative if enabled
            if use_discriminative_reg:
                wd = reg_base * (reg_decay_factor ** (n_groups - 1 - i))
            else:
                wd = weight_decay

            param_groups.append({
                "params": params_list,
                "lr": lr,
                "weight_decay": wd,
            })
            

        # Safety fallback
        if not param_groups:
            print('no param groups')
            all_trainable = [p for (n, p) in (named_groups) if p.requires_grad]
            param_groups = [{"params": all_trainable, "lr": base_lr, "weight_decay": weight_decay}]

        self.print_param_group_summary(param_groups, tag="FINAL OPTIMIZER PARAM GROUPS")
        return lambda _: self._get_base_optimizer(param_groups, cfg)

    # ---------------------
    # Build LR scheduler  
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


@torch._dynamo.disable
class LightningFusionOptimizerFactory:
    """
    - Gradual unfreeze of DWI/DCE: backbone / block1+2 / block3
    - Keeps fusion head always trainable
    - Supports discriminative LR and weight decay
    - Debug printing of parameter groups
    - Scheduler support
    """

    def __init__(self, dwi_model: torch.nn.Module, dce_model: torch.nn.Module, fusion_model: torch.nn.Module, parameters: dict):
        self.dwi_model = dwi_model
        self.dce_model = dce_model
        self.fusion_model = fusion_model
        self.parameters = parameters

        # Config knobs
        self.num_backbone_groups = parameters.get("backbone_num_groups", 3)
        self.backbone_freeze_on_start = parameters.get("backbone_freeze_on_start", True)
        self.layers_unfrozen = 0

        # Split DWI and DCE into gradual-unfreeze groups & remove classifciation heads
        self.dwi_named_groups = self.group_model_with_backbone_params(dwi_model, use_backbone=self.parameters["dwi_model_parameters"]["use_backbone"], expected_num_groups=self.num_backbone_groups)
        self.dce_named_groups = self.group_model_with_backbone_params(dce_model, use_backbone=self.parameters["dce_model_parameters"]["use_backbone"], expected_num_groups=self.num_backbone_groups)
        self.fusion_named = list(fusion_model.named_parameters())  # always trainable
        

        # Freeze DWI/DCE backbones if requested
        if self.backbone_freeze_on_start:
            self._freeze_all_backbone_groups()        
            self.newly_unfrozen_groups = []

        # Build optimizer + scheduler factories
        self.optimizer_fn = self._build_optimizer()
        self.scheduler_fn = self._build_scheduler()


    # ---------------------
    # Parameter grouping
    # ---------------------
    @staticmethod
    def group_model_with_backbone_params(model, use_backbone=True, expected_num_groups=3):
        """
        Name-based grouping with safer handling:
          - groups: [backbone, (block1+block2), block3+other]
        Skips classification/mask/projector heads listed in skip_prefixes.
        Returns list of groups where each group is list of (name, param).
        """
        backbone_params, block1_params, block2_params, block3_params, other_params = [], [], [], [], []
        # ensure tuple
        for name, p in model.named_parameters():
          #print(name)
          # skip unwanted heads
          if "classification_head" in name:
              continue
          if use_backbone and ("backbone" in name or "backbone_neck" in name):
              backbone_params.append((name, p))
          elif "block1" in name:
              block1_params.append((name, p))
          elif "block2" in name:
              block2_params.append((name, p))
          elif "block3" in name:
              block3_params.append((name, p))
          else:
              other_params.append((name, p))


        if use_backbone:
            # combine block1+block2 
            block2_params = block1_params + block2_params
            block1_params = backbone_params

        groups = [block1_params, block2_params, block3_params + other_params]

        # Debug: print sizes so you can detect collapsed grouping early
        print(f"[DEBUG] group_model_with_backbone_params sizes: {[len(g) for g in groups]}")
        
        return groups
    # ---------------------
    # Freezing
    # ---------------------
    def _freeze_all_backbone_groups(self):
        # Freeze DWI: operate on the real model parameters
        for name, p in self.dwi_model.named_parameters():
            p.requires_grad = False
        # Freeze DCE
        for name, p in self.dce_model.named_parameters():
            p.requires_grad = False

        # Debug print 
        print("[DEBUG] After freeze: trainable params (DWI):")
        for name, p in self.dwi_model.named_parameters():
            if p.requires_grad:
                print("  (still trainable!)", name)
        print("[DEBUG] After freeze: trainable params (DCE):")
        for name, p in self.dce_model.named_parameters():
            if p.requires_grad:
                print("  (still trainable!)", name)


    def _build_optimizer(self):
        """
        Build the initial optimizer factory. If backbone_freeze_on_start is True,
        backbone groups are excluded from optimizer initially (only fusion head included).
        """
        cfg = self.parameters["fusion_model_parameters"]
        p = cfg["optimizer_parameters"]
        use_discriminative_lr = p.get("discriminative_lr", False)
        freeze_backbone = self.backbone_freeze_on_start

        # If not discriminative LR, create a single param group (Lightning will pass params)
        if not use_discriminative_lr:
            return lambda params: self._get_base_optimizer(params, cfg)

        # Discriminative LR: build named group sequence
        lr_decay_factor = p.get("lr_decay_factor", 2.0)
        weight_decay = p.get("weight_decay", 0.0)
        base_lr = p.get("lr", 1e-3)
        use_discriminative_reg = p.get("discriminative_reg", False)
        reg_base = p.get("reg_base", weight_decay)
        reg_decay_factor = p.get("reg_decay_factor", 2.0)

        # Merge groups by depth (dce groups then dwi groups)
        num_backbone_groups = max(len(self.dce_named_groups), len(self.dwi_named_groups))
        merged_groups = []
        for i in range(num_backbone_groups):
            g = []
            if i < len(self.dce_named_groups):
                g += self.dce_named_groups[i]
            if i < len(self.dwi_named_groups):
                g += self.dwi_named_groups[i]
            merged_groups.append(g)

        # Always append fusion head as the last group
        merged_groups.append(self.fusion_named)

        param_groups = []
        n_groups = max(1, len(merged_groups))
        for i, named_group in enumerate(merged_groups):
            # collect parameters (named_group is list of (name, param))
            params_list = [p for (_n, p) in named_group]

            # If backbone is frozen on start, exclude backbone groups from initial optimizer
            if freeze_backbone and i < (n_groups - 1):
                params_list = []

            if not params_list:
                continue

            lr = base_lr / (lr_decay_factor ** (n_groups - 1 - i))
            wd = (reg_base * (reg_decay_factor ** (n_groups - 1 - i))) if use_discriminative_reg else weight_decay

            param_groups.append({"params": params_list, "lr": lr, "weight_decay": wd})

        # Safety fallback: everything
        if not param_groups:
            all_params = [p for (n, p) in (list(self.dwi_model.named_parameters()) +
                                          list(self.dce_model.named_parameters()) +
                                          list(self.fusion_named))]
            param_groups = [{"params": all_params, "lr": base_lr, "weight_decay": weight_decay}]

        self.print_param_group_summary(param_groups, tag="FINAL FUSION PARAM GROUPS (initial)")
        return lambda _: self._get_base_optimizer(param_groups, cfg)
  



    def _unfreeze_named_group(self, named_group, model):
        """
        named_group: list of (name, param)
        model: the actual model to modify
        Returns: (count_unfrozen, list_of_newly_unfrozen_param_objects)
        """
        model_params = dict(model.named_parameters())
        newly = []
        cnt = 0
        for name, _ in named_group:
            if name in model_params:
                mp = model_params[name]
                if not mp.requires_grad:
                    mp.requires_grad = True
                    cnt += 1
                    newly.append(mp)
        return cnt, newly

    def gradual_unfreeze(self, epoch, unfreeze_every_n_epochs=20):
        """
        Unfreeze ONE next group at the scheduled epochs (deep->shallow).
        Call from your LightningModule.on_train_epoch_start 
        Returns list of newly unfrozen Parameter objects 
        """

        # do not unfreeze at epoch 0
        if epoch == 0:
            return []

        # only at exact multiples of unfreeze_every_n_epochs
        if unfreeze_every_n_epochs <= 0 or epoch % unfreeze_every_n_epochs != 0:
            return []

        # how many steps already done
        done_steps = self.layers_unfrozen
        if done_steps >= self.num_backbone_groups:
            return []  # nothing left to unfreeze

        # determine which group to unfreeze (deep -> shallow)
        # index 0 = deepest in your groups list, so pick:
        group_idx = self.num_backbone_groups - 1 - done_steps
        #group_idx = self.layers_unfrozen 
        if group_idx < 0 or group_idx >= self.num_backbone_groups:
            return []

        dwi_named = self.dwi_named_groups[group_idx]
        dce_named = self.dce_named_groups[group_idx]

        cnt_dwi, new_dwi = self._unfreeze_named_group(dwi_named, self.dwi_model)
        cnt_dce, new_dce = self._unfreeze_named_group(dce_named, self.dce_model)

        newly = new_dwi + new_dce

        if cnt_dwi > 0 or cnt_dce > 0:
            print(f"[INFO] Epoch {epoch}: Unfroze ONLY group {group_idx} (DWI: {cnt_dwi}, DCE: {cnt_dce})")
            # increment counter only when we actually unfreeze something
            self.layers_unfrozen += 1
        else:
            print(f"[DEBUG] Epoch {epoch}: attempted to unfreeze group {group_idx} but nothing changed")

        # return newly-unfrozen Parameter objects for downstream sync
        return newly



    def sync_unfrozen_params_to_optimizer(self, optimizer, newly_unfrozen_params):
        """
        Add newly_unfrozen_params (list of Parameter) to optimizer as a single param-group.
        Ensures we do not add any parameter already present in optimizer.
        """
        if not newly_unfrozen_params:
            print("[DEBUG] sync: no newly_unfrozen_params")
            return

        existing_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        to_add = [p for p in newly_unfrozen_params if isinstance(p, torch.nn.Parameter) and id(p) not in existing_ids]
        if not to_add:
            print("[DEBUG] sync: after filtering, nothing to add (maybe already present).")
            return

        # compute backbone lr/wd scheduling
        base_lr = self.parameters.get("backbone_unfreeze_lr", 1e-4)
        factor = self.parameters.get("backbone_unfreeze_lr_factor", 1.0)
        backbone_lr = base_lr * (factor ** (self.layers_unfrozen - 1))  # note: layers_unfrozen already incremented in gradual_unfreeze

        base_wd = self.parameters["dwi_model_parameters"]['optimizer_parameters'].get('reg_base', 0.0)
        factor_wd = self.parameters["dwi_model_parameters"]['optimizer_parameters'].get('reg_decay_factor', 1.0)
        wd = base_wd * (factor_wd ** (self.layers_unfrozen - 1))

        optimizer.add_param_group({"params": to_add, "lr": backbone_lr, "weight_decay": wd})
        print(f"[INFO] sync: Added {len(to_add)} newly-unfrozen params to optimizer with lr={backbone_lr:.6g}, wd={wd:.6g}")


    
    # ---------------------
    # Base optimizer builder
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

    def _build_optimizer(self):
        cfg = self.parameters["fusion_model_parameters"]
        p = cfg["optimizer_parameters"]
        use_discriminative_lr = p.get("discriminative_lr", False)
        if not use_discriminative_lr:
            return lambda params: self._get_base_optimizer(params, cfg)

        # gather config
        lr_decay_factor = p.get("lr_decay_factor", 2.0)
        weight_decay = p.get("weight_decay", 0.0)
        base_lr = p.get("lr", 1e-3)
        use_discriminative_reg = p.get("discriminative_reg", False)
        reg_base = p.get("reg_base", weight_decay)
        reg_decay_factor = p.get("reg_decay_factor", 2.0)

        # merge DCE + DWI groups by depth
        num_backbone_groups = max(len(self.dce_named_groups), len(self.dwi_named_groups))
        merged_groups = []

        if not self.backbone_freeze_on_start:
          for i in range(num_backbone_groups):
              group = []
              if i < len(self.dce_named_groups):
                  group += self.dce_named_groups[i]
              if i < len(self.dwi_named_groups):
                  group += self.dwi_named_groups[i]
              if group:
                  merged_groups.append(group)

        # append fusion head as last group
        merged_groups.append(self.fusion_named)

        # build param groups with per-depth LR/WD
        param_groups = []
        n_groups = len(merged_groups)
        for i, named_group in enumerate(merged_groups):
            params_list = [p for (_n, p) in named_group]
            if not params_list:
                continue
            # deeper groups → larger LR
            lr = base_lr / (lr_decay_factor ** (n_groups - 1 - i))
            # discriminative WD if requested
            wd = (reg_base * (reg_decay_factor ** (n_groups - 1 - i))) if use_discriminative_reg else weight_decay
            param_groups.append({"params": params_list, "lr": lr, "weight_decay": wd})

        if not param_groups:
            # fallback: everything
            all_params = [p for (n, p) in (list(self.dwi_model.named_parameters()) +
                                          list(self.dce_model.named_parameters()) +
                                          list(self.fusion_named))]
            param_groups = [{"params": all_params, "lr": base_lr, "weight_decay": weight_decay}]

        self.print_param_group_summary(param_groups, tag="FINAL FUSION PARAM GROUPS")
        return lambda _: self._get_base_optimizer(param_groups, cfg)


    # ---------------------
    # Scheduler
    # ---------------------
    def _build_scheduler(self):
        cfg = self.parameters["fusion_model_parameters"]
        sch_cfg = cfg.get("scheduler", None)
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
        raise ValueError(f"Unknown scheduler: {sch_cfg['name']}")

    # ---------------------
    # Debug print helpers
    # ---------------------
    @staticmethod
    def print_param_group_summary(param_groups, tag=""):
        print("\n" + "-"*60)
        print(f"[DEBUG] {tag}")
        for i, g in enumerate(param_groups):
            cnt = len(g.get("params", []))
            lr = g.get("lr", None)
            wd = g.get("weight_decay", None)
            print(f"  ParamGroup {i}: count={cnt}  lr={lr}  wd={wd}")
        print("-"*60 + "\n")


def remove_classification_head(named_params, head_prefixes=("fc", "classifier", "head")):
    return [(n, p) for (n, p) in named_params 
            if not n.startswith(head_prefixes)]