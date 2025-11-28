import torch
import torch.nn as nn
import torch.optim as optim

from loss import *
import pytorch_lightning as pl



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

#lightning optimizer selector
class LightningOptimizerFactory():

    def __init__(self, model, parameters, model_type):
          #super().__init__()
          self.parameters = parameters
          self.model= model
          self.model_type = model_type
          self.optimizer_fn  = self.get_optimizer()
          
    #def get_optimizer(self, model, parameters, model_type):
    def get_optimizer(self):
        #params unused
        ___optimizer_parameters =  self.parameters[f"{self.model_type}_model_parameters"]['optimizer_parameters']
        optimizer_type = ___optimizer_parameters['name'].lower()
        if optimizer_type == "adamw":
            return lambda params: torch.optim.AdamW(
            params,
            lr=___optimizer_parameters["lr"],
            betas=___optimizer_parameters["betas"],
            eps=___optimizer_parameters["eps"],
            weight_decay=___optimizer_parameters["weight_decay"],
            amsgrad=___optimizer_parameters["amsgrad"],
        )
        elif optimizer_type == "adam":
            return  lambda params: torch.optim.Adam(
            params,
            lr=___optimizer_parameters["lr"],
            betas=___optimizer_parameters["betas"],
            eps=___optimizer_parameters["eps"],
            weight_decay=___optimizer_parameters["weight_decay"],
            amsgrad=___optimizer_parameters["amsgrad"],
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

