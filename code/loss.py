from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch


def proj_cosine_loss(a, b, eps=1e-8):
    cos_sim = F.cosine_similarity(a, b, dim=1, eps=eps)
    return (1.0 - cos_sim).mean()

class DiceBCELoss(nn.Module):
    """
    Foreground-only Dice + BCE loss, 2D/3D compatible.
    Input:
        pred_logits: (B, 1, H, W) or (B, 1, D, H, W) raw logits
        target: (B, 1, H, W) or (B, 1, D, H, W) binary mask (0/1)
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, pred_logits, target):
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred_logits, target)

        # Sigmoid to get probabilities
        pred_probs = torch.sigmoid(pred_logits)

        # Flatten batch + spatial dims
        pred_flat = pred_probs.view(pred_probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Foreground Dice
        intersection = (pred_flat * target_flat).sum(dim=1)
        denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.eps
        dice_score = (2.0 * intersection) / denom
        dice_loss = 1.0 - dice_score.mean()

        # Combine BCE + Dice
        loss = self.bce_weight * bce + self.dice_weight * dice_loss
        return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W) or (B, 1, D, H, W)
        targets: same shape, {0,1}
        """
        probs = torch.sigmoid(logits)

        dims = tuple(range(2, probs.ndim))  # spatial dims
        intersection = (probs * targets).sum(dims)
        union = probs.sum(dims) + targets.sum(dims)

        dice = (2. * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


#note, will overwrite smoothed loss
class FocalLoss(nn.Module):
    # Original FocalLoss implementation (scalar alpha)
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

# #note, will overwrite smoothed loss
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        # alpha can be None (uniform weighting), a scalar, or a tensor of weights
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (int, float)):
                 # Use scalar alpha
                 alpha = self.alpha
            else:
                 # Use per-class alpha weights
                 # Ensure alpha is on the same device as inputs
                 alpha = self.alpha.to(inputs.device)
                 
                 # Handle potential soft labels (one-hot/smoothed) by recovering indices
                 if targets.ndim > 1:
                     target_indices = targets.argmax(dim=1)
                 else:
                     target_indices = targets
                     
                 # Gather alpha values corresponding to the target classes
                 # Explicitly cast indices to long (int64) for gather
                 alpha = alpha.gather(0, target_indices.view(-1).long())
                 
            focal_loss = alpha * (1 - pt)**self.gamma * ce_loss
        else:
            # No alpha weighting (equivalent to alpha=1)
            focal_loss = (1 - pt)**self.gamma * ce_loss


        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class SoftFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Convert hard labels to one-hot if needed
        if targets.dim() == 1:
            targets = F.one_hot(targets, logits.size(1)).float()

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        focal_weight = (1 - probs) ** self.gamma

        loss = -(targets * focal_weight * log_probs).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class SoftWeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if class_weights is not None:
            self.class_weights = class_weights.view(1, -1)  # [1,C]
        else:
            self.class_weights = None
        
    def forward(self, logits, targets):

        # Convert hard labels to one-hot
        if targets.dim() == 1:
            targets = F.one_hot(targets, logits.size(1)).float()

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        focal_weight = (1 - probs) ** self.gamma

        if self.class_weights is not None:
            focal_weight = focal_weight * self.class_weights

        loss = -(targets * focal_weight * log_probs).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothing(nn.Module):
    """
    label smoother
    """
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        # pred: model predictions (logits or probabilities)
        # target: ground truth labels (integer indices)

        # Force target to be LongTensor (int64) for scatter, needed for compatibility
        target = target.long()

        # Create a tensor of the same shape as predictions, filled with the smoothing value
        # and then distribute the smoothing mass
        true_dist = pred.data.clone()
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return true_dist