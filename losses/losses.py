import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Pixel-wise weighted cross entropy loss.
    Used in U-Net paper for handling class imbalance and emphasizing borders.
    """
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets, weight_map):
        """
        Args:
            logits: Tensor (B, C, H, W)
            targets: Tensor (B, H, W)
            weight_map: Tensor (B, H, W)
        
        Returns:
            Weighted cross entropy loss
        """
        loss = self.ce(logits, targets)  # (B, H, W)
        loss = loss * weight_map  # Apply pixel-wise weights
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    """
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor (B, C, H, W)
            targets: Tensor (B, H, W)
        
        Returns:
            Dice loss
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Compute Dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combination of Weighted Cross Entropy and Dice Loss
    """
    
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_loss = WeightedCrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits, targets, weight_map=None):
        """
        Args:
            logits: Tensor (B, C, H, W)
            targets: Tensor (B, H, W)
            weight_map: Tensor (B, H, W) or None
        
        Returns:
            Combined loss
        """
        dice = self.dice_loss(logits, targets)
        
        if weight_map is not None:
            ce = self.ce_loss(logits, targets, weight_map)
        else:
            ce = F.cross_entropy(logits, targets)
        
        return self.ce_weight * ce + self.dice_weight * dice
