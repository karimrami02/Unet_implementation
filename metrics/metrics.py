import torch
import numpy as np


def dice_score(pred, target, smooth=1e-6):
    """
    Calculate Dice score (F1 score) for binary segmentation.
    
    Args:
        pred: Predicted mask (B, H, W) or (H, W)
        target: Ground truth mask (B, H, W) or (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice score
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item() if isinstance(dice, torch.Tensor) else dice


def iou_score(pred, target, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union) score.
    
    Args:
        pred: Predicted mask (B, H, W) or (H, W)
        target: Ground truth mask (B, H, W) or (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item() if isinstance(iou, torch.Tensor) else iou


def pixel_accuracy(pred, target):
    """
    Calculate pixel accuracy.
    
    Args:
        pred: Predicted mask (B, H, W) or (H, W)
        target: Ground truth mask (B, H, W) or (H, W)
    
    Returns:
        Pixel accuracy
    """
    pred = pred.flatten()
    target = target.flatten()
    
    correct = (pred == target).sum()
    total = target.numel()
    
    acc = correct.float() / total
    return acc.item() if isinstance(acc, torch.Tensor) else acc


def precision_recall(pred, target, smooth=1e-6):
    """
    Calculate precision and recall.
    
    Args:
        pred: Predicted mask (B, H, W) or (H, W)
        target: Ground truth mask (B, H, W) or (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Tuple of (precision, recall)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    true_positive = (pred * target).sum()
    pred_positive = pred.sum()
    target_positive = target.sum()
    
    precision = (true_positive + smooth) / (pred_positive + smooth)
    recall = (true_positive + smooth) / (target_positive + smooth)
    
    if isinstance(precision, torch.Tensor):
        precision = precision.item()
    if isinstance(recall, torch.Tensor):
        recall = recall.item()
    
    return precision, recall


class MetricTracker:
    """
    Track metrics during training/evaluation
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.dice_scores = []
        self.iou_scores = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
    
    def update(self, pred, target):
        """
        Update metrics with new predictions
        
        Args:
            pred: Predicted mask (B, H, W)
            target: Ground truth mask (B, H, W)
        """
        batch_size = pred.size(0)
        
        for i in range(batch_size):
            dice = dice_score(pred[i], target[i])
            iou = iou_score(pred[i], target[i])
            acc = pixel_accuracy(pred[i], target[i])
            prec, rec = precision_recall(pred[i], target[i])
            
            self.dice_scores.append(dice)
            self.iou_scores.append(iou)
            self.accuracies.append(acc)
            self.precisions.append(prec)
            self.recalls.append(rec)
    
    def get_metrics(self):
        """
        Get average metrics
        
        Returns:
            Dictionary of average metrics
        """
        return {
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'accuracy': np.mean(self.accuracies) if self.accuracies else 0.0,
            'precision': np.mean(self.precisions) if self.precisions else 0.0,
            'recall': np.mean(self.recalls) if self.recalls else 0.0,
        }
