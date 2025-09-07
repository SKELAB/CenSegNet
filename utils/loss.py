import torch
import torch.nn as nn

# Dice Loss class
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Small constant to prevent division by zero

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Apply sigmoid to convert logits to probabilities
        intersection = (pred * target).sum()  # Compute intersection of predicted and target
        union = pred.sum() + target.sum()  # Compute union of predicted and target
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # Dice coefficient
        return 1 - dice  # Dice loss is 1 - Dice coefficient


# BCE + Dice Loss class
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight  # Weight for BCE loss
        self.dice_weight = dice_weight  # Weight for Dice loss
        self.bce_loss = nn.BCEWithLogitsLoss()  # BCE loss with logits
        self.dice_loss = DiceLoss(smooth=smooth)  # Dice loss

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)  # Compute BCE loss
        dice = self.dice_loss(pred, target)  # Compute Dice loss
        return self.bce_weight * bce + self.dice_weight * dice  # Weighted sum of BCE and Dice
