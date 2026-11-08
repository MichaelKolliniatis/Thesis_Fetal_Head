"""
Custom Loss Functions for Multi-Class Medical Image Segmentation
Περιλαμβάνει το Multi-Class Batch Soft Dice Loss και τον συνδυασμό
Cross-Entropy + Soft Dice Loss (CEDiceLoss) για αντιμετώπιση
έντονης ανισορροπίας κλάσεων (Class Imbalance).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Πολυκλασικό Soft Dice Loss σε επίπεδο batch (pseudo-volume).
    Αγνοεί το μαύρο φόντο (κλάση 0) και υπολογίζει τον μέσο όρο μόνο στις δομές
    που εμφανίζονται στο τρέχον batch, ώστε τα άδεια batches (π.χ. χωρίς CSP)
    να μην σπρώχνουν τις πιθανότητες της σπάνιας κλάσης σε κορεσμό (0.0).
    """

    def __init__(self, smooth=1e-5, ignore_background=True):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Έξοδος του δικτύου πριν το Softmax, σχήματος (B, C, H, W)
            targets (torch.Tensor): Πραγματικές μάσκες με δείκτες κλάσεων, σχήματος (B, H, W)
        """
        num_classes = logits.shape[1]

        # 1. Μετατροπή των raw logits σε πιθανότητες [0, 1] ανά pixel μέσω Softmax
        probs = F.softmax(logits, dim=1)

        # 2. One-Hot κωδικοποίηση της πραγματικής μάσκας από (B, H, W) σε (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 3. Εξαίρεση του φόντου (κλάση 0) ώστε να βελτιστοποιούνται ισότιμα τα όργανα
        if self.ignore_background and num_classes > 1:
            probs = probs[:, 1:, :, :]
            targets_one_hot = targets_one_hot[:, 1:, :, :]

        # 4. Υπολογισμός Batch-level Soft Dice ανά κλάση αθροίζοντας σε (Batch, Height, Width)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        pred_sum = torch.sum(probs, dim=dims)
        target_sum = torch.sum(targets_one_hot, dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # Λαμβάνουμε υπόψη μόνο τις δομές που υπάρχουν πραγματικά στο τρέχον batch (target_sum > 0)
        present_mask = target_sum > 0
        if present_mask.any():
            return 1.0 - dice_per_class[present_mask].mean()
        return logits.sum() * 0.0


class CEDiceLoss(nn.Module):
    """
    Συνδυαστική συνάρτηση κόστους: Weighted Cross-Entropy Loss + Masked Soft Dice Loss.
    - Τα βάρη κλάσεων (class_weights) στην Cross-Entropy αποτρέπουν τον κορεσμό του Softmax
      (vanishing gradient) σε σπάνιες/μικροσκοπικές δομές όπως το CSP.
    - Το Soft Dice Loss μεγιστοποιεί απευθείας την επικάλυψη κάθε δομής ανεξαρτήτως μεγέθους.
    """

    def __init__(self, ce_weight=1.0, dice_weight=1.0, class_weights=(0.5, 1.0, 15.0, 5.0), smooth=1e-5):
        super(CEDiceLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        self.dice_loss = SoftDiceLoss(smooth=smooth, ignore_background=True)

    def forward(self, logits, targets):
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, targets, weight=weights)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice
