import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    Standard Cross Entropy Loss wrapper for compatibility.
    """
    def __init__(self, class_weights=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
    def forward(self, inputs, targets):
        return self.loss(inputs, targets)

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    Args:
        gamma (float): Focusing parameter.
        class_weights (torch.Tensor): Optional class weights.
    """
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_modulator = (1 - pt) ** self.gamma
        ce_loss = F.nll_loss(log_probs, targets, weight=self.class_weights, reduction='none')
        return (focal_modulator * ce_loss).mean()

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss.
    Args:
        smoothing (float): Smoothing factor.
        class_weights (torch.Tensor): Optional class weights.
    """
    def __init__(self, smoothing=0.1, class_weights=None):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
        ce_loss = -torch.sum(true_dist * log_probs, dim=1)
        if self.class_weights is not None:
            weighted_loss = self.class_weights[targets] * ce_loss
            return weighted_loss.mean()
        else:
            return ce_loss.mean()

class CPFLSLoss(nn.Module):
    """
    Cost-sensitive Focal Label Smoothing (CP-FLS) Loss.

    This loss function combines three key ideas for handling highly imbalanced classification:
    1. Focal Loss: To focus training on hard-to-classify examples by down-weighting easy ones.
    2. Label Smoothing: To prevent the model from becoming overconfident.
    3. Cost-sensitive Weighting: To give more importance to minority classes.

    Args:
        class_weights (torch.Tensor): A tensor of weights for each class.
        gamma (float): The focusing parameter for the Focal Loss (gamma > 0).
        smoothing (float): The label smoothing factor (0.0 <= smoothing < 1.0).
    """
    def __init__(self, class_weights, gamma=2.0, smoothing=0.1):
        super(CPFLSLoss, self).__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing value must be between 0 and 1, but got {smoothing}")
        if gamma < 0:
            raise ValueError(f"gamma value must be non-negative, but got {gamma}")

        self.class_weights = class_weights
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): The model's raw output (logits) of shape (N, C).
            targets (torch.Tensor): The ground truth labels of shape (N).

        Returns:
            torch.Tensor: The calculated loss.
        """
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = F.softmax(inputs, dim=1)

        # Apply label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)

        # Calculate the cross-entropy part of the loss with smoothed labels
        ce_loss = -torch.sum(true_dist * log_probs, dim=1)

        # Get the probabilities of the true classes
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_modulator = (1 - pt) ** self.gamma

        # Apply cost-sensitive weights and focal modulator
        weighted_loss = self.class_weights[targets] * focal_modulator * ce_loss

        return weighted_loss.mean()