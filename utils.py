import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import brier_score_loss, confusion_matrix, precision_recall_curve
from sklearn.calibration import calibration_curve
from torchvision.datasets import CIFAR10
import json
import seaborn as sns

def get_imbalanced_cifar10(root, imbalance_ratio=0.1, train=True, transform=None, download=True):
    """
    Creates an imbalanced version of the CIFAR-10 dataset.

    Args:
        root (str): Root directory of dataset.
        imbalance_ratio (float): The ratio of the number of samples in the smallest class
                                 to the number of samples in the largest class.
        train (bool): If True, creates a training set, otherwise creates a test set.
        transform (callable, optional): A function/transform to apply to the images.
        download (bool): If true, downloads the dataset from the internet.

    Returns:
        A subset of the CIFAR10 dataset with a long-tailed class distribution.
    """
    cifar10 = CIFAR10(root, train=train, transform=transform, download=download)
    
    if not train:
        # Test set is not imbalanced
        return cifar10

    targets = np.array(cifar10.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    num_classes = len(classes)
    
    # Get the number of samples for the majority class
    max_count = class_counts[0]

    # Calculate the number of samples for each class
    imbalanced_class_counts = [int(max_count * (imbalance_ratio ** (i / (num_classes - 1.0)))) for i in range(num_classes)]

    print("Imbalanced class distribution:")
    for i, count in enumerate(imbalanced_class_counts):
        print(f"  Class {i}: {count} samples")

    # Get indices for each class
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]

    # Subsample indices for each class
    imbalanced_indices = []
    for i in range(num_classes):
        indices = class_indices[i]
        np.random.shuffle(indices)
        imbalanced_indices.extend(indices[:imbalanced_class_counts[i]])

    # Create a subset with the imbalanced indices
    imbalanced_dataset = torch.utils.data.Subset(cifar10, imbalanced_indices)
    
    return imbalanced_dataset

def get_class_weights(dataset, num_classes, q=0.5):
    """
    Calculates cost-sensitive weights for each class.
    weight = (max_samples / num_samples_j) ^ q

    Args:
        dataset (torch.utils.data.Dataset): The dataset.
        num_classes (int): The number of classes.
        q (float): Hyperparameter to control the strength of the weighting.

    Returns:
        torch.Tensor: A tensor of weights for each class.
    """
    # For Subset, we need to access the underlying targets
    if isinstance(dataset, torch.utils.data.Subset):
        targets = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        targets = np.array(dataset.targets)

    class_counts = np.bincount(targets, minlength=num_classes)
    max_count = np.max(class_counts)
    
    # Avoid division by zero for classes with no samples
    weights = [(max_count / count) ** q if count > 0 else 0 for count in class_counts]
    
    print(f"Calculated class weights (q={q}):")
    for i, w in enumerate(weights):
        print(f"  Class {i}: {w:.4f}")

    return torch.FloatTensor(weights)

def calculate_metrics(y_true, y_pred_probs, y_pred_labels):
    """
    Calculates accuracy, F1 score, and AUC.

    Args:
        y_true (np.array): True labels.
        y_pred_probs (np.array): Predicted probabilities (for AUC).
        y_pred_labels (np.array): Predicted labels (for Acc and F1).

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    acc = accuracy_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels, average='macro')
    
    # For multi-class AUC, use one-vs-rest
    num_classes = y_pred_probs.shape[1]
    if num_classes > 2:
        auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
    else: # Binary case
        auc = roc_auc_score(y_true, y_pred_probs[:, 1])

    return {'accuracy': acc, 'f1_score': f1, 'auc': auc}

def plot_metrics(history, save_path='results/training_plots.png'):
    """
    Plots and saves the training and validation metrics.

    Args:
        history (dict): A dictionary containing lists of metrics per epoch.
        save_path (str): Path to save the plot image.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Validation Metrics')

    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy over Epochs')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    axes[1, 0].plot(history['train_f1'], label='Train F1-Score')
    axes[1, 0].plot(history['val_f1'], label='Validation F1-Score')
    axes[1, 0].set_title('F1-Score over Epochs')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].legend()

    axes[1, 1].plot(history['train_auc'], label='Train AUC')
    axes[1, 1].plot(history['val_auc'], label='Validation AUC')
    axes[1, 1].set_title('AUC over Epochs')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Training plots saved to {save_path}")


def calculate_metrics(y_true, y_pred, y_prob=None, n_bins=10):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
    if y_prob is not None:
        if len(np.unique(y_true)) == 2:
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            metrics['brier'] = brier_score_loss(y_true, y_prob[:, 1])
            prob_true, prob_pred = calibration_curve(y_true, y_prob[:, 1], n_bins=n_bins)
            metrics['ece'] = np.abs(prob_true - prob_pred).mean()
        else:
            metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            metrics['brier'] = None
            metrics['ece'] = None
    else:
        metrics['auc'] = None
        metrics['ece'] = None
        metrics['brier'] = None
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def plot_calibration_curve(y_true, y_prob, save_path, n_bins=10):
    if len(np.unique(y_true)) == 2:
        prob_true, prob_pred = calibration_curve(y_true, y_prob[:, 1], n_bins=n_bins)
    else:
        prob_true, prob_pred = calibration_curve(y_true, np.max(y_prob, axis=1), n_bins=n_bins)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Reliability')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(y_true, y_prob, save_path):
    if len(np.unique(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    else:
        precision, recall, _ = precision_recall_curve(y_true, np.max(y_prob, axis=1))
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(save_path)
    plt.close()