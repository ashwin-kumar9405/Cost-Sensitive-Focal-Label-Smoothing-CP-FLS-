import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import os
import json

from model import SimpleCNN
from cpfls_loss import CPFLSLoss, CrossEntropyLoss, FocalLoss, LabelSmoothingLoss
from utils import get_imbalanced_cifar10, get_class_weights, calculate_metrics, plot_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds_labels = []
    all_preds_probs = []

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_preds_labels.extend(preds.cpu().numpy())
            all_preds_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds_labels), np.array(all_preds_probs))
    return epoch_loss, metrics

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds_labels = []
    all_preds_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_preds_labels.extend(preds.cpu().numpy())
            all_preds_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds_labels), np.array(all_preds_probs))
    return epoch_loss, metrics

def main(args):
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load and prepare datasets
    print("Loading CIFAR-10 dataset...")
    train_dataset = get_imbalanced_cifar10(
        root=args.dataset_path,
        imbalance_ratio=args.imbalance_ratio,
        train=True,
        transform=transform_train,
        download=True
    )
    
    # Use the original balanced test set for validation and testing
    test_set = get_imbalanced_cifar10(root=args.dataset_path, train=False, transform=transform_test, download=True)
    val_size = int(0.5 * len(test_set))
    test_size = len(test_set) - val_size
    val_dataset, test_dataset = random_split(test_set, [val_size, test_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = SimpleCNN(num_classes=10).to(device)
    class_weights = get_class_weights(train_dataset, num_classes=10, q=args.cost_weight_q).to(device)

    # Select loss function
    loss_type = getattr(args, 'loss_type', 'cpfls')
    if loss_type == 'ce':
        criterion = CrossEntropyLoss(class_weights=class_weights)
    elif loss_type == 'focal':
        criterion = FocalLoss(gamma=args.gamma, class_weights=class_weights)
    elif loss_type == 'ls':
        criterion = LabelSmoothingLoss(smoothing=args.smoothing, class_weights=class_weights)
    else:
        criterion = CPFLSLoss(class_weights=class_weights, gamma=args.gamma, smoothing=args.smoothing)
    print(f"Using loss function: {loss_type}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': [], 'val_auc': []
    }
    best_val_f1 = 0.0
    best_model_path = os.path.join(args.results_dir, 'best_model.pth')

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_score']:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}, AUC: {val_metrics['auc']:.4f}")

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_score'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])

        # Save best model
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1-score: {best_val_f1:.4f}")

    # Plot and save metrics
    plot_metrics(history, save_path=os.path.join(args.results_dir, 'training_plots.png'))

    # Evaluate best model on test set
    print("\nEvaluating best model on the test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    # Print test results with safe formatting for None values
    def safe_fmt(val):
        return f"{val:.4f}" if isinstance(val, float) else str(val)
    print(f"Test Results -> Loss: {test_loss:.4f}, Accuracy: {safe_fmt(test_metrics['accuracy'])}, F1-score: {safe_fmt(test_metrics['f1_score'])}, AUC: {safe_fmt(test_metrics['auc'])}, ECE: {safe_fmt(test_metrics['ece'])}, Brier: {safe_fmt(test_metrics['brier'])}")

    # Get test predictions and probabilities for plots
    y_true = []
    y_pred = []
    y_prob = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Save confusion matrix
    from utils import plot_confusion_matrix, plot_calibration_curve, plot_pr_curve
    plot_confusion_matrix(y_true, y_pred, os.path.join(args.results_dir, 'conf_matrix.png'))
    if len(np.unique(y_true)) == 2:
        plot_calibration_curve(y_true, y_prob, os.path.join(args.results_dir, 'calibration_curve.png'))
        plot_pr_curve(y_true, y_prob, os.path.join(args.results_dir, 'pr_curve.png'))
    else:
        print("Skipping calibration curve and PR curve plot for multiclass classification.")
    # Save per-epoch metrics
    metrics_history = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'train_f1': history['train_f1'],
        'val_f1': history['val_f1'],
        'train_auc': history['train_auc'],
        'val_auc': history['val_auc']
    }
    with open(os.path.join(args.results_dir, 'metrics_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=4)

    # Save final results
    final_results = {
        'best_validation_f1': best_val_f1,
        'test_metrics': test_metrics,
        'hyperparameters': vars(args)
    }
    results_path = os.path.join(args.results_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Final results saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classifier with CP-FLS Loss on Imbalanced CIFAR-10.')
    parser.add_argument('--dataset_path', type=str, default='./data', help='Path to CIFAR-10 dataset.')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--imbalance_ratio', type=float, default=0.01, help='Ratio of minority to majority class samples (e.g., 0.01 for 1:100).')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma parameter.')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing parameter.')
    parser.add_argument('--cost_weight_q', type=float, default=0.5, help='Exponent for cost-sensitive weighting.')
    parser.add_argument('--loss_type', type=str, default='cpfls', choices=['ce','focal','ls','cpfls'], help='Loss function type: ce, focal, ls, cpfls.')
    args = parser.parse_args()
    main(args)