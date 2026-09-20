import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve
from sklearn.manifold import TSNE

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
)
import numpy as np
import torch

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_detailed(model, dataloader, criterion, device):
    """Evaluate binary predictions and select the validation F1 threshold.

    Returns loss, classification metrics, the confusion matrix, labels,
    positive-class probabilities, and the selected threshold. Probabilities
    use sigmoid because the model is trained with BCEWithLogitsLoss.
    """
    model.eval()
    gt = []
    prob_positive_list = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch

            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device).float()            # {0,1} -> {0.0,1.0}

            logits = model(inputs)                        # shape: [batch]
            loss = criterion(logits, labels)             # BCEWithLogitsLoss
            total_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(logits)                 # shape: [batch], in (0,1)
            prob_positive_list.extend(probs.cpu().tolist())
            gt.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)

    # Select the threshold that maximizes F1 on this validation set.
    prob_np = np.array(prob_positive_list)
    gt_np   = np.array(gt)

    precisions, recalls, thresholds = precision_recall_curve(gt_np, prob_np)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    # precision_recall_curve returns one fewer threshold than precision/recall.
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    final_preds = (prob_np >= best_threshold).astype(int)

    acc  = accuracy_score(gt_np, final_preds)
    prec = precision_score(gt_np, final_preds, zero_division=0)
    rec  = recall_score(gt_np, final_preds, zero_division=0)
    f1   = f1_score(gt_np, final_preds, zero_division=0)
    try:
        auc = roc_auc_score(gt_np, prob_np)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(gt_np, final_preds)

    return avg_loss, acc, prec, rec, f1, auc, cm, gt, prob_positive_list, best_threshold


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch


def visualize_features(model, dataloader, device, perplexity=5, save_path=None):
    """Project fused model features to 2D with t-SNE and plot by class."""
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            inputs = [x.to(device) for x in inputs]
            features, _ = model.get_features(inputs)
            features_list.append(features.cpu())
            labels_list.append(labels.cpu())
    features_all = torch.cat(features_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features_2d = tsne.fit_transform(features_all.numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=labels_all.numpy(), cmap='coolwarm', alpha=0.7)
    plt.title("t-SNE Visualization of Fused Features")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Label")

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved t-SNE plot to {save_path}")
    else:
        plt.show()
