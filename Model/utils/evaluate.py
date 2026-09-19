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
    """
    验证函数，返回 avg_loss, acc, precision, recall, f1, AUC, 混淆矩阵, 真实标签列表, 预测正类概率列表, 最佳阈值。
    适配 BCEWithLogitsLoss：使用 sigmoid 而非 softmax。
    """
    model.eval()
    gt = []
    prob_positive_list = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            # 解包 batch
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch

            # 送入 device，并确保标签为 float
            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device).float()            # {0,1} -> {0.0,1.0}

            # 前向 + loss
            logits = model(inputs)                        # shape: [batch]
            loss = criterion(logits, labels)             # BCEWithLogitsLoss
            total_loss += loss.item() * labels.size(0)

            # 概率计算
            probs = torch.sigmoid(logits)                 # shape: [batch], in (0,1)
            prob_positive_list.extend(probs.cpu().tolist())
            gt.extend(labels.cpu().tolist())

    # 平均 loss
    avg_loss = total_loss / len(dataloader.dataset)

    # >>>> 自适应阈值 <<<<
    prob_np = np.array(prob_positive_list)
    gt_np   = np.array(gt)

    precisions, recalls, thresholds = precision_recall_curve(gt_np, prob_np)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    # thresholds 长度比 f1s 少 1，如果 best_idx 超出范围，就用 0.5
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # 用最佳阈值重新计算预测标签
    final_preds = (prob_np >= best_threshold).astype(int)

    # 计算各项指标
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
    """
    提取模型融合后的特征，利用 t-SNE 降维到2D，并绘制散点图（颜色代表类别）。
    如果提供了 save_path 参数，则保存图像到指定路径；否则调用 plt.show() 展示图像。
    """
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
        print(f"t-SNE图已保存到 {save_path}")
    else:
        plt.show()

