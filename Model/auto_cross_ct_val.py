import time

from matplotlib import font_manager
from tqdm import tqdm

import numpy as np
import os
import datetime
import copy
import pandas as pd
import torch
import torch.nn as nn
import random as _py_random
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Optional process-memory reporting.
try:
    import psutil
except Exception:
    psutil = None


from Models import ComplexMultiStreamCNN
from Models import ComplexMultiStreamCNN
from utils.Dataset import TSVDataset
from utils.train import train_one_epoch
from utils.evaluate import evaluate_detailed


output_dir = "../experimental/"

import json
import shutil
from typing import Dict, List, Tuple, Optional
from sklearn.manifold import TSNE

TIMES_TTF = os.path.expanduser("~/front/times.ttf")

if os.path.exists(TIMES_TTF):
    font_manager.fontManager.addfont(TIMES_TTF)

    try:
        prop = font_manager.FontProperties(fname=TIMES_TTF)
        print("Loaded font:", prop.get_name())
    except Exception as e:
        print("Font load failed:", e)

def _select_best_epochs(metrics_tsv: str) -> Dict[str, Optional[int]]:
    """Return the best 1-based epoch for each available validation metric."""
    if not os.path.isfile(metrics_tsv):
        return dict(auc=None, f1=None, valloss=None, avg=None)

    df = pd.read_csv(metrics_tsv, sep='\t')
    needs = {'val_auc':'auc', 'val_f1':'f1', 'val_loss':'valloss'}
    for col in needs:
        if col not in df.columns:
            needs[col] = None

    out = {'auc': None, 'f1': None, 'valloss': None, 'avg': None}
    if needs['val_auc']:
        e = int(df['epoch'].iloc[df['val_auc'].idxmax()])
        out['auc'] = e
    if needs['val_f1']:
        e = int(df['epoch'].iloc[df['val_f1'].idxmax()])
        out['f1'] = e
    if needs['val_loss']:
        e = int(df['epoch'].iloc[df['val_loss'].idxmin()])
        out['valloss'] = e
    if needs['val_auc'] and needs['val_f1']:
        avg_series = (df['val_auc'] + df['val_f1']) / 2.0
        e = int(df['epoch'].iloc[avg_series.idxmax()])
        out['avg'] = e
    return out


def _resolve_checkpoint_for_epoch(outs_dir: str, epoch: int) -> Optional[str]:
    """Return a saved checkpoint for an epoch, or ``None`` if unavailable."""
    if epoch is None:
        return None
    ckpt = os.path.join(outs_dir, f"model_epoch_{epoch:03d}.pth")
    return ckpt if os.path.isfile(ckpt) else None


def export_multi_best_models(session_dir: str,
                             criteria: Tuple[str, ...] = ('auc', 'f1', 'valloss', 'avg')) -> None:
    """Export best checkpoints by AUC, F1, validation loss, and mean score.

    F1 prefers ``best_model.pth``. Other criteria use saved epoch snapshots.
    Missing snapshots are recorded in ``export_warnings.txt``.
    """
    warn_lines = []
    for ct in sorted(os.listdir(session_dir)):
        outs_dir = os.path.join(session_dir, ct)
        if not os.path.isdir(outs_dir):
            continue

        metrics_tsv = os.path.join(outs_dir, "epoch_metrics.tsv")
        best_epochs = _select_best_epochs(metrics_tsv)

        if 'f1' in criteria:
            best_f1_path = os.path.join(outs_dir, "best_model.pth")
            if os.path.isfile(best_f1_path):
                shutil.copyfile(best_f1_path, os.path.join(outs_dir, "best_by_f1.pth"))
            else:
                ep = best_epochs.get('f1')
                src = _resolve_checkpoint_for_epoch(outs_dir, ep)
                if src:
                    shutil.copyfile(src, os.path.join(outs_dir, "best_by_f1.pth"))
                else:
                    warn_lines.append(f"[{ct}] best_by_f1 epoch={ep} checkpoint not found.")

        for key, fname in [('auc','best_by_auc.pth'),
                           ('valloss','best_by_valloss.pth'),
                           ('avg','best_by_avg.pth')]:
            if key not in criteria:
                continue
            ep = best_epochs.get(key)
            src = _resolve_checkpoint_for_epoch(outs_dir, ep)
            if src:
                shutil.copyfile(src, os.path.join(outs_dir, fname))
            else:
                warn_lines.append(f"[{ct}] {fname} epoch={ep} checkpoint not found.")

        with open(os.path.join(outs_dir, "best_model_selection.json"), "w") as jf:
            json.dump(best_epochs, jf, indent=2)

    if warn_lines:
        with open(os.path.join(session_dir, "export_warnings.txt"), "w") as wf:
            wf.write("\n".join(warn_lines))
        print("Some best epochs have no saved snapshot; see export_warnings.txt.")
    else:
        print("Finished exporting best models by metric.")


@torch.no_grad()
def _collect_logits(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return raw logits, labels, and filenames for one data loader."""
    model.eval()
    all_logits, all_labels, all_names = [], [], []
    for branches, labels, names in loader:
        # branches: list of [B, C, H, W] tensors
        branches = [b.to(device, non_blocking=True) for b in branches]
        logits = model(branches)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        elif logits.ndim != 1:
            logits = logits.view(logits.shape[0], -1).mean(1)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.numpy())
        all_names.extend(list(names))
    return np.concatenate(all_logits), np.concatenate(all_labels), all_names


def build_tsne_for_multi_best(session_dir: str,
                              data_dir: str,
                              mut: str,
                              target_cancer_types: Optional[List[str]] = None,
                              batch_size: int = 32,
                              perplexity: float = 30.0,
                              n_iter: int = 1000,
                              random_state: int = 42) -> None:
    """Build a t-SNE projection from the available best-model logits.

    Writes coordinates, parameters, and a scatter plot under each cancer
    project's output directory.
    """
    dataset = TSVDataset(data_dir)
    if len(dataset) == 0:
        print("No valid samples found."); return
    first_sample, _, _ = dataset[0]
    input_shapes = [(t.shape[1], t.shape[2]) for t in first_sample]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_types = sorted(set(fn.split('_')[0] for _, _, fn in dataset))
    cancer_types = [t for t in (target_cancer_types or all_types) if t in all_types]

    for ct in cancer_types:
        outs_dir = os.path.join(session_dir, ct)
        if not os.path.isdir(outs_dir):
            continue

        tag_to_fname = {
            'f1':      'best_by_f1.pth',
            'auc':     'best_by_auc.pth',
            'valloss': 'best_by_valloss.pth',
            'avg':     'best_by_avg.pth',
        }
        existing = {tag: os.path.join(outs_dir, fn)
                    for tag, fn in tag_to_fname.items()
                    if os.path.isfile(os.path.join(outs_dir, fn))}
        if not existing:
            print(f"[{ct}] No best_by_* models found; skipping t-SNE.")
            continue

        indices = list(range(len(dataset)))
        val_indices = [i for i in indices if dataset[i][2].split('_')[0] == ct]
        val_loader = DataLoader(
            Subset(dataset, val_indices), batch_size=batch_size, shuffle=False,
            collate_fn=multi_branch_collate, num_workers=1
        )

        matrix_list, tags = [], []
        labels_ref, names_ref = None, None
        for tag, path in existing.items():
            model = ComplexMultiStreamCNN(input_shapes, num_classes=1).to(device)
            state = torch.load(path, map_location=device)
            model.load_state_dict(state, strict=False)
            logits, labels, names = _collect_logits(model, val_loader, device)
            matrix_list.append(logits.reshape(-1, 1))
            tags.append(tag)
            if labels_ref is None:
                labels_ref, names_ref = labels, names

        X = np.concatenate(matrix_list, axis=1)  # [N, M]
        # t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                    init='pca', learning_rate='auto', random_state=random_state)
        coords = tsne.fit_transform(X)  # [N, 2]

        df_out = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'label': labels_ref,
            'file_name': names_ref,
        })
        for j, tag in enumerate(tags):
            df_out[f'logit_{tag}'] = X[:, j]

        df_out.to_csv(os.path.join(outs_dir, "tsne_coords.tsv"), sep='\t', index=False)
        with open(os.path.join(outs_dir, "tsne_params.json"), "w") as jf:
            json.dump({
                'models_used': tags,
                'perplexity': perplexity,
                'n_iter': n_iter,
                'random_state': random_state,
                'mut': mut,
                'val_cancer_type': ct,
                'N_samples': int(X.shape[0]),
                'M_models': int(X.shape[1]),
            }, jf, indent=2)

        plt.figure(figsize=(5,4))
        scatter = plt.scatter(df_out['x'], df_out['y'], c=df_out['label'], s=10, alpha=0.8)
        plt.title(f"t-SNE of best models — {ct}")
        plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2'); plt.grid(True, ls=':')
        plt.savefig(os.path.join(outs_dir, "tsne_scatter.png"), dpi=200)
        plt.close()
        print(f"[{ct}] Wrote t-SNE outputs to {outs_dir}")


def get_prefix(fn: str) -> str:
    name = os.path.splitext(fn)[0]
    return name.split('_aug_')[0]

def set_seed(seed: int = 42):
    _py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def moving_average(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode='valid')

def _bytes_to_mb(x):
    return float(x) / (1024.0 ** 2)

def gpu_name_total_mb(device_idx=0):
    if not torch.cuda.is_available():
        return ("CPU", None)
    props = torch.cuda.get_device_properties(device_idx)
    return (props.name, _bytes_to_mb(props.total_memory))

def gpu_mem_info_mb():
    """Return free and total CUDA memory in MiB when CUDA is available."""
    if not torch.cuda.is_available():
        return (None, None)
    try:
        free, total = torch.cuda.mem_get_info()
        return (_bytes_to_mb(free), _bytes_to_mb(total))
    except Exception:
        return (None, None)

def gpu_peak_alloc_reserved_mb(device_idx=0):
    if not torch.cuda.is_available():
        return (None, None)
    try:
        alloc = torch.cuda.max_memory_allocated(device_idx)
        reserv = torch.cuda.max_memory_reserved(device_idx)
        return (_bytes_to_mb(alloc), _bytes_to_mb(reserv))
    except Exception:
        return (None, None)

def cpu_rss_mb():
    if psutil is None:
        return None
    try:
        p = psutil.Process(os.getpid())
        return _bytes_to_mb(p.memory_info().rss)
    except Exception:
        return None

def multi_branch_collate(batch):
    if not batch:
        return None
    num_branches = len(batch[0][0])
    branch_batches = []
    for j in range(num_branches):
        branch_batches.append(torch.stack([sample[0][j] for sample in batch], dim=0))
    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    file_names = [sample[2] for sample in batch]
    return branch_batches, labels, file_names


def auto_cross_val(data_dir,
                   mut="del",
                   target_cancer_types=None,
                   seed=42):
    """Run leave-one-cancer-project-out training and validation."""
    set_seed(seed)
    dataset = TSVDataset(data_dir)
    if len(dataset) == 0:
        print("No valid samples found.")
        return

    first_sample, _, _ = dataset[0]
    input_shapes = [(t.shape[1], t.shape[2]) for t in first_sample]
    print("Input branch shapes:", input_shapes)

    all_types = set(fn.split('_')[0] for _, _, fn in dataset)
    if target_cancer_types is not None:
        cancer_types = [t for t in target_cancer_types if t in all_types]
        missing = set(target_cancer_types) - set(cancer_types)
        if missing:
            print(f"Warning: requested cancer types not found and ignored: {missing}")
    else:
        cancer_types = sorted(all_types)
    print("Cancer types to run:", cancer_types)

    num_epochs, batch_size, learning_rate = 1, 8, 1e-5  # amp:lr=1e-5
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

    save_last_k = 30

    base_dir = os.path.join(output_dir, mut)
    session_dir = os.path.join(
        base_dir,
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(seed)
    )
    os.makedirs(session_dir, exist_ok=True)

    gpu_name, gpu_total_mb = gpu_name_total_mb(0)
    with open(os.path.join(session_dir, "hyperparameters.txt"), "w") as f:
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"device: {device}\n")
        f.write("scheduler: ReduceLROnPlateau\n")
        f.write(f"torch: {torch.__version__}\n")
        f.write(f"cuda_available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"gpu_name: {gpu_name}\n")
            f.write(f"gpu_total_mb: {gpu_total_mb:.1f}\n")
            try:
                f.write(f"cuda_version: {torch.version.cuda}\n")
            except Exception:
                pass

    summary_tsv = os.path.join(session_dir, "summary_metrics.tsv")
    header = [
        'cancer_type','best_epoch','best_loss','best_auc','best_f1',
        'train_size','val_size'
    ]
    if not os.path.exists(summary_tsv):
        with open(summary_tsv, 'w') as f:
            f.write("\t".join(header) + "\n")

    for val_cancer_type in cancer_types:
        print(f"\n===== Held-out validation type: {val_cancer_type} =====")
        indices = list(range(len(dataset)))
        train_indices = [i for i in indices if dataset[i][2].split('_')[0] != val_cancer_type]
        val_indices   = [i for i in indices if dataset[i][2].split('_')[0] == val_cancer_type]

        print(f"Training size: {len(train_indices)}, validation size: {len(val_indices)}")

        train_loader = DataLoader(
            Subset(dataset, train_indices), batch_size=batch_size,
            shuffle=True, collate_fn=multi_branch_collate,
            num_workers=1, persistent_workers=True, pin_memory=True, prefetch_factor=2
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices), batch_size=batch_size,
            shuffle=False, collate_fn=multi_branch_collate,
            num_workers=1
        )

        model = ComplexMultiStreamCNN(input_shapes, num_classes=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
        criterion = nn.BCEWithLogitsLoss()

        train_loss_hist, train_acc_hist = [], []
        val_loss_hist, val_acc_hist, val_auc_hist, val_f1_hist = [], [], [], []

        # Select by validation F1, using AUC as the tie-breaker.
        best_f1, best_auc_tiebreak = -float('inf'), -float('inf')
        best_epoch, best_state = -1, None

        outs_dir = os.path.join(session_dir, val_cancer_type)
        os.makedirs(outs_dir, exist_ok=True)

        lastk_manifest = os.path.join(outs_dir, "last20_models_manifest.txt")
        if os.path.exists(lastk_manifest):
            os.remove(lastk_manifest)

        loss_fig, loss_ax = plt.subplots(figsize=(6, 4))
        acc_fig, acc_ax = plt.subplots(figsize=(6, 4))

        timings_tsv = os.path.join(outs_dir, "timings.tsv")
        with open(timings_tsv, "w") as f:
            f.write(
                "epoch\tseconds\tlr\tgpu_alloc_peak_mb\tgpu_reserved_peak_mb\t"
                "gpu_free_begin_mb\tgpu_free_end_mb\tcpu_rss_begin_mb\tcpu_rss_end_mb\n"
            )

        pbar = tqdm(range(1, num_epochs + 1), desc=f"[{val_cancer_type}] Epochs", unit="epoch", leave=True)
        for epoch in pbar:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            gpu_free_begin_mb, _gpu_total_begin_mb = gpu_mem_info_mb()
            cpu_rss_begin_mb = cpu_rss_mb()

            ep_t0 = time.time()

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, *_ = \
                evaluate_detailed(model, val_loader, criterion, device)
            _, train_acc, _, _, _, _, *_ = \
                evaluate_detailed(model, train_loader, criterion, device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ep_secs = time.time() - ep_t0
            alloc_peak_mb, reserv_peak_mb = gpu_peak_alloc_reserved_mb()
            gpu_free_end_mb, _gpu_total_end_mb = gpu_mem_info_mb()
            cpu_rss_end_mb = cpu_rss_mb()
            lr_now = optimizer.param_groups[0]['lr']

            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            val_loss_hist.append(val_loss)
            val_acc_hist.append(val_acc)
            val_auc_hist.append(val_auc)
            val_f1_hist.append(val_f1)

            pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss":   f"{val_loss:.4f}",
                "val_auc":    f"{val_auc:.4f}",
                "val_f1":     f"{val_f1:.4f}",
                "lr":         f"{lr_now:.2e}",
                "sec":        f"{ep_secs:.1f}"
            })

            print(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}"
                f" | LR: {lr_now:.2e} | Time: {ep_secs:.1f}s"
            )

            with open(timings_tsv, "a") as f:
                f.write(
                    f"{epoch}\t{ep_secs:.3f}\t{lr_now:.6e}\t"
                    f"{(alloc_peak_mb if alloc_peak_mb is not None else 'NA')}\t"
                    f"{(reserv_peak_mb if reserv_peak_mb is not None else 'NA')}\t"
                    f"{(gpu_free_begin_mb if gpu_free_begin_mb is not None else 'NA')}\t"
                    f"{(gpu_free_end_mb if gpu_free_end_mb is not None else 'NA')}\t"
                    f"{(cpu_rss_begin_mb if cpu_rss_begin_mb is not None else 'NA')}\t"
                    f"{(cpu_rss_end_mb if cpu_rss_end_mb is not None else 'NA')}\n"
                )

            scheduler.step(val_loss)

            current_val_filenames = []
            for batch in val_loader:
                _, _, names = batch
                current_val_filenames.extend(names)
            with open(os.path.join(outs_dir, f"val_filenames_epoch_{epoch}.txt"), "w") as vf:
                vf.write("\n".join(current_val_filenames))

            improved = (val_f1 > best_f1) or (np.isclose(val_f1, best_f1) and (val_auc > best_auc_tiebreak))
            if improved:
                best_f1, best_auc_tiebreak = val_f1, val_auc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            loss_ax.clear()
            loss_ax.plot(range(1, epoch+1), train_loss_hist, label='Training Loss')
            loss_ax.plot(range(1, epoch+1), val_loss_hist,   label='Validation Loss')
            loss_ax.set_xlabel('Epoch'); loss_ax.set_ylabel('Loss'); loss_ax.legend(); loss_ax.grid(True)
            loss_fig.savefig(os.path.join(outs_dir, "loss_curve_raw.png"), dpi=150)

            acc_ax.clear()
            acc_ax.plot(range(1, epoch+1), train_acc_hist, label='Training Acc')
            acc_ax.plot(range(1, epoch+1), val_acc_hist,   label='Validation Acc')
            acc_ax.set_xlabel('Epoch'); acc_ax.set_ylabel('Accuracy'); acc_ax.legend(); acc_ax.grid(True)
            acc_fig.savefig(os.path.join(outs_dir, "accuracy_curve_raw.png"), dpi=150)

            if epoch >= (num_epochs - save_last_k + 1):
                ep_ckpt = os.path.join(outs_dir, f"model_epoch_{epoch:03d}.pth")
                torch.save(model.state_dict(), ep_ckpt)
                with open(lastk_manifest, "a") as mf:
                    mf.write(f"epoch={epoch}\t{os.path.basename(ep_ckpt)}\n")

        epoch_tsv = os.path.join(outs_dir, "epoch_metrics.tsv")
        with open(epoch_tsv, 'w') as f:
            f.write("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\tval_auc\tval_f1\n")
        for e in range(num_epochs):
            with open(epoch_tsv, 'a') as f:
                f.write(
                    f"{e+1}\t{train_loss_hist[e]:.6f}\t{train_acc_hist[e]:.6f}"
                    f"\t{val_loss_hist[e]:.6f}\t{val_acc_hist[e]:.6f}"
                    f"\t{val_auc_hist[e]:.6f}\t{val_f1_hist[e]:.6f}\n"
                )

        window = 5
        train_s = moving_average(train_loss_hist, window)
        val_s   = moving_average(val_loss_hist, window)
        epochs_s = list(range(1 + (window-1)//2, num_epochs - (window-1)//2 + 1))
        plt.figure()
        plt.plot(epochs_s, train_s, label=f'Training Loss (MA{window})')
        plt.plot(epochs_s, val_s,   label=f'Validation Loss (MA{window})')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outs_dir, "loss_curve_smoothed.png"))
        plt.close()

        auc_s = moving_average(val_auc_hist, window)
        plt.figure()
        plt.plot(epochs_s, auc_s, marker='o', label=f'Val AUC (MA{window})')
        plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outs_dir, "auc_curve_smoothed.png")); plt.close()

        acc_train_s = moving_average(train_acc_hist, window)
        acc_val_s   = moving_average(val_acc_hist, window)
        plt.figure()
        plt.plot(epochs_s, acc_train_s, label=f'Train Acc (MA{window})')
        plt.plot(epochs_s, acc_val_s,   label=f'Val   Acc (MA{window})')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outs_dir, "accuracy_curve_smoothed.png")); plt.close()

        assert best_state is not None, "No best state was produced; check the training and validation flow."
        torch.save(best_state, os.path.join(outs_dir, "best_model.pth"))
        model.load_state_dict(best_state)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_cm, gt, probs, _ = \
            evaluate_detailed(model, val_loader, criterion, device)
        print(f"\n*** Validation {val_cancer_type} (best F1 at epoch {best_epoch}) ***")
        print(
            f"Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f},"
            f" Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}"
        )

        record = [
            val_cancer_type,
            str(best_epoch),
            f"{val_loss_hist[best_epoch-1]:.6f}",
            f"{val_auc_hist[best_epoch-1]:.6f}",
            f"{val_f1_hist[best_epoch-1]:.6f}",
            str(len(train_indices)),
            str(len(val_indices))
        ]
        with open(summary_tsv, 'a') as f:
            f.write("\t".join(record) + "\n")

        fpr, tpr, thresholds = roc_curve(gt, probs)
        df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        df.to_csv(os.path.join(outs_dir, 'roc.tsv'), sep='\t', index=False)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC (AUC={val_auc:.4f})')
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outs_dir, "roc_val.png"))
        plt.close()

        with open(timings_tsv, "a") as f:
            f.write("TOTAL\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n")

    return session_dir



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz

def plot_metrics_and_roc(root_dir,
                         cancer_types,
                         metrics_filename='epoch_metrics.tsv',
                         roc_filename='roc.tsv',
                         save_path=None):
    """Plot train/validation loss and ROC panels for each cancer type."""
    plt.rcParams.update({
        'font.size':       12,
        'axes.titlesize':  14,
        'axes.labelsize':  12,
        'figure.dpi':      200,
        'lines.linewidth': 2,
        'font.family':     'serif',
        'font.serif':      ['Times New Roman', 'Times'],
    })

    letters = [chr(i) for i in range(ord('a'), ord('z')+1)]
    label_counter = 0

    n = len(cancer_types)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.subplots_adjust(left=0.12, top=0.92, wspace=0.3, hspace=0.4)

    for col, ct in enumerate(cancer_types):
        ax0 = axes[0, col]
        ax0.text(-0.05, 1.02, f'({letters[label_counter]})',
                 transform=ax0.transAxes,
                 fontsize='medium', fontweight='bold',
                 va='bottom', ha='left')
        label_counter += 1

        path0 = os.path.join(root_dir, ct, metrics_filename)
        if os.path.isfile(path0):
            df = pd.read_csv(path0, sep='\t')
            ax0.plot(df['epoch'], df['train_loss'], label='train_loss', linestyle='-')
            ax0.plot(df['epoch'], df['val_loss'],   label='val_loss',   linestyle='--')
            ax0.set_ylabel('Loss')
            ax0.legend(fontsize='small')
            ax0.grid(True, linestyle=':', alpha=0.6)
        else:
            ax0.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax0.set_title(f'{ct} — Loss')

        ax1 = axes[1, col]
        ax1.text(-0.05, 1.02, f'({letters[label_counter]})',
                 transform=ax1.transAxes,
                 fontsize='medium', fontweight='bold',
                 va='bottom', ha='left')
        label_counter += 1

        path1 = os.path.join(root_dir, ct, roc_filename)
        if os.path.isfile(path1):
            df_roc = pd.read_csv(path1, sep='\t')
            fpr, tpr = df_roc['fpr'].values, df_roc['tpr'].values

            ax1.plot(fpr, tpr, label='ROC')
            ax1.plot([0, 1], [0, 1], '--', color='gray', label='random')

            auc = trapz(tpr, fpr)
            ax1.legend(loc='lower right', fontsize='small', frameon=True)

            ax1.text(0.95, 0.2, f'AUC = {auc:.3f}',
                     transform=ax1.transAxes,
                     fontsize='small',
                     ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor='black',
                               alpha=0.8))

            ax1.set_xlabel('FPR')
            ax1.set_ylabel('TPR')
            ax1.grid(True, linestyle=':', alpha=0.6)
        else:
            ax1.text(0.5, 0.5, 'No roc.tsv', ha='center', va='center')
        ax1.set_title(f'{ct} — ROC')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved combined plot to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    k_runs = 5
    # cancer_list = ['LAML','GBM','UCEC','SARC','BRCA','BLCA','LUSC','HNSC','LUAD','ESCA','CESC','KICH','KIRC','KIRP','COAD','ACC','LGG','LIHC','CHOL','DLBC']
    # cancer_list = ['LAML','GBM','UCEC','SARC','BRCA','BLCA','LUSC','HNSC','LUAD','ESCA','KIRC','ACC', 'LGG', 'LIHC']
    # cancer_list = ['BLCA','BRCA','GBM','UCEC','HNSC']
    cancer_list = ['LIHC']

    # cancer_list = ['ACC', 'DLBC', 'LGG', 'LIHC']
    # cancer_list = ['ACC','CHOL','DLBC', 'LGG', 'LIHC']
    session_dirs = []
    mut_type = "amp"

    seeds = [530512]
    for se in seeds:
        print(f"\n===== Run with seed {se} =====")
        if mut_type == "amp":
            out_dir = auto_cross_val(
                # data_dir="../GeneratedSamples_amp_compressed_XR_matrix_norm",
                data_dir="../GeneratedSamples_amp_compressed_baseline_improved_0908",

                # data_dir=f"../GeneratedSamples_{mut_type}_compressed_baseline_improved__100_/",
                mut=mut_type,
                target_cancer_types=cancer_list,
                seed=int(se),
            )
        elif mut_type == "del":
            out_dir = auto_cross_val(
                data_dir=f"../GeneratedSamples_{mut_type}_compressed/",
                mut=mut_type,
                target_cancer_types=cancer_list,
                seed=int(se),
            )
        if out_dir is None:
            continue
        combined_png = os.path.join(output_dir, "combined_metrics_and_roc.png")
        session_dirs.append(out_dir)
        with open(os.path.join(out_dir, "hyperparameters.txt"), 'a') as f:
            f.write(f"seed: {se}\n")
    print("All runs completed.")

    for sd in session_dirs:
        print(f"\n=== Exporting best models by metric @ {sd} ===")
        export_multi_best_models(sd, criteria=('auc', 'f1', 'valloss', 'avg'))

        print(f"=== Building multi-model t-SNE @ {sd} ===")
        build_tsne_for_multi_best(
            session_dir=sd,
            data_dir=f"../GeneratedSamples_{mut_type}_compressed_baseline_improved_5120/",
            mut=mut_type,
            target_cancer_types=cancer_list,
            batch_size=32, perplexity=30, n_iter=1000, random_state=42
        )
