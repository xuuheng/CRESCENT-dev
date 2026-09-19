import torch
from torch.cuda.amp import GradScaler
from torch.amp import autocast

scaler = GradScaler()

# 在 train 外层初始化
scaler = GradScaler()

def train_one_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=0.9):
    model.train()
    total_loss = 0

    for batch in dataloader:
        if len(batch) == 3:
            inputs, labels, _ = batch
        else:
            inputs, labels = batch

        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device).float()

        optimizer.zero_grad()

        # 半精度前向
        with autocast(device_type='cuda', enabled=True):
            logits = model(inputs)
            loss = criterion(logits, labels)

        # 放大梯度并反向
        scaler.scale(loss).backward()

        # —— 关键信息：先 unscale，再裁剪，再 step ——
        scaler.unscale_(optimizer)  # 把梯度从 \"放大\" 状态还原
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # 更新参数 & 更新 scaler
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(dataloader.dataset)
