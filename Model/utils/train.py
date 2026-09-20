import torch
from torch.cuda.amp import GradScaler
from torch.amp import autocast

scaler = GradScaler()

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

        with autocast(device_type='cuda', enabled=True):
            logits = model(inputs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        # Unscale before clipping so the threshold applies to real gradients.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(dataloader.dataset)
