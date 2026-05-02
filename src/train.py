"""Training loop with early stopping, CSV logging, and checkpointing."""
import os
import time
import copy
import csv
import numpy as np
import torch
import torch.nn as nn

from src.config import DEVICE, CHECKPOINTS_DIR, LOGS_DIR


def train_model(model, train_loader, val_loader, optimizer, num_epochs,
                patience=10, device=DEVICE, experiment_name="experiment",
                checkpoint_every=5, scheduler=None, use_amp=False, verbose=True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "lr": []}

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0

    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"{experiment_name}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])

    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        v_loss_sum, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                v_loss_sum += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                v_correct += preds.eq(labels).sum().item()
                v_total += labels.size(0)
        val_loss = v_loss_sum / v_total
        val_acc = v_correct / v_total

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, train_acc, val_acc, current_lr])

        if verbose:
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Train L:{train_loss:.4f} A:{train_acc:.4f} | "
                  f"Val L:{val_loss:.4f} A:{val_acc:.4f} | LR:{current_lr:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch} (best @ {best_epoch})")
                break

        if checkpoint_every and epoch % checkpoint_every == 0:
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            ckpt = os.path.join(CHECKPOINTS_DIR, f"{experiment_name}_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt)

    total_time = time.time() - t_start
    model.load_state_dict(best_model_wts)
    return model, history, best_epoch, total_time


def evaluate_model(model, test_loader, device=DEVICE):
    model.eval(); model = model.to(device)
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs))
