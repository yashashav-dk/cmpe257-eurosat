"""Stratified loaders for CNN and classical pipelines."""
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    IMG_SIZE, BATCH_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, EUROSAT_MEAN, EUROSAT_STD,
    AUGMENTATION_CONFIG,
)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_base_transforms(img_size=IMG_SIZE, normalize="eurosat"):
    mean = IMAGENET_MEAN if normalize == "imagenet" else EUROSAT_MEAN
    std = IMAGENET_STD if normalize == "imagenet" else EUROSAT_STD
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_augmentation_transforms(img_size=IMG_SIZE, normalize="eurosat"):
    cfg = AUGMENTATION_CONFIG
    mean = IMAGENET_MEAN if normalize == "imagenet" else EUROSAT_MEAN
    std = IMAGENET_STD if normalize == "imagenet" else EUROSAT_STD
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=cfg["rotation_degrees"]),
        transforms.ColorJitter(**cfg["color_jitter"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _get_stratified_indices(dataset, seed=SEED):
    targets = list(dataset.targets)
    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_RATIO, stratify=targets, random_state=seed,
    )
    train_val_targets = [targets[i] for i in train_val_idx]
    relative_val_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_ratio,
        stratify=train_val_targets, random_state=seed,
    )
    return train_idx, val_idx, test_idx


def get_datasets(img_size=IMG_SIZE, normalize="eurosat", augment_train=False, seed=SEED):
    base_tf = get_base_transforms(img_size, normalize)
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=base_tf)
    train_idx, val_idx, test_idx = _get_stratified_indices(full_dataset, seed)

    if augment_train:
        aug_tf = get_augmentation_transforms(img_size, normalize)
        train_full = datasets.ImageFolder(root=DATA_DIR, transform=aug_tf)
        train_dataset = Subset(train_full, train_idx)
    else:
        train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(img_size=IMG_SIZE, normalize="eurosat", augment_train=False,
                    batch_size=BATCH_SIZE, seed=SEED, num_workers=2):
    train_ds, val_ds, test_ds = get_datasets(img_size, normalize, augment_train, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_numpy_splits(seed=SEED):
    raw_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=raw_tf)
    train_idx, val_idx, test_idx = _get_stratified_indices(full_dataset, seed)

    def _extract(indices):
        images, labels = [], []
        for i in indices:
            img, lbl = full_dataset[i]
            images.append(img.permute(1, 2, 0).numpy())
            labels.append(lbl)
        return np.array(images), np.array(labels)

    X_tr_img, y_tr = _extract(train_idx)
    X_val_img, y_val = _extract(val_idx)
    X_te_img, y_te = _extract(test_idx)
    X_tr = X_tr_img.reshape(len(X_tr_img), -1)
    X_val = X_val_img.reshape(len(X_val_img), -1)
    X_te = X_te_img.reshape(len(X_te_img), -1)
    return (X_tr, y_tr, X_val, y_val, X_te, y_te,
            X_tr_img, X_val_img, X_te_img)
