"""Shared helpers: build the data views, train one variant, evaluate it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .batch import WindowDataset
from .config import Config
from .data import load_dataset
from .evaluate import evaluate_split, predict_window
from .train import Trainer


def prepare(cfg: Config, data_path, device="cpu"):
    ds = load_dataset(data_path)
    wds_train = WindowDataset(ds.subset(ds.split == "train"), cfg, device=device)
    wds_val = WindowDataset(ds.subset(ds.split == "val"), cfg, device=device)
    wds_all = WindowDataset(ds, cfg, device=device)
    return ds, wds_train, wds_val, wds_all


def train_variant(cfg: Config, data_path, out_dir, device="cpu", verbose=True):
    ds, wds_train, wds_val, wds_all = prepare(cfg, data_path, device)
    trainer = Trainer(cfg, wds_train, wds_val, device=device, out_dir=out_dir)
    trainer.fit()
    if verbose:
        print(f"[train] finished, best val WAPE = {trainer.best:.3f}%")
    return trainer, wds_all


def evaluate_model(model, cfg: Config, wds_all: WindowDataset, splits=("val", "test1", "test2", "test3"),
                   device="cpu", verbose=True):
    summary = {}
    for sp in splits:
        mask = wds_all.split == sp
        if not mask.any():
            continue
        if verbose:
            print(f"[eval] {sp} (n={int(mask.sum())})")
        summary[sp] = evaluate_split(model, wds_all, mask, cfg, device, verbose=verbose)
    return summary


def save_summary(summary: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
