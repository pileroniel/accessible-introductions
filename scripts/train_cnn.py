"""
My CNN for garment-type classification: ResNet-50 pretrained on ImageNet, fine-tuned on my subset.
Same data, splits and metrics as scripts/train_baseline.py, so the two models compare directly.

Steps (each is a function below):
  1. Load my subset with ImageFolder (same class order as the baseline).
  2. Phase 1: freeze the backbone and train only my new head (a linear probe).
  3. Phase 2: unfreeze the top of the backbone and fine-tune it with a smaller learning rate.
  4. After every epoch I score VAL and keep the weights with the best val macro-F1 (early stopping).
  5. Evaluate ONLY those best weights on TEST (once).
  6. Write results, training curves, confusion matrix, misclassified examples and the checkpoint.

Run options (from the repo root, in the senior-project env):
    python scripts/train_cnn.py --sanity                    # overfit 64 images to check for bugs (~1 min)
    python scripts/train_cnn.py --data data/interim/garment_trial --head-epochs 2 --epochs 2 --reports reports/cnn_trial
    python scripts/train_cnn.py                             # full run

Run numbering:
    --reports is a base name. Each run gets the next free number after it, so with
    reports/cnn_full_1 already there, my next full run writes to reports/cnn_full_2.
    The checkpoint takes the same name (models/checkpoints/resnet50_cnn_full_2.pt),
    so every report folder matches exactly one saved model and no run overwrites another.

Outputs (for run n):
    reports/cnn_full_n/run_config.json     every setting and the exact command, written before training starts
    reports/cnn_full_n/results.md, history.csv, curves.png, confusion_matrix_test.png, misclassified_test.jpg
    models/checkpoints/resnet50_cnn_full_n.pt    my best weights + class names + preprocessing settings

LEARN:
  Transfer learning:     https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
  Mixed precision (AMP): https://pytorch.org/docs/stable/amp.html
  Label smoothing:       https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
"""
import argparse
import copy
import csv
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so I can import "src" when running scripts/...
# These imports have to come after the sys.path line above, so I tell the linters that's on purpose.
from src.classification.cnn import (CNN_PARAMS, build_model, freeze_batchnorm_stats,  # noqa: E402  # pylint: disable=wrong-import-position
                                    make_transform, save_checkpoint, set_trainable)
from src.classification.metrics import (load_class_to_group, misclassified_grid,  # noqa: E402  # pylint: disable=wrong-import-position
                                        plot_confusion, score)

SEED = 42


def seed_everything(seed=SEED):
    """Fixed seeds give me the same shuffles and augmentations each run (GPU maths can still differ slightly)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def next_run_dir(base: Path) -> Path:
    """I return the next numbered folder for this kind of run and create it.
    1. I look for siblings named <base name>_<number>, e.g. cnn_full_1 and cnn_full_2
    2. I take the highest number and add 1 (or start at 1 if there are none)
    I create the folder straight away, so a second run started at the same time can't take the same number."""
    base.parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(base.name)}_(\d+)$")
    taken = [int(m.group(1)) for d in base.parent.iterdir() if d.is_dir() and (m := pattern.match(d.name))]
    run_dir = base.parent / f"{base.name}_{max(taken, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


# 1. Loading data, augmenting during training, saving some useful window configs like persistent workers

def make_data(data_dir: Path, batch: int, workers: int, sanity: bool):
    """I build train/val/test loaders and return (loaders dict, class names, test image paths)."""
    for split in ("train", "val", "test"):
        if not (data_dir / split).is_dir():
            sys.exit(f"Missing {data_dir / split}. Build the subset first (scripts/build_garment_subset.py).")
    train_ds = ImageFolder(data_dir / "train", transform=make_transform(train=True))
    val_ds = ImageFolder(data_dir / "val", transform=make_transform(train=False))
    test_ds = ImageFolder(data_dir / "test", transform=make_transform(train=False))
   
    # ImageFolder sorts class folders alphabetically, the same order as my baseline, so label 3 means
    # the same class in every split and in both models' confusion matrices.
    if not train_ds.classes == val_ds.classes == test_ds.classes:
        sys.exit("Class folders differ between train, val and test. Rebuild the subset.")
  
    classes = train_ds.classes
    test_paths = [p for p, _ in test_ds.samples]

    if sanity:
        # 1. I take 64 training images with no augmentation
        # 2. I train and score on those same images; this setup purposefully reaches 100% acc
        plain = ImageFolder(data_dir / "train", transform=make_transform(train=False))
        idx = list(range(0, len(plain), max(1, len(plain) // 64)))[:64]
        train_ds = val_ds = test_ds = Subset(plain, idx)
        test_paths = [plain.samples[i][0] for i in idx]

    # persistent workers: on my Windows laptop, starting worker processes is slow, so I keep them alive between epochs
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch, shuffle=True, **kw),
        "val": DataLoader(val_ds, batch_size=batch * 2, shuffle=False, **kw),
        "test": DataLoader(test_ds, batch_size=batch * 2, shuffle=False, **kw),
    }
    return loaders, classes, test_paths


# 2. Training

def make_optimizer(model, lr_head, lr_backbone, weight_decay):
    """Two learning rates: my new head starts random and needs big steps, while the pretrained
    backbone is already good and only needs small adjustments."""
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc")]
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")]
    groups = [{"params": head, "lr": lr_head}]
    if backbone:  # empty in phase 1
        groups.append({"params": backbone, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    """I make one pass over the training images and return (mean loss, accuracy) on them.
    use_amp True means float16 maths (faster, less memory), False means full float32."""
    model.train()
    freeze_batchnorm_stats(model)
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # Mixed precision: most maths runs in float16, which roughly halves memory so ResNet-50 fits in my 4 GB virtual ram
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss): # if we get any NaNs, theres likely a problem with the floating point precision calculations
            raise RuntimeError(f"Training loss became {loss.item()}. Stopping instead of training on NaN. \n Rerun, and if it happens again, add --no-amp to train in float32.")
        
        scaler.scale(loss).backward()   # the scaler stops tiny float16 gradients from rounding to zero
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict_proba(model, loader, device, use_amp):
    """I return (class probabilities, true labels) for every image in a loader."""
    model.eval()  # turns off dropout and uses BatchNorm's stored averages
    probs, labels = [], []
    for images, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def run_phase(phase, epochs, model, loaders, args, device, class_to_group, best, history):
    """I train one phase and score VAL after every epoch. Whenever val macro-F1 beats my best so far,
    I keep a copy of the weights. That copy, not the last epoch, is what I test and save."""
    if epochs == 0:
        return
    n_trainable = set_trainable(model, phase, args.unfreeze)
    print(f"\nPhase {phase}: {n_trainable:,} trainable parameters")
    # a fresh optimiser each phase, because phase 2 trains weights phase 1's optimiser never saw
    optimizer = make_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # lr glides down to ~0
    
    # label smoothing: with ~16% wrong labels, I don't want my model 100% sure of any training label
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)  # only needed for float16

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, scaler, device, args.amp)
        scheduler.step()
        probs, y = predict_proba(model, loaders["val"], device, args.amp)
        m, _ = score(y, probs, class_to_group)
        secs = time.perf_counter() - t0
        history.append({"phase": phase, "epoch": epoch, "train_loss": round(tr_loss, 4),
                        "train_acc": round(tr_acc, 4), **{f"val_{k}": round(v, 4) for k, v in m.items()},
                        "seconds": round(secs)})
        improved = m["macro_f1"] > best["macro_f1"]
        if improved:
            # I copy the weights to the CPU so the copy doesn't use my limited GPU memory
            best.update(macro_f1=m["macro_f1"], phase=phase, epoch=epoch, val=m,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"  phase {phase} epoch {epoch:2d}  train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val acc {m['accuracy']:.3f} macro-F1 {m['macro_f1']:.3f}  ({secs:.0f}s){'  *best*' if improved else ''}")


# 3. Timing

@torch.no_grad()
def ms_per_image(model, device, size, n=30):
    """I time one image at a time (batch of 1), the way my app will call the model, after warm-up runs."""
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(5):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()  # GPU work is queued, so I wait for it before starting the clock
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n


# 4. Figures

def plot_curves(history, out_path: Path):
    """I plot train loss, and train vs val accuracy, across both phases. If train accuracy keeps rising
    while val accuracy falls, my model is memorising the training images (overfitting)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(1, len(history) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    a1.plot(x, [h["train_loss"] for h in history], marker="o")
    a1.set_title("Train loss")
    a2.plot(x, [h["train_acc"] for h in history], marker="o", label="train accuracy")
    a2.plot(x, [h["val_accuracy"] for h in history], marker="o", label="val accuracy")
    a2.plot(x, [h["val_macro_f1"] for h in history], marker="o", label="val macro-F1")
    a2.set_title("Accuracy")
    a2.legend()
    phase2 = next((i + 1 for i, h in enumerate(history) if h["phase"] == 2), None)
    for ax in (a1, a2):
        ax.set_xlabel("Epoch (both phases)")
        if phase2 and phase2 > 1:
            ax.axvline(phase2 - 0.5, color="grey", linestyle="--", linewidth=1)  # where fine-tuning starts
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/processed/garment"))
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "garment_classes.json")
    ap.add_argument("--reports", type=Path, default=Path("reports/cnn_full"),
                    help="base name; each run is numbered, e.g. reports/cnn_full_2")
    ap.add_argument("--model-out", type=Path, default=None,
                    help="default: models/checkpoints/resnet50_<run folder name>.pt")
    ap.add_argument("--head-epochs", type=int, default=3, help="phase 1: head only")
    ap.add_argument("--epochs", type=int, default=4, help="phase 2: fine-tuning")
    ap.add_argument("--unfreeze", choices=["layer4", "layer3", "all"], default="layer3")
    ap.add_argument("--batch", type=int, default=32, help="drop to 16 if I run out of GPU memory")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true",
                    help="train in float32 instead of float16; slower and uses more memory, but avoids NaN")
    ap.add_argument("--sanity", action="store_true", help="overfit 64 images to check my setup")
    args = ap.parse_args()
    if args.sanity:
        args.head_epochs, args.epochs, args.unfreeze = 0, 15, "all"
        args.label_smoothing, args.reports = 0.0, Path("reports/cnn_sanity")

    # float16 only works on the GPU, and --no-amp turns it off there too
    args.amp = torch.cuda.is_available() and not args.no_amp

    # I number this run and record its settings before training, so even a crashed run leaves a trace
    run_dir = next_run_dir(args.reports)
    args.model_out = args.model_out or Path("models/checkpoints") / f"resnet50_{run_dir.name}.pt"
    started = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_config = {"run": run_dir.name, "started": started, "command": " ".join(sys.argv),
                  **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "cnn_params": CNN_PARAMS}
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Run: {run_dir.name}  (reports in {run_dir})")

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False  # True picked fp16 kernels that gave NaN on my GTX 1650 in some runs
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU (this will be very slow)'}"
          f" | precision {'float16 (AMP)' if args.amp else 'float32'}")

    # 1. data
    loaders, classes, test_paths = make_data(args.data, args.batch, args.workers, args.sanity)
    class_to_group = load_class_to_group(args.config, classes)
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    print(f"Images: train {len(loaders['train'].dataset)} | val {len(loaders['val'].dataset)} | "
          f"test {len(loaders['test'].dataset)}")

    # 2-4. two phases, keeping the best val macro-F1 weights across both
    model = build_model(len(classes)).to(device)
    best, history = {"macro_f1": -1.0}, []
    run_phase(1, args.head_epochs, model, loaders, args, device, class_to_group, best, history)
    run_phase(2, args.epochs, model, loaders, args, device, class_to_group, best, history)
    model.load_state_dict(best["state"])
    print(f"\nBest on val: phase {best['phase']} epoch {best['epoch']} (macro-F1 {best['macro_f1']:.3f})")

    # 5. TEST, once, best weights only
    probs, y_te = predict_proba(model, loaders["test"], device, args.amp)
    test_m, test_pred = score(y_te, probs, class_to_group)
    print(f"TEST: accuracy {test_m['accuracy']:.3f} | macro-F1 {test_m['macro_f1']:.3f} | "
          f"top-2 {test_m['top2_accuracy']:.3f} | group-level {test_m['group_accuracy']:.3f}")
    gpu_ms = ms_per_image(model, device, CNN_PARAMS["img_size"]) if device.type == "cuda" else None
    cpu_ms = ms_per_image(copy.deepcopy(model).cpu(), torch.device("cpu"), CNN_PARAMS["img_size"], n=10)

    # 6. outputs, all into this run's numbered folder
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    plot_curves(history, run_dir / "curves.png")
    plot_confusion(y_te, test_pred, classes, run_dir / "confusion_matrix_test.png",
                   f"CNN (ResNet-50, fine-tuned) - test set - {run_dir.name}")
    n_shown = misclassified_grid(test_paths, y_te, test_pred, classes, run_dir / "misclassified_test.jpg")
    report_txt = classification_report(y_te, test_pred, target_names=classes, digits=3)

    pct = lambda v: f"{v * 100:.1f}%"
    md = [f"# CNN results ({run_dir.name}): ResNet-50 pretrained on ImageNet, fine-tuned", "",
          f"Run: `{run_dir.name}` | started {started} | checkpoint `{'none (sanity run)' if args.sanity else args.model_out}`",
          f"Command: `{run_config['command']}`", "",
          f"Data: `{args.data}` | classes: {len(classes)} | train {len(loaders['train'].dataset)} / "
          f"val {len(loaders['val'].dataset)} / test {len(loaders['test'].dataset)}",
          f"Settings: phase 1 {args.head_epochs} epochs (head only), phase 2 {args.epochs} epochs "
          f"(unfreeze {args.unfreeze}), batch {args.batch}, lr head {args.lr_head} / backbone {args.lr_backbone}, "
          f"weight decay {args.weight_decay}, label smoothing {args.label_smoothing}, "
          f"pad to square {CNN_PARAMS['pad_to_square']}, precision {'float16 (AMP)' if args.amp else 'float32'}", "",
          "## Training history (validation set)", "",
          "| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for h in history:
        mark = " **(best)**" if (h["phase"], h["epoch"]) == (best["phase"], best["epoch"]) else ""
        md.append(f"| {h['phase']} | {h['epoch']}{mark} | {h['train_loss']} | {pct(h['train_acc'])} | "
                  f"{pct(h['val_accuracy'])} | {pct(h['val_macro_f1'])} | {pct(h['val_top2_accuracy'])} | "
                  f"{pct(h['val_group_accuracy'])} | {h['seconds']} |")
    md += ["", f"## Test set (best weights only: phase {best['phase']}, epoch {best['epoch']})", "",
           "| Metric | Value |", "|---|---|",
           f"| Accuracy | {pct(test_m['accuracy'])} |",
           f"| Macro-F1 | {pct(test_m['macro_f1'])} |",
           f"| Top-2 accuracy | {pct(test_m['top2_accuracy'])} |",
           f"| Group-level accuracy | {pct(test_m['group_accuracy'])} |",
           f"| Prediction time, GPU | {f'{gpu_ms:.1f} ms / image' if gpu_ms else 'n/a'} |",
           f"| Prediction time, CPU | {cpu_ms:.1f} ms / image |", "",
           "## Per-class results (test)", "", "```", report_txt, "```", "",
           f"Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` ({n_shown} random test mistakes).", ""]
    (run_dir / "results.md").write_text("\n".join(md), encoding="utf-8")

    if not args.sanity:
        model.load_state_dict(best["state"])
        save_checkpoint(args.model_out, model, classes,
                        extra={"selected": {"phase": best["phase"], "epoch": best["epoch"]},
                               "val": best["val"], "test": test_m})
    print(f"\nWrote {run_dir / 'results.md'}, curves, confusion matrix, misclassified grid"
          f"{'' if args.sanity else f'; model saved to {args.model_out}'}")


if __name__ == "__main__":   # required on Windows, because DataLoader workers re-import this file
    main()
