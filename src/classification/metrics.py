"""
My evaluation code, shared by every garment model (the HOG baseline and the CNN),
so all of them are scored and plotted in exactly the same way.

LEARN:
  Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html  (accuracy, macro-F1, top-k)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score

SEED = 42

# TODO - remove the conditional because we have a config
def load_class_to_group(config_path: Path, classes):
    """I look up each class's group (e.g. blazer, jacket and coat are all outerwear) in my class config.
    If the config is missing, every class is its own group, so group accuracy equals plain accuracy.x"""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))["classes"] if Path(config_path).exists() else {}
    return [cfg.get(c, {}).get("group", c) for c in classes]


def score(y, scores, class_to_group):
    """I score any model from its true labels and per-class scores (probabilities, or SVM decision values).
    Returns (metrics dict, predicted labels). Macro-F1 weighs every class equally."""
    scores = np.asarray(scores)
    pred = scores.argmax(axis=1)
    groups = np.array(class_to_group)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        # counts as right if the true class is in my model's two highest-scoring classes
        "top2_accuracy": float(top_k_accuracy_score(y, scores, k=2, labels=range(scores.shape[1]))),
        # would the safer group-level phrase ("a jacket" for blazer/jacket/coat) have been right?
        "group_accuracy": float(np.mean(groups[y] == groups[pred])),
    }, pred


# these functions were reused and copied from train_baseline.py

def plot_confusion(y, pred, classes, out_path: Path, title: str):
    """I save a confusion matrix where each row shows where the images of one true class ended up."""
    import matplotlib
    matplotlib.use("Agg")             # draw straight to a file, no window needed
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    rate = cm / cm.sum(axis=1, keepdims=True).clip(min=1)          # row-normalised: recall per true class
    # one-hue sequential ramp, from the surface colour to dark blue
    cmap = LinearSegmentedColormap.from_list(
        "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
    n = len(classes)
    for i in range(n):
        for j in range(n):
            if rate[i, j] >= 0.01:   # label only non-trivial cells
                # white text on dark cells, near-black text on light cells
                ax.text(j, i, f"{rate[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if rate[i, j] > 0.5 else "#1f1f1e")
    ax.set_xticks(range(n), classes, rotation=45, ha="right", fontsize=9, color="#4a4a47")
    ax.set_yticks(range(n), classes, fontsize=9, color="#4a4a47")
    # minor ticks sit on the cell borders, so the grid below draws lines between cells
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)          # 2px surface gap between cells
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Predicted class", color="#4a4a47")
    ax.set_ylabel("True class", color="#4a4a47")
    ax.set_title(title + "\n(% of each true class; diagonal = correct)", fontsize=11, color="#1f1f1e")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def misclassified_grid(paths, y, pred, classes, out_path: Path, n=40, tile=150, seed=SEED):
    """I save a grid of up to n random test mistakes, each captioned with its true and predicted class,
    so I can see which errors are my model's fault and which are wrong labels. Returns how many I drew."""
    wrong = np.flatnonzero(y != pred)
    if len(wrong) == 0:
        return 0
    rng = np.random.default_rng(seed)
    pick = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
    # 8 tiles per row, each with a 30 px caption strip underneath for the two text lines
    cols = 8
    rows = int(np.ceil(len(pick) / cols))
    caption = 30
    grid = Image.new("RGB", (cols * tile, rows * (tile + caption)), "white")
    draw = ImageDraw.Draw(grid)
    for k, idx in enumerate(pick):
        r, c = divmod(k, cols)
        with Image.open(paths[idx]) as im:
            im = im.convert("RGB")
            im.thumbnail((tile, tile))
            grid.paste(im, (c * tile, r * (tile + caption)))
        y0 = r * (tile + caption) + tile + 2           # top of the caption strip under this tile
        draw.text((c * tile + 3, y0), f"true: {classes[y[idx]]}", fill=(30, 30, 30))
        draw.text((c * tile + 3, y0 + 13), f"pred: {classes[pred[idx]]}", fill=(180, 30, 30))
    grid.save(out_path, quality=90)
    return len(pick)

