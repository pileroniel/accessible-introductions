"""
My non-deep-learning BASELINE for garment-type classification:
    I feed HOG + colour histogram features into multinomial logistic regression or a linear SVM.

Steps (each is a function below):
  1. Load the cleaned subset (ImageFolder layout from build_garment_subset.py).
  2. Extract features once per image and cache them (data/interim/baseline_features/).
  3. Train logistic regression and a linear SVM on TRAIN, each with a few values of C.
  4. Pick the single best one by macro-F1 on VAL.
  5. Evaluate ONLY that model on TEST (once).
  6. Write a results table, confusion matrix, misclassified examples and the saved model.

Run options
    python scripts/train_baseline.py              # full run (~10-20 min on my CPU)
    python scripts/train_baseline.py --quick      # fewer settings, for a first try (~5 min)
    python scripts/train_baseline.py --data data/interim/garment_trial --quick   # tiny stest on the small interim data

Outputs:
    reports/baseline/results.md                 will use this for my midterm report
    reports/baseline/results.csv
    reports/baseline/confusion_matrix_test.png
    reports/baseline/misclassified_test.jpg     model mistakes vs. label mistakes
    models/checkpoints/baseline_hog.joblib      my saved best model + class names + feature settings

Sources:
  Pipelines & scaling:  https://scikit-learn.org/stable/modules/compose.html
  Model selection:      https://scikit-learn.org/stable/modules/cross_validation.html  (why val != test)
  Metrics:              https://scikit-learn.org/stable/modules/model_evaluation.html  (accuracy, macro-F1, top-k)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, dump
from PIL import Image, ImageDraw
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, top_k_accuracy_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so I can import "src" when running scripts/...
# This import has to come after the sys.path line above, so I tell the linters that's on purpose.
from src.classification.features import FEATURE_PARAMS, extract  # noqa: E402  # pylint: disable=wrong-import-position

SEED = 42


# 1. Data

def load_split(data_dir: Path, split: str, classes=None):
    """I read the ImageFolder layout data_dir/<split>/<class>/*.jpg and return (paths, integer labels, class names).
    I pass the train class list in for val and test so label 3 means the same class in every split."""
    split_dir = data_dir / split
    if classes is None:
        classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    paths, labels = [], []
    for i, c in enumerate(classes):
        files = sorted((split_dir / c).glob("*.jpg"))
        paths += files
        labels += [i] * len(files)
    return paths, np.array(labels), classes


# 2. Features (cached)

def get_features(paths, split, cache_dir: Path, jobs: int, recompute: bool):
    """I return (feature matrix, ms per image to extract) for one split, reusing my cache when I can.
    Extracting features is the slow part, so I only redo it when the images or FEATURE_PARAMS change."""
    cache = cache_dir / f"{split}.npz"
    # The cache key is the feature settings plus the exact list of image paths. If either changes,
    # the key stops matching and I extract again instead of silently using stale features.
    key = json.dumps({"params": FEATURE_PARAMS, "paths": [str(p) for p in paths]}, sort_keys=True)
    if cache.exists() and not recompute:
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            print(f"  {split}: loaded cached features {z['X'].shape}")
            return z["X"], float(z["ms_per_image"])
    print(f"  {split}: extracting features for {len(paths)} images ...")
    t0 = time.perf_counter()
    X = []
    # I work in chunks of 2,000 images so I can print progress; each chunk runs on all my CPU cores.
    chunk = 2000
    for start in range(0, len(paths), chunk):
        X += Parallel(n_jobs=jobs)(delayed(extract)(p) for p in paths[start:start + chunk])
        print(f"    {min(start + chunk, len(paths))}/{len(paths)}")
    X = np.stack(X)
    ms = (time.perf_counter() - t0) * 1000 / len(paths)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, key=np.array(key), ms_per_image=ms)
    return X, ms


# 3. Candidate models

def candidates(quick: bool):
    """(model family, setting description, unfitted estimator). I compare the settings on VAL.

    I use two linear classifiers, the classic partners for HOG features, plus a majority-class floor:
      - Multinomial logistic regression: one model over all 10 classes (softmax); gives probabilities,
        which my app needs for its confidence fallback ("a blazer" vs the safer "a jacket").
      - Linear SVM: maximum-margin, one-vs-rest (one yes/no model per class); no probabilities.
    C controls regularisation: smaller C means a simpler model that is less likely to overfit.

    Classifiers I considered for thousands of HOG features, and why I kept only the first two:
      Logistic regression  Good. Fast, and outputs probabilities, which my confidence fallback needs.
                           This is why I chose it.
      Linear SVM           Good. The classic partner for HOG (Dalal & Triggs used it), with accuracy
                           similar to logistic regression. It doesn't produce probabilities directly.
      RBF (kernel) SVM     Often more accurate, but slow. Training time grows sharply with dataset size,
                           so on 15,000 images it could take a long time. I could try it on a subsample.
      Random forest        Usually weaker here. Excellent for tables with dozens of meaningful columns,
                           but with thousands of correlated HOG numbers each tree only sees a few of them
                           at a time.
      KNN                  Weakest. In thousands of dimensions, distances between images stop being
                           meaningful (the "curse of dimensionality"). It's also slow at prediction, since
                           every test image is compared with all 15,000 training images. Better with PCA first.
    """
    lr_C = [0.01] if quick else [0.0001, 0.001, 0.01]
    svm_C = [0.001] if quick else [0.0001, 0.001, 0.01]
    # The majority-class model always guesses the most common class; any real model has to beat it.
    out = [("Majority class", "-", DummyClassifier(strategy="most_frequent"))]
    # StandardScaler first in each pipeline: HOG and colour numbers are on different scales, and both
    # linear models train better when every feature has mean 0 and variance 1. Because the scaler is
    # inside the pipeline, it learns its means from TRAIN only and is saved together with the model.
    for C in lr_C:
        # with the default lbfgs solver, scikit-learn fits a multinomial (softmax) model for >2 classes
        out.append(("Logistic regression", f"C={C}",
                    make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))))
    for C in svm_C:
        out.append(("Linear SVM", f"C={C}",
                    make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=5000, random_state=SEED))))
    return out


def scores_of(model, X):
    """Per-class scores for top-k accuracy (probabilities if available, else SVM decision values)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)


def evaluate(model, X, y, n_classes, class_to_group):
    """I score one fitted model on one split and return (metrics dict, predictions).
    Macro-F1 averages over classes equally, so a weak rare class can't hide behind a strong common one."""
    t0 = time.perf_counter()
    pred = model.predict(X)
    ms = (time.perf_counter() - t0) * 1000 / len(X)
    groups = np.array(class_to_group)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        # counts as right if the true class is in my model's two highest-scoring classes
        "top2_accuracy": top_k_accuracy_score(y, scores_of(model, X), k=2, labels=range(n_classes)),
        # would the safer group-level phrase ("a jacket" for blazer/jacket/coat) have been right?
        "group_accuracy": float(np.mean(groups[y] == groups[pred])),
        "predict_ms_per_image": ms,
    }, pred


# 6. Figures

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


# Main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/processed/garment"))
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "garment_classes.json")
    ap.add_argument("--reports", type=Path, default=Path("reports/baseline"))
    ap.add_argument("--model-out", type=Path, default=Path("models/checkpoints/baseline_hog.joblib"))
    ap.add_argument("--cache", type=Path, default=None, help="feature cache folder (default: data/interim/baseline_features/<data name>)")
    ap.add_argument("--quick", action="store_true", help="one setting per model family")
    ap.add_argument("--jobs", type=int, default=-1, help="CPU cores for feature extraction (-1 = all)")
    ap.add_argument("--recompute", action="store_true", help="ignore cached features")
    args = ap.parse_args()
    # one cache folder per dataset, so the trial subset and the full subset never overwrite each other
    cache_dir = args.cache or Path("data/interim/baseline_features") / args.data.name

    # 1. data
    tr_paths, y_tr, classes = load_split(args.data, "train")
    va_paths, y_va, _ = load_split(args.data, "val", classes)
    te_paths, y_te, _ = load_split(args.data, "test", classes)
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    print(f"Images: train {len(y_tr)} | val {len(y_va)} | test {len(y_te)}")
    # Each class's group (e.g. blazer, jacket and coat are all in "jacket") comes from my class config.
    # If the config is missing, every class is its own group, so group accuracy equals plain accuracy.
    groups_cfg = json.loads(args.config.read_text(encoding="utf-8"))["classes"] if args.config.exists() else {}
    class_to_group = [groups_cfg.get(c, {}).get("group", c) for c in classes]

    # 2. features
    print("\nFeatures:")
    X_tr, ms_feat = get_features(tr_paths, "train", cache_dir, args.jobs, args.recompute)
    X_va, _ = get_features(va_paths, "val", cache_dir, args.jobs, args.recompute)
    X_te, _ = get_features(te_paths, "test", cache_dir, args.jobs, args.recompute)
    print(f"  {X_tr.shape[1]} numbers per image | ~{ms_feat:.1f} ms per image to extract (parallel)")

    # 3-4. train on TRAIN, compare on VAL
    print("\nTraining and validating:")
    rows, fitted = [], []
    for family, setting, model in candidates(args.quick):
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0
        m, _ = evaluate(model, X_va, y_va, len(classes), class_to_group)
        rows.append({"model": family, "setting": setting, "train_seconds": round(train_s, 1),
                     **{f"val_{k}": round(v, 4) for k, v in m.items()}})
        fitted.append(model)
        print(f"  {family:20s} {setting:16s} val acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}"
              f"  ({train_s:.0f}s)")

    # The winner is whichever row has the highest val macro-F1; rows and fitted share the same order.
    best_i = int(np.argmax([r["val_macro_f1"] for r in rows]))
    best, best_row = fitted[best_i], rows[best_i]
    print(f"\nBest on val: {best_row['model']} ({best_row['setting']})")

    # 5. TEST, once, best model only
    test_m, test_pred = evaluate(best, X_te, y_te, len(classes), class_to_group)
    print(f"TEST: accuracy {test_m['accuracy']:.3f} | macro-F1 {test_m['macro_f1']:.3f} | "
          f"top-2 {test_m['top2_accuracy']:.3f} | group-level {test_m['group_accuracy']:.3f}")

    # 6. outputs
    args.reports.mkdir(parents=True, exist_ok=True)
    with open(args.reports / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_confusion(y_te, test_pred, classes, args.reports / "confusion_matrix_test.png",
                   f"Baseline ({best_row['model']}, HOG + colour) - test set")
    n_shown = misclassified_grid(te_paths, y_te, test_pred, classes, args.reports / "misclassified_test.jpg")
    report_txt = classification_report(y_te, test_pred, target_names=classes, digits=3)

    # Build results.md: a summary line, the validation table (best first), the test table,
    # then the per-class report from scikit-learn.
    pct = lambda v: f"{v * 100:.1f}%"
    md = ["# Baseline results: HOG + colour histogram features", "",
          f"Data: `{args.data}` | classes: {len(classes)} | train {len(y_tr)} / val {len(y_va)} / test {len(y_te)}",
          f"Features: {X_tr.shape[1]} per image (HOG {FEATURE_PARAMS['orientations']} orientations, "
          f"{FEATURE_PARAMS['pixels_per_cell']}px cells on {FEATURE_PARAMS['size']}x{FEATURE_PARAMS['size']}; "
          f"HSV histogram {FEATURE_PARAMS['hsv_bins']}; log aspect ratio)", "",
          "## Model selection (validation set)", "",
          "| Model | Setting | Val accuracy | Val macro-F1 | Val top-2 | Val group-level | Train time (s) |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["val_macro_f1"]):
        mark = " **(best)**" if r is best_row else ""
        md.append(f"| {r['model']}{mark} | {r['setting']} | {pct(r['val_accuracy'])} | {pct(r['val_macro_f1'])} | "
                  f"{pct(r['val_top2_accuracy'])} | {pct(r['val_group_accuracy'])} | {r['train_seconds']} |")
    md += ["", f"Chance level for {len(classes)} balanced classes: {pct(1 / len(classes))}.", "",
           f"## Test set (best model only: {best_row['model']}, {best_row['setting']})", "",
           "| Metric | Value |", "|---|---|",
           f"| Accuracy | {pct(test_m['accuracy'])} |",
           f"| Macro-F1 | {pct(test_m['macro_f1'])} |",
           f"| Top-2 accuracy | {pct(test_m['top2_accuracy'])} |",
           f"| Group-level accuracy (e.g. blazer/jacket/coat -> 'jacket') | {pct(test_m['group_accuracy'])} |",
           f"| Feature extraction time | {ms_feat:.1f} ms / image |",
           f"| Classifier prediction time | {test_m['predict_ms_per_image']:.2f} ms / image |", "",
           "## Per-class results (test)", "", "```", report_txt, "```", "",
           "Figures: `confusion_matrix_test.png`, "
           f"`misclassified_test.jpg` ({n_shown} random test mistakes; some are label errors, not model errors).", ""]
    (args.reports / "results.md").write_text("\n".join(md), encoding="utf-8")

    # I save the class names and feature settings with the model, so my app can rebuild the exact
    # same features and turn a predicted number back into a class name.
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": best, "classes": classes, "feature_params": FEATURE_PARAMS,
          "selected": best_row, "test": test_m}, args.model_out)
    print(f"\nWrote {args.reports / 'results.md'}, confusion matrix, misclassified grid; model -> {args.model_out}")


if __name__ == "__main__":   # required on Windows for parallel feature extraction
    main()
