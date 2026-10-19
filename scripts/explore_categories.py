"""
On this script, we look at how the data meets our target variables, i.e t-shirt, blouse, etc

Prints how many images each of DeepFashion's 50 categories has in train/val/test,
whether it is upper-body (1), lower-body (2) or full-body (3), and which of my MVP
classes it is mapped to in config/garment_classes.json (or "-" if unused).

Also saves the table to reports/deepfashion_category_counts.csv for my midterm report.

Run from the project root (Code/):
    python scripts/explore_categories.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

# shared with build_garment_subset.py
from deepfashion_lists import find_annotation_file, read_annotation_file

# Paths are relative to the project root, which is why the script must be run from Code/
ROOT = Path("data/raw/deepfashion/attr_predict")                                    # unzipped DeepFashion benchmark
CONFIG = Path(__file__).resolve().parent.parent / "config" / "garment_classes.json"  # my MVP class definitions
OUT_CSV = Path("reports/deepfashion_category_counts.csv")                           # where the table gets saved


# 1. Read the three annotation files we need (verbose=True prints each file's row count and columns)
#   categories - one [name, body type] pair per category; position i holds category id i+1
#   img_cat    - one [image path, category id] pair per image
#   partition  - maps image path to "train", "val" or "test"
print("Reading annotation files...")
categories = read_annotation_file(find_annotation_file(ROOT, "list_category_cloth.txt"), verbose=True)
img_cat = read_annotation_file(find_annotation_file(ROOT, "list_category_img.txt"), verbose=True)
partition = dict(read_annotation_file(find_annotation_file(ROOT, "list_eval_partition.txt"), verbose=True))

# 2. Load my class config and flip it so each DeepFashion category name
#    looks up the MVP class it belongs to, e.g. "tee" gives "t_shirt". Lowercase so case doesn't matter.
#    If there's no config yet, every category just shows "-" in the table.
cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["classes"] if CONFIG.exists() else {}
mapped = {n.lower(): cls for cls, spec in cfg.items() for n in spec["deepfashion"]}

# 3. Count images per (category id, split), e.g. counts[(3, "train")] = 4,012.
#    Images missing from the partition file are counted under "?" 
counts = Counter()
for img, cat_id in img_cat:
    counts[(int(cat_id), partition.get(img, "?"))] += 1

# 4. Build one table row per category: its body type, image count per split, total,
#    and which MVP class it feeds into. Add a "-" if none
type_name = {"1": "upper", "2": "lower", "3": "full"}  # DeepFashion stores body type as 1/2/3
rows = []
for i, (name, ctype) in enumerate(categories, start=1):  # category ids are 1-based
    train_count, validate_count, test_count = counts[(i, "train")], counts[(i, "val")], counts[(i, "test")]
    rows.append({"id": i, "category": name, "body": type_name.get(ctype, ctype),
                 "train": train_count, "val": validate_count, "test": test_count, "total": train_count + validate_count + test_count,
                 "mvp_class": mapped.get(name.lower(), "-")})

# 5. Print the table, grouped by body type and biggest categories first within each group
rows.sort(key=lambda r: (r["body"], -r["total"]))
print(f"\n{'category':16s} {'body':6s} {'train':>7s} {'val':>6s} {'test':>6s}  mvp_class")
for r in rows:
    print(f"{r['category']:16s} {r['body']:6s} {r['train']:7d} {r['val']:6d} {r['test']:6d}  {r['mvp_class']}")

# 6. Add up the train images of every category mapped to each MVP class. This is the most
#    build_garment_subset.py could keep per class, so it shows which classes will fall short of the cap.
print("\nImages available per MVP class (train split):")
per_class = Counter()
for r in rows:
    if r["mvp_class"] != "-":
        per_class[r["mvp_class"]] += r["train"]
for cls in cfg:
    print(f"  {cls:14s} {per_class[cls]:7d}")

# 7. Save the same table as a CSV for the report
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved {OUT_CSV}")
