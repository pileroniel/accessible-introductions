"""
Check that the DeepFashion downloads are complete and in the expected places.

Run from the project root (the Code/ folder):
    python scripts/check_deepfashion.py

Expected layout:
    data/raw/deepfashion/attr_predict/   Category & Attribute Prediction benchmark
    data/raw/deepfashion/multimodal/     DeepFashion-MultiModal
"""
from pathlib import Path

ROOT = Path("data/raw/deepfashion")
ATTR = ROOT / "attr_predict"
MM = ROOT / "multimodal"


def first(folder: Path, name: str):
    """Find a file anywhere under folder (zip contents sometimes add an extra level)."""
    hits = sorted(folder.rglob(name), key=lambda p: ("fine" in str(p).lower(), len(str(p))))
    return hits[0] if hits else None


def count(folder: Path, *exts):
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts) if folder.exists() else 0


def report(label, ok, detail=""):
    print(f"[{'OK ' if ok else 'MISSING'}] {label}{'  ' + detail if detail else ''}")
    return ok


all_ok = True

print("\n== Category & Attribute Prediction benchmark ==")
for name in ["list_category_cloth.txt", "list_category_img.txt", "list_bbox.txt", "list_eval_partition.txt"]:
    p = first(ATTR, name)
    all_ok &= report(name, p is not None, str(p) if p else "")

listed = None
p = first(ATTR, "list_category_img.txt")
if p:
    listed = int(p.read_text(encoding="utf-8").split()[0])  # line 1 of every list file = row count
on_disk = count(ATTR, ".jpg")
all_ok &= report("images on disk", on_disk > 0 and (listed is None or on_disk >= 0.99 * listed),
                 f"{on_disk:,} on disk vs {listed:,} listed" if listed else f"{on_disk:,} on disk")
all_ok &= report("Anno_fine folder", any(d.is_dir() and d.name.lower() == "anno_fine" for d in ATTR.rglob("*")))

print("\n== DeepFashion-MultiModal ==")
mm_imgs = count(MM, ".jpg")
all_ok &= report("images", mm_imgs >= 44_000, f"{mm_imgs:,} on disk (expected 44,096)")
mm_masks = count(MM, ".png")
all_ok &= report("parsing masks (.png)", mm_masks >= 12_000, f"{mm_masks:,} on disk (expected ~12,701)")
label_txts = [t for t in MM.rglob("*.txt") if "keypoint" not in t.name.lower()] if MM.exists() else []
all_ok &= report("label files (shape/fabric/color)", len(label_txts) >= 3, ", ".join(t.name for t in label_txts))
all_ok &= report("textual descriptions (.json)", any(MM.rglob("*.json")) if MM.exists() else False)

leftover = list(ROOT.rglob("*.zip")) if ROOT.exists() else []
if leftover:
    print(f"\nNote: {len(leftover)} .zip file(s) still present; delete them after unzipping to save space.")

print("\nAll good - ready to build the subset." if all_ok else "\nSomething is missing; re-download or re-unzip the items marked MISSING.")