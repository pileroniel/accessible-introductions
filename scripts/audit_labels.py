"""
I use this to estimate how often DeepFashion labels are WRONG, without checking every image.

I look at a small random sample per class and mark the mislabelled ones; the script turns that
into an error rate per class with a 95% confidence interval. That number goes in my report.

Step 1 - make the audit sheets (default: 30 random TEST images per class):
    python scripts/audit_labels.py make
  This writes:
    reports/label_audit/sheet_<class>.jpg   numbered thumbnails, one sheet per class
    reports/label_audit/audit.csv           one row per image, with an empty 'wrong' column

Step 2 - I open each sheet; in audit.csv I put 1 (or x / yes) in the 'wrong' column for every number whose
         garment is NOT the class the sheet is named after, and leave the rest blank.
         (Unsure? I put 1 only if a reasonable person would call it something else.)

Step 3 - summarise:
    python scripts/audit_labels.py summarize
  This writes reports/label_audit/summary.md

Why the TEST split? Wrong test labels directly lower my measured accuracy (my model is marked
wrong when it was right), so this tells me how to read my results. The train split comes from
the same source, so its error rate should be similar.

LEARN: Northcutt et al., "Pervasive Label Errors in Test Sets Destabilize ML Benchmarks" (NeurIPS 2021)
       https://arxiv.org/abs/2103.14749  and the companion site https://labelerrors.com
"""
import argparse
import csv
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path("reports/label_audit")


def make(data: Path, split: str, per_class: int, seed: int, force: bool = False):
    """Step 1: I draw a random sample per class, save one numbered contact sheet per class
    and write audit.csv with an empty 'wrong' column for me to fill in."""
    # Safety check: if I already marked rows in audit.csv, running 'make' again would overwrite
    # my work with a blank file, so I stop unless I pass --force.
    csv_path = OUT / "audit.csv"
    if csv_path.exists() and not force:
        with open(csv_path, encoding="utf-8-sig") as fh:
            marked = sum(is_marked(r.get("wrong", "")) for r in csv.DictReader(fh))
        if marked:
            raise SystemExit(f"{csv_path} already has {marked} marked rows. Running 'make' again would erase them.\n"
                             f"Run 'summarize' instead, or add --force if you really want fresh sheets.")
    random.seed(seed)                # fixed seed so I get the same sample if I rerun this
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls_dir in sorted(d for d in (data / split).iterdir() if d.is_dir()):
        # I sort the files first so the random sample depends only on the seed, not on disk order.
        files = sorted(cls_dir.glob("*.jpg"))
        sample = random.sample(files, min(per_class, len(files)))

        # Sheet layout: 6 tiles per row, each 180 px square plus a 22 px strip for its number,
        # with a 40 px header at the top for the class name.
        cols, tile, cap = 6, 180, 22
        nrows = math.ceil(len(sample) / cols)
        sheet = Image.new("RGB", (cols * tile, nrows * (tile + cap) + 40), "white")
        d = ImageDraw.Draw(sheet)
        d.text((8, 12), f"Class: {cls_dir.name}   -   mark the numbers that are NOT a {cls_dir.name}", fill="black")
        for i, f in enumerate(sample, start=1):
            # Grid position of tile i (row r, column c), then its top-left corner in pixels.
            r, c = divmod(i - 1, cols)
            x, y = c * tile, 40 + r * (tile + cap)
            with Image.open(f) as im:
                im = im.convert("RGB")
                im.thumbnail((tile - 8, tile - 8))   # shrinks in place and keeps the aspect ratio
                sheet.paste(im, (x + 4, y))
            # The red number under each tile matches the 'number' column in audit.csv.
            d.text((x + 6, y + tile - 4), f"#{i}", fill=(200, 0, 0))
            rows.append({"class": cls_dir.name, "number": i, "path": str(f), "wrong": ""})
        sheet.save(OUT / f"sheet_{cls_dir.name}.jpg", quality=90)

    # One CSV for all classes; I fill in the 'wrong' column by hand in Step 2.
    with open(OUT / "audit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "number", "path", "wrong"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT/'audit.csv'} and one sheet per class in {OUT}")


def is_marked(value: str) -> bool:
    """I accept the usual ways of marking a cell: 1, 1.0, x, yes, y, true, wrong (any case).
    1.0 is there because Excel can turn a typed 1 into 1.0 when it re-saves the file."""
    return str(value).strip().lower() in {"1", "1.0", "x", "yes", "y", "true", "wrong"}


def wilson(k: int, n: int, z: float = 1.96):
    """My 95% confidence interval for a proportion (Wilson); I use it because it behaves well for small n and rates near 0."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize():
    """Step 3: I read my filled-in audit.csv and write a markdown table of error rates
    (per class and overall) to summary.md."""
    csv_path = OUT / "audit.csv"
    with open(csv_path, encoding="utf-8-sig") as fh:   # utf-8-sig: copes with files re-saved by Excel
        rows = list(csv.DictReader(fh))

    # Sanity checks before I compute anything: the 'wrong' column has to exist, and at least one
    # row has to be marked. Zero marks almost always means the file wasn't saved, or 'make' was
    # run again and wiped it, so I stop with a hint instead of reporting a 0% error rate.
    if not rows or "wrong" not in rows[0]:
        raise SystemExit(f"{csv_path} has no 'wrong' column; columns found: {list(rows[0]) if rows else 'none'}")
    if not any(is_marked(r["wrong"]) for r in rows):
        # Any values I typed that is_marked() doesn't recognise, so the message can show them.
        filled = sorted({r["wrong"].strip() for r in rows if r["wrong"].strip()})
        raise SystemExit(
            f"No rows in {csv_path} are marked as wrong.\n"
            + (f"Values found in the 'wrong' column: {filled} (use 1 to mark a mislabelled image)\n" if filled else
               "The 'wrong' column is empty: was the file saved after marking, or was 'make' run again afterwards?\n")
            + "If you really checked every image and found none, that is very unlikely for DeepFashion; double-check.")

    # Tally per class: k is how many I marked wrong, n is how many I checked.
    # A row counts as wrong if is_marked() accepts its 'wrong' cell; blank counts as correct.
    per = {}
    for r in rows:
        k, n = per.get(r["class"], (0, 0))
        per[r["class"]] = (k + is_marked(r["wrong"]), n + 1)

    lines = ["# Label audit (random sample per class)", "",
             "| class | checked | mislabelled | error rate | 95% CI |", "|---|---|---|---|---|"]
    K = N = 0                         # running totals across all classes for the overall row
    # Worst classes first, sorted by error rate k / n from highest to lowest.
    for cls, (k, n) in sorted(per.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        lo, hi = wilson(k, n)
        lines.append(f"| {cls} | {n} | {k} | {k / n:.0%} | {lo:.0%}-{hi:.0%} |")
        K, N = K + k, N + n
    # Overall row, with one more decimal place since it pools every sample.
    lo, hi = wilson(K, N)
    lines.append(f"| **all** | {N} | {K} | {K / N:.1%} | {lo:.1%}-{hi:.1%} |")
    lines += ["", "With ~30 images per class the per-class intervals are wide; the overall rate is the"
              " more reliable number. A model's measured test accuracy can be at most about"
              f" {1 - K / N:.0%} if the model never repeats these label errors."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    # The module docstring doubles as the --help text, so my usage steps show up there too.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make", "summarize"])
    ap.add_argument("--data", type=Path, default=Path("data/processed/garment"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true", help="let 'make' overwrite an audit.csv that already has marks")
    a = ap.parse_args()
    make(a.data, a.split, a.per_class, a.seed, a.force) if a.step == "make" else summarize()