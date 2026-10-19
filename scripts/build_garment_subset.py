"""
Build the cleaned GARMENT-TYPE subset from DeepFashion's
"Category and Attribute Prediction Benchmark" (no password needed).

What it does, in order:
  1. Reads the annotation lists (category names, each image's category, bounding boxes,
     official train/val/test partition).
  2. Maps DeepFashion's 50 categories onto my MVP classes (config/garment_classes.json).
  3. For each class and split, walks a shuffled list of candidates and keeps an image only
     if it passes the cleaning checks (file opens, box is valid and big enough, not a
     duplicate of an image already kept anywhere, which prevents train/test leakage).
  4. Crops each kept image to its clothing bounding box (+ padding), resizes, and saves it
     in PyTorch ImageFolder layout:  out/<split>/<class>/<file>.jpg
  5. Writes manifest.csv (one row per kept image), report.md (counts + why images were
     dropped) and a preview grid so I can check for label quality.

Resources I found useful for this section:
  - Dataset format: the README guide document inside the Anno/ and Eval/ folders.
  - ImageFolder layout: https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
  - Why dedup across splits matters:  https://scikit-learn.org/stable/common_pitfalls.html)

Usage:
  python scripts/build_garment_subset.py            (uses the defaults below)
  python scripts/build_garment_subset.py --per-class train=200 val=50 test=50   (quick trial run)
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

# Functions for reading the deepfashion labels (.txt files) live in deepfashion_lists.py,
# because explore_categories.py uses them too
from deepfashion_lists import find_annotation_file, read_annotation_file


def resolve_image(root: Path, rel: str) -> Path:
    # Lists say "img/Category_Name/img_00000001.jpg" and img.zip is unzipped directly under root,
    # so root/rel is the image. Missing files are caught and counted in check_candidate().
    return root / rel


# cleaning + saving the files

def file_md5(path: Path) -> str:
    # hash out function to differentiate images
    # INFO however, this may fail when we crop out an image and then try to compare with the uncropped image again
    return hashlib.md5(path.read_bytes()).hexdigest()


def crop_and_save(src: Path, box, out_path: Path, pad: float, short_side: int):
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1 # the bounding boxes width and height

        #we  pad the box a little so the classifier sees some context (collar, sleeves)
        # pad is a fraction of how much we're padding 

        x1 = max(0, int(x1 - pad * bw)) # move our left edge more left (i.e substract from x)
        y1 = max(0, int(y1 - pad * bh)) # move our top edge up
        x2 = min(w, int(x2 + pad * bw)) # move our  right edge right, i.e add from right edge
        y2 = min(h, int(y2 + pad * bh)) # move the bottom edge down

        crop = im.crop((x1, y1, x2, y2))

        scale = short_side / min(crop.size) # find the ratio of the old to new padded image, then use this ratio to shrink it 
        
        if scale < 1:  # only shrink (downsampling),  upscaling will just make the images bigger and blurrier with no information added to it
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.BICUBIC)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, quality=92)
        return crop.size


def check_candidate(src: Path, box, min_box: int):
    """Decide whether an image and its bounding box are clean enough to keep.

        Checks, in order: the file exists, the image opens without corruption, the box
        (clipped to the image edges) is a valid rectangle, its short side is at least
        `min_box` pixels, and its aspect ratio is between 1:4 and 4:1.

        Args:
            src: Path to the image file.
            box: DeepFashion bounding box (x1, y1, x2, y2), top-left origin, in pixels.
            min_box: Smallest allowed short side of the box, in pixels.

        Returns:
            A tuple (ok, reason, clipped_box, size):
                ok: True if the image passed every check.
                reason: "ok", or why it failed ("missing_file", "corrupt_file",
                    "invalid_box", "box_too_small", "extreme_aspect_ratio").
                clipped_box: The box trimmed to fit inside the image, or None if rejected.
                size: The image's (width, height), or None if it couldn't be read.
        """

    # Check 1: does the file exist on disk at all?
    if not src.exists():
        return False, "missing_file", None, None

    # Check 2: can the image actually be opened? (catches broken/half-downloaded files)
    try:
        with Image.open(src) as im:
            im.verify()  # scans the file for corruption without decoding all the pixels
        # verify() leaves the image object unusable afterwards, so we open it a second time to read its size

        with Image.open(src) as im:
            w, h = im.size  # image width and height in pixels
    except Exception:  # any error while opening = the file is corrupt
        return False, "corrupt_file", None, None

    # Check 3: clip the box so it stays inside the image
    # (some DeepFashion boxes stick slightly past the image edge)
    x1, y1, x2, y2 = box                     # top-left (x1, y1) and bottom-right (x2, y2) corners
    x1, y1 = max(0, x1), max(0, y1)          # left/top edges can't go below 0
    x2, y2 = min(w, x2), min(h, y2)          # right/bottom edges can't go past the image width/height

    # Check 4: is the box still a real rectangle? (right edge must be right of left edge, bottom below top)
    if x2 <= x1 or y2 <= y1:
        return False, "invalid_box", None, (w, h)

    # Check 5: is the box big enough? If its shorter side is under min_box pixels (default 48),
    # the garment is too tiny to teach the classifier anything useful
    if min(x2 - x1, y2 - y1) < min_box:
        return False, "box_too_small", None, (w, h)

    # Check 6: is the box a sensible shape? ar = aspect ratio = width / height
    #   ar > 4    - more than 4x wider than tall (a thin horizontal strip)
    #   ar < 0.25 - more than 4x taller than wide (a thin vertical sliver)
    # Shapes like that are usually bad annotations, not real garments
    ar = (x2 - x1) / (y2 - y1)
    if ar > 4 or ar < 0.25:
        return False, "extreme_aspect_ratio", None, (w, h)

    # Passed every check: keep it, and hand back the clipped box for crop_and_save() to use
    return True, "ok", (x1, y1, x2, y2), (w, h)


def preview_grid(rows, out_path: Path, per_class=8, tile=128):
    """ Builds one  picture showing a few sample crops from every class, so we can check the  labels in one go

    Each class gets one row: its name on the left, then up to `per_class`
    thumbnails from the train split. Rows are in alphabetical order by class.
    Nothing is saved if `rows` has no train images.

    Args:
        rows: Kept-image records built in main(). Each one is a dict that
            needs at least the "split", "label" and "out_path" keys.
        out_path: Where to save the preview JPEG.
        per_class: Maximum number of thumbnails per class.
        tile: Size of each thumbnail cell in pixels (tile x tile).
    """

    # Step 1: collect up to `per_class` training images for each class
    by_class = defaultdict(list)  # maps class name to a list of image paths (defaultdict makes an empty list automatically)
    for r in rows:
        # only use the train split, and stop adding once this class already has enough samples
        if r["split"] == "train" and len(by_class[r["label"]]) < per_class:
            by_class[r["label"]].append(r["out_path"])

    classes = sorted(by_class)  # class names in alphabetical order (one row of the grid each)
    if not classes:              # nothing was kept, so nothing to draw
        return

    # Step 2: make a blank white canvas
    #   width  = room for `per_class` thumbnails + 140 px on the left for the class name
    #   height = one tile-high row per class
    grid = Image.new("RGB", (tile * per_class + 140, tile * len(classes)), "white") # type: ignore[arg-type]
    draw = ImageDraw.Draw(grid)  # a "pen" we can use to write text on the canvas

    # Step 3: fill in each row
    for i, c in enumerate(classes):  # i = row number (0, 1, 2...), c = class name
        # write the class name on the left, vertically centred in its row
        draw.text((5, i * tile + tile // 2), c, fill="black")
        for j, p in enumerate(by_class[c]):  # j = column number, p = path of a saved crop
            with Image.open(p) as im:
                im.thumbnail((tile, tile))  # shrink to fit inside tile x tile, keeping its shape
                # paste it in: x = after the 140 px label area + j tiles across, y = row i
                grid.paste(im, (140 + j * tile, i * tile))

    # Step 4: save the finished preview as a JPEG
    grid.save(out_path, quality=90)


#

def main():

    # 1. Command-line options. Every setting has a default, so plain `python scripts/build_garment_subset.py`
    #    works; pass a flag (e.g. --pad 0.2) to override one. `args.<name>` holds the final values.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/deepfashion/attr_predict"), help="Unzipped 'Category and Attribute Prediction Benchmark' folder")
    ap.add_argument("--out", type=Path, default=Path("data/processed/garment"))
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config" / "garment_classes.json")
    ap.add_argument("--per-class", nargs="+", default=["train=1500", "val=300", "test=300"],
                    help="max images per class per split, e.g. train=1500 val=300 test=300")
    ap.add_argument("--min-box", type=int, default=48, help="drop boxes whose short side is below this (pixels)")
    ap.add_argument("--pad", type=float, default=0.10, help="padding around the box, as a fraction of box size")
    ap.add_argument("--short-side", type=int, default=256, help="resize crops so the short side is at most this")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 2. Extract infor from the parser arguments/settings:
    #   caps - per-split image limits, e.g. ["train=1500", "val=300"] becomes {"train": 1500, "val": 300}
    #   cfg  - my class definitions from garment_classes.json (which DeepFashion categories make up
    #          each MVP class, plus the "spoken" name)
    #   seed - fixes the random shuffle so re-running the script picks exactly the same images
    caps ={k: int(v) for k, v in (s.split("=") for s in args.per_class)}
    cfg = json.loads(args.config.read_text(encoding="utf-8"))["classes"]
    random.seed(args.seed)

    # 3. Read DeepFashion's four annotation files into lookup tables:
    #   cat_names - list of the 50 category names; position i holds category id i+1
    #   img_cat   - maps image path to category id, e.g. "img/Sheer_Blouse/img_01.jpg" gives 3 ,i.e a dict of {"img_001": 3,"img_005": 11 }
    #   bboxes    - maps image path to its (x1, y1, x2, y2) box (top-left / bottom-right corners) {"img_001": (20,30, 200, 500)}
    #   partition - maps image path to "train", "val" or "test" (DeepFashion's official split) {"img_001": "train"}
    
    cat_names = [row[0] for row in read_annotation_file(find_annotation_file(args.root, "list_category_cloth.txt"))]

    # 
    img_cat = {row[0]: int(row[1]) for row in read_annotation_file(find_annotation_file(args.root, "list_category_img.txt"))}

    bboxes = {row[0]: tuple(map(int, row[1:5])) for row in read_annotation_file(find_annotation_file(args.root, "list_bbox.txt"))}

    partition = {row[0]: row[1] for row in read_annotation_file(find_annotation_file(args.root, "list_eval_partition.txt"))}

    # fail early if the first listed image isn't where we expect. It means that the folder layout is wrong because the unzipping process wasn't done well
    first_img = next(iter(img_cat))
    if not resolve_image(args.root, first_img).exists():
        raise SystemExit(f"Expected {resolve_image(args.root, first_img)} but it does not exist.\n"
                         f"Check that img.zip was unzipped so that {args.root / 'img'} holds the category folders.")

    # 4. Map each DeepFashion category to an MVP class (case-insensitive so "button-down" == "Button-Down")
    #   name_to_class flips the config around: instead of "class to its DeepFashion categories" it becomes
    #   "DeepFashion category to class", so each image's category can be looked up directly.
    #   unmatched_cfg collects category names in my config that don't exist in DeepFashion (usually typos);
    #   they're listed as a warning at the bottom of report.md.
    name_to_class = {}
    for cls, spec in cfg.items():
        for n in spec["deepfashion"]:
            name_to_class[n.lower()] = cls
    known = {n.lower() for n in cat_names}
    unmatched_cfg = sorted(n for n in name_to_class if n not in known)

    # Sort every image into a "pool" by (split, class). An image only gets a pool if:
    #   - its category belongs to one of my classes (categories I didn't map are skipped),
    #   - its split is one I asked for in --per-class,
    #   - it has a bounding box.
    # These pools are the candidates that step 5 picks from.
    pools = defaultdict(list)  # maps (split, class) to a list of image names
    for img, cat_id in img_cat.items():
        cname = cat_names[cat_id - 1]  # category ids are 1-based
        cls = name_to_class.get(cname.lower())
        split = partition.get(img)
        if cls and split in caps and img in bboxes:
            pools[(split, cls)].append(img)

    # 5. Clean + save
    #   seen_md5 - maps the fingerprint of every image kept so far to the split it went to (for duplicate detection)
    #   drops    - tally of why images were rejected, e.g. {"box_too_small": 120, "duplicate_of_test": 4}
    #   rows     - one record per kept image; becomes manifest.csv
    seen_md5 = {}
    drops = Counter()
    rows = []

    # process test first so that, if a duplicate exists, the copy kept is the test one
    # and the train copy is dropped (train should never contain test images)
    for split in sorted(caps, key=lambda s: {"test": 0, "val": 1, "train": 2}.get(s, 3)):
        for cls in sorted(cfg):

            # For this (split, class): shuffle the candidates, then walk through them keeping images
            # until the cap is reached. Each candidate must:
            #   1. pass check_candidate() (file OK, box valid and big enough), else count the reason and skip
            #   2. not be an exact duplicate of an image already kept in ANY split, else skip
            # Survivors are cropped + saved to out/<split>/<class>/<original-category>__<image-name>.jpg
            # and recorded in `rows`. Shuffling first means we get a random sample, not just the first N files.
            cand = pools[(split, cls)]
            random.shuffle(cand)
            kept = 0
            for img in cand:
                if kept >= caps[split]:
                    break
                src = resolve_image(args.root, img)
                ok, reason, box, size = check_candidate(src, bboxes[img], args.min_box)
                if not ok:
                    drops[reason] += 1
                    continue
                h = file_md5(src)
                if h in seen_md5:
                    drops["duplicate_of_" + seen_md5[h]] += 1
                    continue
                seen_md5[h] = split
                src_cat = Path(img).parent.name
                out_path = args.out / split / cls / f"{src_cat}__{Path(img).stem}.jpg"
                crop_and_save(src, box, out_path, args.pad, args.short_side)
                rows.append({"split": split, "label": cls, "source_category": cat_names[img_cat[img] - 1],
                             "deepfashion_path": img, "out_path": str(out_path),
                             "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                             "orig_w": size[0], "orig_h": size[1], "md5": h})
                kept += 1
            print(f"{split:5s} {cls:14s} kept {kept:5d} / {len(cand):6d} candidates")

    # 6. Write the outputs: manifest, report, preview

    # manifest.csv: a spreadsheet with one row per saved crop (where it came from, its label, its box,
    # its md5). Useful for tracing any training image back to the original DeepFashion file.
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["split"])
        w.writeheader(); w.writerows(rows)

    counts = Counter((r["split"], r["label"]) for r in rows)
    
    splits = sorted(caps, key=lambda s: {"train": 0, "val": 1, "test": 2}.get(s, 3))

    lines = ["# Garment subset report", "",
             f"Source: {args.root}  |  seed {args.seed}  |  caps {caps}  |  min box {args.min_box}px", "",
             "| class | spoken | " + " | ".join(splits) + " | available (train pool) |",
             "|---|---|" + "---|" * len(splits) + "---|"]

    
    for cls in sorted(cfg):
        lines.append(f"| {cls} | {cfg[cls]['spoken']} | " + " | ".join(str(counts[(s, cls)]) for s in splits)
                     + f" | {len(pools[('train', cls)])} |")
    drop_lines = [f"- {k}: {v}" for k, v in drops.most_common()] or ["- none"]
    lines += ["", "## Dropped during cleaning", ""] + drop_lines
    src_mix = Counter((r["label"], r["source_category"]) for r in rows if r["split"] == "train")
    lines += ["", "## Source categories inside each class (train)", ""]


    for cls in sorted(cfg):
        parts = [f"{c} {n}" for (l, c), n in sorted(src_mix.items()) if l == cls]
        lines.append(f"- {cls}: " + ", ".join(parts))

    if unmatched_cfg:
        lines += ["", "## WARNING: config names not found in list_category_cloth.txt", ""] + [f"- {n}" for n in unmatched_cfg]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    preview_grid(rows, args.out / "preview_train.jpg")
    print(f"\nSaved {len(rows)} crops. See {args.out/'report.md'} and {args.out/'preview_train.jpg'}")


if __name__ == "__main__":
    main()