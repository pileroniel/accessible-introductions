"""
Shared helpers for reading DeepFashion's annotation lists (.txt files).
Used by build_garment_subset.py and explore_categories.py.
"""
from pathlib import Path


def find_annotation_file(root: Path, file_name: str) -> Path:
    """DeepFashion has shipped its lists under Anno/, Anno_coarse/, Anno_fine/ and Eval/.
    Prefer the coarse/original copy for category + bbox lists (they cover all 289k images)."""

    # root.rglob recursively unveils all the folders, subfolders and file and returns files that match 'file_name'

    # we sort it to prefer the files in Anno_coarse because Anno_coarse has labelled all the 289,222 images. Anno_fine, has just taken a subset of these images and annotated them with some attributes

    # note that sorting starts with smaller lengths first and False first before truea (i.e Anno_coarse befor Anno_fine)

    hits = sorted(root.rglob(file_name), key=lambda p: (any(part.lower() == "anno_fine" for part in p.parts), len(p.parts)))
    if not hits:
        raise SystemExit(f"Could not find {file_name} anywhere under {root}. "
                         f"Run scripts/check_deepfashion.py to check the dataset is unzipped correctly.")
    return hits[0]


def read_annotation_file(path: Path, verbose: bool = False):
    """Read and extract the label data from the files by excluding the first 2 lines, i.e: line 1 = number of rows, line 2 = column header, then data.

    With verbose=True, also prints the file's row count and column names (handy when exploring)."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if verbose:
        print(f"  {path.name}: {lines[0].strip()} rows | columns: {lines[1].strip()}")
    return [ln.split() for ln in lines[2:] if ln.strip()]  # this line here skips the two lines and then strips the spaces in the lines
