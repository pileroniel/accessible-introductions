"""
My hand-crafted image features for the non-deep-learning baseline.

I turn one image into one fixed-length vector of numbers:
    [ HOG (shape: edge directions)  |  HSV colour histogram  |  log aspect ratio ]

I keep this in its own module so I use the exact same function at training time
(scripts/train_baseline.py) and later at inference time in my app pipeline.
If training and inference computed features differently, my model would silently get worse.

LEARN:
  HOG visual walkthrough: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html
  skimage.feature.hog API: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
  HSV colour space:         https://en.wikipedia.org/wiki/HSL_and_HSV
"""
import numpy as np
from PIL import Image
from skimage.feature import hog

# I keep all my feature settings here so I can save them alongside the trained model.
FEATURE_PARAMS = {
    "size": 128,                  # I resize every crop to size x size before HOG
    "orientations": 9,            # HOG: number of edge-direction bins (0-180 degrees)
    "pixels_per_cell": 16,        # HOG: 128/16 gives me 8 x 8 cells
    "cells_per_block": 2,         # HOG: I normalise contrast over 2 x 2 cell blocks
    "hsv_bins": [12, 4, 4],       # my colour histogram bins for hue, saturation, value
}


def hog_features(rgb: np.ndarray, p=FEATURE_PARAMS) -> np.ndarray:
    """My shape descriptor. With my defaults: 7 x 7 blocks x (2 x 2 cells x 9 bins) = 1,764 numbers."""
    return hog(rgb,
               orientations=p["orientations"],
               pixels_per_cell=(p["pixels_per_cell"],) * 2,
               cells_per_block=(p["cells_per_block"],) * 2,
               block_norm="L2-Hys",
               channel_axis=-1)        # I use the strongest gradient across R, G, B at each pixel


def color_features(img: Image.Image, p=FEATURE_PARAMS) -> np.ndarray:
    """I build separate histograms of hue, saturation and value, each normalised to sum to 1 (12+4+4 = 20 numbers).
    I use HSV instead of RGB so 'which colour' (hue) is separated from 'how bright' (value),
    and the same red sweater in shade and in sunlight looks more alike to my model."""
    hsv = np.asarray(img.convert("HSV"))
    parts = []
    for ch, bins in enumerate(p["hsv_bins"]):
        h, _ = np.histogram(hsv[..., ch], bins=bins, range=(0, 256))
        parts.append(h / h.sum())
    return np.concatenate(parts)


def extract(path_or_img, p=FEATURE_PARAMS) -> np.ndarray:
    """I compute the full feature vector for one image (a file path or a PIL image)."""
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = img.convert("RGB")
    # Resizing to a square stretches the image, which throws away its shape (a long dress vs a
    # cropped jacket), so I keep the original aspect ratio as one extra feature.
    aspect = np.log(img.width / img.height)
    small = img.resize((p["size"], p["size"]), Image.BILINEAR)
    return np.concatenate([
        hog_features(np.asarray(small), p),
        color_features(small, p),
        [aspect],
    ]).astype(np.float32)


def feature_length(p=FEATURE_PARAMS) -> int:
    return len(extract(Image.new("RGB", (p["size"], p["size"])), p))
