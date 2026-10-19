from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

def features(img):                       # img: 128x128 RGB numpy array
    shape = hog(img, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), channel_axis=-1)
    color = np.histogram(img, bins=32)[0] / img.size
    return np.concatenate([shape, color])

X_train = [features(im) for im in train_images]
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
clf.fit(X_train, y_train)
print(clf.score([features(im) for im in test_images], y_test))