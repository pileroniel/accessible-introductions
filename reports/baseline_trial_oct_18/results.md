# Baseline results: HOG + colour histogram features

Data: `data\interim\garment_trial` | classes: 10 | train 2000 / val 500 / test 500
Features: 1785 per image (HOG 9 orientations, 16px cells on 128x128; HSV histogram [12, 4, 4]; log aspect ratio)

## Model selection (validation set)

| Model | Setting | Val accuracy | Val macro-F1 | Val top-2 | Val group-level | Train time (s) |
|---|---|---|---|---|---|---|
| Logistic regression **(best)** | C=0.01 | 37.6% | 37.2% | 52.0% | 52.4% | 0.9 |
| Linear SVM | C=0.001 | 35.2% | 33.9% | 50.6% | 49.6% | 9.7 |
| Majority class | - | 10.0% | 1.8% | 20.0% | 30.0% | 0.0 |

Chance level for 10 balanced classes: 10.0%.

## Test set (best model only: Logistic regression, C=0.01)

| Metric | Value |
|---|---|
| Accuracy | 37.4% |
| Macro-F1 | 36.7% |
| Top-2 accuracy | 56.0% |
| Group-level accuracy (e.g. blazer/jacket/coat -> 'jacket') | 52.6% |
| Feature extraction time | 3.8 ms / image |
| Classifier prediction time | 0.01 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.351     0.400     0.374        50
    cardigan      0.140     0.120     0.129        50
        coat      0.421     0.480     0.449        50
       dress      0.559     0.660     0.606        50
      hoodie      0.380     0.380     0.380        50
      jacket      0.239     0.220     0.229        50
shirt_blouse      0.140     0.140     0.140        50
     sweater      0.371     0.260     0.306        50
     t_shirt      0.408     0.400     0.404        50
    tank_top      0.630     0.680     0.654        50

    accuracy                          0.374       500
   macro avg      0.364     0.374     0.367       500
weighted avg      0.364     0.374     0.367       500

```

Figures: `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes; some are label errors, not model errors).
