# Baseline results: HOG + colour histogram features

Data: `data\processed\garment` | classes: 10 | train 15000 / val 3000 / test 3000
Features: 1785 per image (HOG 9 orientations, 16px cells on 128x128; HSV histogram [12, 4, 4]; log aspect ratio)

## Model selection (validation set)

| Model | Setting | Val accuracy | Val macro-F1 | Val top-2 | Val group-level | Train time (s) |
|---|---|---|---|---|---|---|
| Logistic regression **(best)** | C=0.001 | 47.2% | 46.6% | 64.8% | 61.5% | 8.5 |
| Logistic regression | C=0.01 | 46.0% | 45.6% | 63.0% | 60.1% | 4.8 |
| Linear SVM | C=0.0001 | 46.8% | 44.8% | 62.8% | 60.4% | 61.7 |
| Logistic regression | C=0.0001 | 45.5% | 44.6% | 62.6% | 59.2% | 17.2 |
| Linear SVM | C=0.001 | 45.9% | 44.4% | 62.1% | 59.7% | 101.9 |
| Linear SVM | C=0.01 | 43.1% | 42.2% | 59.6% | 57.1% | 170.8 |
| Majority class | - | 10.0% | 1.8% | 20.0% | 30.0% | 0.0 |

Chance level for 10 balanced classes: 10.0%.

## Test set (best model only: Logistic regression, C=0.001)

| Metric | Value |
|---|---|
| Accuracy | 47.8% |
| Macro-F1 | 47.1% |
| Top-2 accuracy | 65.5% |
| Group-level accuracy (e.g. blazer/jacket/coat -> 'jacket') | 61.4% |
| Feature extraction time | 3.4 ms / image |
| Classifier prediction time | 0.01 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.453     0.433     0.443       300
    cardigan      0.335     0.293     0.313       300
        coat      0.549     0.677     0.606       300
       dress      0.640     0.753     0.692       300
      hoodie      0.457     0.440     0.448       300
      jacket      0.364     0.343     0.353       300
shirt_blouse      0.332     0.300     0.315       300
     sweater      0.372     0.357     0.364       300
     t_shirt      0.578     0.493     0.532       300
    tank_top      0.606     0.687     0.644       300

    accuracy                          0.478      3000
   macro avg      0.468     0.478     0.471      3000
weighted avg      0.468     0.478     0.471      3000

```

Figures: `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes; some are label errors, not model errors).
