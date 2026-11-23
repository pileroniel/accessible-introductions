# CNN results: ResNet-50 pretrained on ImageNet, fine-tuned

Data: `data\processed\garment` | classes: 10 | train 64 / val 64 / test 64
Settings: phase 1 0 epochs (head only), phase 2 15 epochs (unfreeze all), batch 32, lr head 0.001 / backbone 0.0001, weight decay 0.0001, label smoothing 0.0, pad to square True

## Training history (validation set)

| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 2.372 | 3.1% | 14.1% | 11.3% | 35.9% | 32.8% | 97 |
| 2 | 2 | 1.9371 | 76.6% | 31.2% | 30.3% | 51.6% | 43.8% | 4 |
| 2 | 3 | 1.5778 | 100.0% | 56.2% | 58.3% | 76.6% | 67.2% | 4 |
| 2 | 4 | 1.2497 | 100.0% | 81.2% | 81.8% | 90.6% | 82.8% | 4 |
| 2 | 5 | 0.9275 | 100.0% | 93.8% | 94.0% | 98.4% | 96.9% | 4 |
| 2 | 6 **(best)** | 0.6918 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 3 |
| 2 | 7 | 0.4777 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 3 |
| 2 | 8 | 0.3775 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 3 |
| 2 | 9 | 0.2592 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 3 |
| 2 | 10 | 0.2107 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 3 |
| 2 | 11 | 0.172 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 3 |
| 2 | 12 | 0.1612 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 3 |
| 2 | 13 | 0.1425 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 3 |
| 2 | 14 | 0.1274 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 3 |
| 2 | 15 | 0.1249 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 3 |

## Test set (best weights only: phase 2, epoch 6)

| Metric | Value |
|---|---|
| Accuracy | 100.0% |
| Macro-F1 | 100.0% |
| Top-2 accuracy | 100.0% |
| Group-level accuracy | 100.0% |
| Prediction time, GPU | 13.6 ms / image |
| Prediction time, CPU | 90.6 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      1.000     1.000     1.000         7
    cardigan      1.000     1.000     1.000         6
        coat      1.000     1.000     1.000         7
       dress      1.000     1.000     1.000         6
      hoodie      1.000     1.000     1.000         7
      jacket      1.000     1.000     1.000         6
shirt_blouse      1.000     1.000     1.000         6
     sweater      1.000     1.000     1.000         7
     t_shirt      1.000     1.000     1.000         6
    tank_top      1.000     1.000     1.000         6

    accuracy                          1.000        64
   macro avg      1.000     1.000     1.000        64
weighted avg      1.000     1.000     1.000        64

```

Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` (0 random test mistakes).
