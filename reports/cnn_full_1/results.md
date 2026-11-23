# CNN results: ResNet-50 pretrained on ImageNet, fine-tuned

Data: `data\processed\garment` | classes: 10 | train 15000 / val 3000 / test 3000
Settings: phase 1 3 epochs (head only), phase 2 12 epochs (unfreeze layer3), batch 32, lr head 0.001 / backbone 0.0001, weight decay 0.0001, label smoothing 0.1, pad to square True

## Training history (validation set)

| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.6554 | 52.6% | 59.9% | 59.2% | 75.9% | 73.9% | 313 |
| 1 | 2 | 1.4677 | 59.2% | 60.4% | 59.7% | 76.9% | 74.7% | 228 |
| 1 | 3 | 1.4258 | 60.9% | 61.9% | 61.3% | 77.2% | 75.7% | 228 |
| 2 | 1 | 1.4238 | 61.2% | 66.5% | 65.8% | 81.1% | 79.4% | 455 |
| 2 | 2 **(best)** | 1.2085 | 70.8% | 69.1% | 68.7% | 82.7% | 81.2% | 405 |
| 2 | 3 | 1.0684 | 76.8% | 67.9% | 67.5% | 82.6% | 80.4% | 391 |
| 2 | 4 | 0.9458 | 82.5% | 66.7% | 66.6% | 81.4% | 79.3% | 391 |
| 2 | 5 | 0.845 | 87.4% | 66.4% | 65.9% | 80.3% | 79.4% | 391 |
| 2 | 6 | 0.7617 | 91.2% | 65.9% | 65.3% | 79.9% | 79.1% | 391 |
| 2 | 7 | 0.6972 | 94.2% | 65.5% | 65.2% | 79.6% | 78.7% | 389 |
| 2 | 8 | 0.6558 | 96.3% | 65.8% | 65.6% | 79.8% | 78.7% | 391 |
| 2 | 9 | 0.6265 | 97.4% | 65.9% | 65.5% | 79.4% | 78.8% | 391 |
| 2 | 10 | 0.6148 | 98.0% | 66.2% | 65.9% | 80.2% | 79.2% | 391 |
| 2 | 11 | 0.5999 | 98.6% | 66.4% | 66.1% | 79.8% | 79.4% | 388 |
| 2 | 12 | 0.5975 | 98.7% | 65.8% | 65.4% | 79.9% | 79.0% | 390 |

## Test set (best weights only: phase 2, epoch 2)

| Metric | Value |
|---|---|
| Accuracy | 66.4% |
| Macro-F1 | 66.1% |
| Top-2 accuracy | 82.0% |
| Group-level accuracy | 78.9% |
| Prediction time, GPU | 13.8 ms / image |
| Prediction time, CPU | 107.1 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.725     0.597     0.654       300
    cardigan      0.566     0.570     0.568       300
        coat      0.681     0.860     0.760       300
       dress      0.830     0.930     0.877       300
      hoodie      0.737     0.627     0.677       300
      jacket      0.581     0.517     0.547       300
shirt_blouse      0.548     0.603     0.575       300
     sweater      0.564     0.543     0.553       300
     t_shirt      0.679     0.600     0.637       300
    tank_top      0.724     0.797     0.759       300

    accuracy                          0.664      3000
   macro avg      0.664     0.664     0.661      3000
weighted avg      0.664     0.664     0.661      3000

```

Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes).
