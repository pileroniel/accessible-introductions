# CNN results (cnn_full_3): ResNet-50 pretrained on ImageNet, fine-tuned

Run: `cnn_full_3` | started 2026-11-22 23:58 | checkpoint `models\checkpoints\resnet50_cnn_full_3.pt`
Command: `scripts/train_cnn.py --epochs 4`

Data: `data\processed\garment` | classes: 10 | train 15000 / val 3000 / test 3000
Settings: phase 1 3 epochs (head only), phase 2 4 epochs (unfreeze layer3), batch 32, lr head 0.001 / backbone 0.0001, weight decay 0.0001, label smoothing 0.1, pad to square True

## Training history (validation set)

| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.6554 | 52.6% | 59.9% | 59.2% | 75.9% | 73.9% | 296 |
| 1 | 2 | 1.4677 | 59.2% | 60.4% | 59.6% | 76.9% | 74.7% | 215 |
| 1 | 3 | 1.4258 | 60.9% | 61.9% | 61.3% | 77.2% | 75.7% | 215 |
| 2 | 1 | 1.4234 | 61.2% | 66.3% | 65.5% | 81.3% | 79.2% | 398 |
| 2 | 2 **(best)** | 1.1978 | 71.1% | 68.8% | 68.5% | 83.0% | 80.7% | 385 |
| 2 | 3 | 1.0354 | 78.2% | 68.3% | 67.8% | 82.6% | 80.9% | 386 |
| 2 | 4 | 0.9084 | 84.2% | 67.8% | 67.7% | 82.6% | 80.6% | 386 |

## Test set (best weights only: phase 2, epoch 2)

| Metric | Value |
|---|---|
| Accuracy | 66.8% |
| Macro-F1 | 66.6% |
| Top-2 accuracy | 81.8% |
| Group-level accuracy | 79.4% |
| Prediction time, GPU | 11.3 ms / image |
| Prediction time, CPU | 97.6 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.728     0.563     0.635       300
    cardigan      0.574     0.580     0.577       300
        coat      0.715     0.853     0.778       300
       dress      0.840     0.930     0.883       300
      hoodie      0.737     0.673     0.704       300
      jacket      0.605     0.517     0.558       300
shirt_blouse      0.526     0.633     0.575       300
     sweater      0.550     0.567     0.558       300
     t_shirt      0.690     0.600     0.642       300
    tank_top      0.729     0.763     0.746       300

    accuracy                          0.668      3000
   macro avg      0.670     0.668     0.666      3000
weighted avg      0.670     0.668     0.666      3000

```

Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes).
