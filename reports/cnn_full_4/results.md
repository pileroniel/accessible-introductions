# CNN results (cnn_full_6): ResNet-50 pretrained on ImageNet, fine-tuned

Run: `cnn_full_6` | started 2026-11-24 14:58 | checkpoint `models\checkpoints\resnet50_cnn_full_6.pt`
Command: `scripts/train_cnn.py --epochs 4 --unfreeze layer4`

Data: `data\processed\garment` | classes: 10 | train 15000 / val 3000 / test 3000
Settings: phase 1 3 epochs (head only), phase 2 4 epochs (unfreeze layer4), batch 32, lr head 0.001 / backbone 0.0001, weight decay 0.0001, label smoothing 0.1, pad to square True

## Training history (validation set)

| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.6554 | 52.6% | 59.8% | 59.1% | 75.9% | 73.8% | 319 |
| 1 | 2 | 1.4677 | 59.3% | 60.3% | 59.6% | 76.9% | 74.7% | 263 |
| 1 | 3 | 1.4258 | 60.9% | 61.9% | 61.3% | 77.2% | 75.7% | 263 |
| 2 | 1 | 1.4028 | 62.2% | 67.4% | 66.8% | 81.8% | 79.9% | 324 |
| 2 | 2 **(best)** | 1.2008 | 71.0% | 68.1% | 67.7% | 82.4% | 80.2% | 324 |
| 2 | 3 | 1.0389 | 78.6% | 67.2% | 66.7% | 81.6% | 80.3% | 324 |
| 2 | 4 | 0.9142 | 84.3% | 67.2% | 67.1% | 81.8% | 79.8% | 324 |

## Test set (best weights only: phase 2, epoch 2)

| Metric | Value |
|---|---|
| Accuracy | 65.3% |
| Macro-F1 | 64.9% |
| Top-2 accuracy | 80.8% |
| Group-level accuracy | 78.1% |
| Prediction time, GPU | 10.6 ms / image |
| Prediction time, CPU | 95.2 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.689     0.560     0.618       300
    cardigan      0.546     0.553     0.550       300
        coat      0.694     0.833     0.758       300
       dress      0.791     0.943     0.860       300
      hoodie      0.792     0.647     0.712       300
      jacket      0.556     0.463     0.505       300
shirt_blouse      0.526     0.563     0.544       300
     sweater      0.550     0.587     0.568       300
     t_shirt      0.671     0.620     0.645       300
    tank_top      0.707     0.757     0.731       300

    accuracy                          0.653      3000
   macro avg      0.652     0.653     0.649      3000
weighted avg      0.652     0.653     0.649      3000

```

Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes).
