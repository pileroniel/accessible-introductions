# CNN results (cnn_full_2): ResNet-50 pretrained on ImageNet, fine-tuned

Run: `cnn_full_2` | started 2026-11-22 22:42 | checkpoint `models\checkpoints\resnet50_cnn_full_2.pt`
Command: `scripts/train_cnn.py --epochs 4 --lr-backbone 3e-5`

Data: `data\processed\garment` | classes: 10 | train 15000 / val 3000 / test 3000
Settings: phase 1 3 epochs (head only), phase 2 4 epochs (unfreeze layer3), batch 32, lr head 0.001 / backbone 3e-05, weight decay 0.0001, label smoothing 0.1, pad to square True

## Training history (validation set)

| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.6554 | 52.6% | 59.9% | 59.2% | 75.9% | 73.9% | 308 |
| 1 | 2 | 1.4677 | 59.2% | 60.3% | 59.6% | 76.9% | 74.7% | 218 |
| 1 | 3 | 1.4258 | 60.9% | 61.9% | 61.3% | 77.2% | 75.7% | 218 |
| 2 | 1 | 1.4927 | 57.8% | 64.2% | 63.4% | 79.7% | 77.5% | 403 |
| 2 | 2 | 1.304 | 66.8% | 66.0% | 65.5% | 80.9% | 78.6% | 389 |
| 2 | 3 | 1.2027 | 70.8% | 65.7% | 65.0% | 81.4% | 78.8% | 390 |
| 2 | 4 **(best)** | 1.1339 | 74.2% | 66.2% | 65.8% | 81.5% | 78.7% | 389 |

## Test set (best weights only: phase 2, epoch 4)

| Metric | Value |
|---|---|
| Accuracy | 64.4% |
| Macro-F1 | 64.2% |
| Top-2 accuracy | 79.9% |
| Group-level accuracy | 77.5% |
| Prediction time, GPU | 25.9 ms / image |
| Prediction time, CPU | 146.2 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.656     0.567     0.608       300
    cardigan      0.541     0.523     0.532       300
        coat      0.689     0.807     0.743       300
       dress      0.823     0.930     0.873       300
      hoodie      0.705     0.637     0.669       300
      jacket      0.552     0.493     0.521       300
shirt_blouse      0.508     0.610     0.555       300
     sweater      0.546     0.557     0.551       300
     t_shirt      0.672     0.607     0.637       300
    tank_top      0.747     0.710     0.728       300

    accuracy                          0.644      3000
   macro avg      0.644     0.644     0.642      3000
weighted avg      0.644     0.644     0.642      3000

```

Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes).
