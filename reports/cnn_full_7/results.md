# CNN results (cnn_full_7): ResNet-50 pretrained on ImageNet, fine-tuned

Run: `cnn_full_7` | started 2026-11-24 17:11 | checkpoint `models\checkpoints\resnet50_cnn_full_7.pt`
Command: `scripts/train_cnn.py --data data/processed/garment_train2000 --epochs 4 --weight-decay 1e-2 --label-smoothing 0.2`

Data: `data\processed\garment_train2000` | classes: 10 | train 20000 / val 3000 / test 3000
Settings: phase 1 3 epochs (head only), phase 2 4 epochs (unfreeze layer3), batch 32, lr head 0.001 / backbone 0.0001, weight decay 0.01, label smoothing 0.2, pad to square True

## Training history (validation set)

| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.7557 | 53.9% | 60.2% | 59.4% | 76.4% | 74.7% | 390 |
| 1 | 2 | 1.6398 | 59.8% | 62.0% | 61.4% | 78.0% | 75.1% | 336 |
| 1 | 3 | 1.6148 | 61.1% | 63.0% | 62.3% | 78.2% | 76.2% | 336 |
| 2 | 1 | 1.598 | 62.1% | 68.1% | 67.3% | 82.2% | 80.3% | 597 |
| 2 | 2 | 1.4362 | 71.2% | 69.7% | 69.3% | 84.4% | 81.0% | 592 |
| 2 | 3 | 1.3169 | 77.5% | 69.6% | 69.2% | 84.2% | 82.0% | 575 |
| 2 | 4 **(best)** | 1.2228 | 83.5% | 70.1% | 69.8% | 83.8% | 82.0% | 576 |

## Test set (best weights only: phase 2, epoch 4)

| Metric | Value |
|---|---|
| Accuracy | 67.0% |
| Macro-F1 | 66.7% |
| Top-2 accuracy | 81.8% |
| Group-level accuracy | 79.6% |
| Prediction time, GPU | 9.8 ms / image |
| Prediction time, CPU | 88.7 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.658     0.597     0.626       300
    cardigan      0.607     0.550     0.577       300
        coat      0.689     0.843     0.759       300
       dress      0.819     0.953     0.881       300
      hoodie      0.770     0.690     0.728       300
      jacket      0.590     0.523     0.555       300
shirt_blouse      0.559     0.613     0.585       300
     sweater      0.561     0.580     0.570       300
     t_shirt      0.673     0.637     0.654       300
    tank_top      0.755     0.710     0.732       300

    accuracy                          0.670      3000
   macro avg      0.668     0.670     0.667      3000
weighted avg      0.668     0.670     0.667      3000

```

Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes).
