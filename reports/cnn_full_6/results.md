# CNN results (cnn_full_6): ResNet-50 pretrained on ImageNet, fine-tuned

Run: `cnn_full_6` | started 2026-11-24 15:37 | checkpoint `models\checkpoints\resnet50_cnn_full_6.pt`
Command: `scripts/train_cnn.py --data data/processed/garment_train2000 --epochs 4 --unfreeze layer4`

Data: `data\processed\garment_train2000` | classes: 10 | train 20000 / val 3000 / test 3000
Settings: phase 1 3 epochs (head only), phase 2 4 epochs (unfreeze layer4), batch 32, lr head 0.001 / backbone 0.0001, weight decay 0.0001, label smoothing 0.1, pad to square True

## Training history (validation set)

| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.613 | 53.6% | 60.3% | 59.5% | 76.1% | 75.0% | 384 |
| 1 | 2 | 1.4551 | 59.7% | 61.8% | 61.2% | 77.8% | 75.2% | 335 |
| 1 | 3 | 1.4234 | 61.1% | 62.7% | 62.0% | 78.3% | 76.1% | 335 |
| 2 | 1 | 1.3833 | 62.9% | 67.1% | 66.4% | 82.0% | 80.2% | 418 |
| 2 | 2 | 1.2005 | 71.0% | 68.3% | 67.9% | 83.6% | 80.5% | 418 |
| 2 | 3 **(best)** | 1.048 | 77.7% | 68.4% | 67.9% | 83.0% | 80.6% | 418 |
| 2 | 4 | 0.9282 | 83.6% | 67.6% | 67.1% | 82.9% | 80.2% | 418 |

## Test set (best weights only: phase 2, epoch 3)

| Metric | Value |
|---|---|
| Accuracy | 65.9% |
| Macro-F1 | 65.5% |
| Top-2 accuracy | 81.2% |
| Group-level accuracy | 78.7% |
| Prediction time, GPU | 13.0 ms / image |
| Prediction time, CPU | 90.2 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.644     0.620     0.632       300
    cardigan      0.537     0.553     0.545       300
        coat      0.700     0.823     0.757       300
       dress      0.811     0.943     0.872       300
      hoodie      0.747     0.650     0.695       300
      jacket      0.640     0.473     0.544       300
shirt_blouse      0.583     0.550     0.566       300
     sweater      0.525     0.600     0.560       300
     t_shirt      0.676     0.613     0.643       300
    tank_top      0.718     0.763     0.740       300

    accuracy                          0.659      3000
   macro avg      0.658     0.659     0.655      3000
weighted avg      0.658     0.659     0.655      3000

```

Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes).
