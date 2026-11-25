# CNN results (cnn_full_5): ResNet-50 pretrained on ImageNet, fine-tuned

Run: `cnn_full_5` | started 2026-11-24 13:20 | checkpoint `models\checkpoints\resnet50_cnn_full_5.pt`
Command: `scripts/train_cnn.py --data data/processed/garment_train2000 --epochs 4`

Data: `data\processed\garment_train2000` | classes: 10 | train 20000 / val 3000 / test 3000
Settings: phase 1 3 epochs (head only), phase 2 4 epochs (unfreeze layer3), batch 32, lr head 0.001 / backbone 0.0001, weight decay 0.0001, label smoothing 0.1, pad to square True

## Training history (validation set)

| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.613 | 53.6% | 60.3% | 59.5% | 76.1% | 75.0% | 393 |
| 1 | 2 | 1.4551 | 59.7% | 61.8% | 61.2% | 77.8% | 75.2% | 335 |
| 1 | 3 | 1.4234 | 61.1% | 62.7% | 62.0% | 78.3% | 76.1% | 334 |
| 2 | 1 | 1.3992 | 62.2% | 67.8% | 67.0% | 82.4% | 79.7% | 575 |
| 2 | 2 | 1.196 | 71.4% | 69.4% | 68.9% | 83.6% | 81.8% | 572 |
| 2 | 3 **(best)** | 1.0428 | 78.0% | 69.7% | 69.3% | 83.6% | 81.7% | 575 |
| 2 | 4 | 0.9246 | 83.6% | 69.5% | 69.2% | 83.9% | 81.9% | 576 |

## Test set (best weights only: phase 2, epoch 3)

| Metric | Value |
|---|---|
| Accuracy | 67.7% |
| Macro-F1 | 67.5% |
| Top-2 accuracy | 81.9% |
| Group-level accuracy | 80.0% |
| Prediction time, GPU | 9.4 ms / image |
| Prediction time, CPU | 80.2 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.698     0.600     0.645       300
    cardigan      0.553     0.577     0.564       300
        coat      0.698     0.853     0.768       300
       dress      0.837     0.927     0.880       300
      hoodie      0.783     0.673     0.724       300
      jacket      0.650     0.507     0.569       300
shirt_blouse      0.585     0.640     0.611       300
     sweater      0.558     0.590     0.574       300
     t_shirt      0.675     0.650     0.662       300
    tank_top      0.747     0.757     0.752       300

    accuracy                          0.677      3000
   macro avg      0.678     0.677     0.675      3000
weighted avg      0.678     0.677     0.675      3000

```

Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes).
