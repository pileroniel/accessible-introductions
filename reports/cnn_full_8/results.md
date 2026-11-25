# CNN results (cnn_full_8): ResNet-50 pretrained on ImageNet, fine-tuned

Run: `cnn_full_8` | started 2026-11-24 19:46 | checkpoint `models\checkpoints\resnet50_cnn_full_8.pt`
Command: `scripts/train_cnn.py --data data/processed/garment_train2000 --epochs 4 --unfreeze all`

Data: `data\processed\garment_train2000` | classes: 10 | train 20000 / val 3000 / test 3000
Settings: phase 1 3 epochs (head only), phase 2 4 epochs (unfreeze all), batch 32, lr head 0.001 / backbone 0.0001, weight decay 0.0001, label smoothing 0.1, pad to square True, precision float16 (AMP)

## Training history (validation set)

| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.613 | 53.6% | 60.3% | 59.5% | 76.1% | 75.0% | 393 |
| 1 | 2 | 1.4551 | 59.7% | 61.8% | 61.2% | 77.8% | 75.2% | 333 |
| 1 | 3 | 1.4234 | 61.1% | 62.7% | 62.0% | 78.3% | 76.1% | 333 |
| 2 | 1 | 1.427 | 60.9% | 65.7% | 65.0% | 81.2% | 78.7% | 880 |
| 2 | 2 | 1.2298 | 70.0% | 68.6% | 68.2% | 84.0% | 80.6% | 884 |
| 2 | 3 | 1.0915 | 75.7% | 69.8% | 69.4% | 83.9% | 81.4% | 878 |
| 2 | 4 **(best)** | 0.9693 | 81.2% | 69.9% | 69.6% | 84.3% | 81.8% | 878 |

## Test set (best weights only: phase 2, epoch 4)

| Metric | Value |
|---|---|
| Accuracy | 68.5% |
| Macro-F1 | 68.1% |
| Top-2 accuracy | 82.4% |
| Group-level accuracy | 80.8% |
| Prediction time, GPU | 9.9 ms / image |
| Prediction time, CPU | 95.1 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.668     0.623     0.645       300
    cardigan      0.642     0.557     0.596       300
        coat      0.704     0.863     0.775       300
       dress      0.827     0.943     0.882       300
      hoodie      0.746     0.697     0.721       300
      jacket      0.601     0.543     0.571       300
shirt_blouse      0.600     0.630     0.615       300
     sweater      0.576     0.603     0.590       300
     t_shirt      0.697     0.620     0.656       300
    tank_top      0.759     0.767     0.763       300

    accuracy                          0.685      3000
   macro avg      0.682     0.685     0.681      3000
weighted avg      0.682     0.685     0.681      3000

```

Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes).
