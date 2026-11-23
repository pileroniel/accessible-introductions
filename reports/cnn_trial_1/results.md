# CNN results: ResNet-50 pretrained on ImageNet, fine-tuned

Data: `data\interim\garment_trial` | classes: 10 | train 2000 / val 500 / test 500
Settings: phase 1 2 epochs (head only), phase 2 2 epochs (unfreeze layer3), batch 32, lr head 0.001 / backbone 0.0001, weight decay 0.0001, label smoothing 0.1, pad to square True

## Training history (validation set)

| Phase | Epoch | Train loss | Train acc | Val acc | Val macro-F1 | Val top-2 | Val group-level | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.0462 | 37.6% | 48.8% | 46.5% | 64.8% | 71.2% | 119 |
| 1 | 2 | 1.7297 | 54.9% | 50.8% | 49.8% | 68.4% | 71.0% | 32 |
| 2 | 1 | 1.6253 | 50.4% | 55.0% | 53.7% | 72.4% | 73.4% | 67 |
| 2 | 2 **(best)** | 1.2328 | 70.3% | 56.8% | 56.3% | 73.8% | 73.8% | 56 |

## Test set (best weights only: phase 2, epoch 2)

| Metric | Value |
|---|---|
| Accuracy | 57.4% |
| Macro-F1 | 57.0% |
| Top-2 accuracy | 75.4% |
| Group-level accuracy | 73.6% |
| Prediction time, GPU | 16.8 ms / image |
| Prediction time, CPU | 101.7 ms / image |

## Per-class results (test)

```
              precision    recall  f1-score   support

      blazer      0.595     0.500     0.543        50
    cardigan      0.500     0.340     0.405        50
        coat      0.571     0.720     0.637        50
       dress      0.776     0.900     0.833        50
      hoodie      0.659     0.540     0.593        50
      jacket      0.365     0.460     0.407        50
shirt_blouse      0.412     0.420     0.416        50
     sweater      0.500     0.560     0.528        50
     t_shirt      0.676     0.500     0.575        50
    tank_top      0.727     0.800     0.762        50

    accuracy                          0.574       500
   macro avg      0.578     0.574     0.570       500
weighted avg      0.578     0.574     0.570       500

```

Figures: `curves.png`, `confusion_matrix_test.png`, `misclassified_test.jpg` (40 random test mistakes).
